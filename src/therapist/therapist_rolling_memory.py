import logging
from typing import Dict, Any, List

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from ..utils import get_llm_model
from ..utils.formatting import format_conversation_history
from ..utils.tracing import enrich_span, traced_span
from ..utils.store import THERAPIST_MEMORY_NS
from .therapist_config import _config, API_MAX_RETRIES, CONTEXT_WINDOW
from .therapist_dtos import TherapistRollingMemoryOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
<role>
You are the therapist's clinical memory consolidator in a Motivational Enhancement Therapy simulation.
Your task is to update the therapist's long-term memory record by extracting and distilling critical clinical signals from the recent conversation window.
</role>

<task>
Read the <recent_transcript> and <prior_memory> provided in the user message. For each field below, apply the stated update rule.

**ACCUMULATIVE FIELDS** — carry forward everything from prior memory, then add/update from the transcript. Do not repeat items already listed; only emit new additions or modifications:
- `biographical_facts`: Specific, verifiable declarative facts about the patient's life (names, dates, demographics).
- `core_values`: Deeply held values the patient expresses (e.g., 'being a good parent', 'independence').

**REPLACED (CONSOLIDATED) FIELDS** — fully overwrite with the latest distilled assessment:
- `target_behavior`: The specific behavior targeted for change. Evolve this from vague to specific as the patient clarifies it.
- `darn_change_talk`: Current active preparatory change talk (Desire, Ability, Reason, Need). Distill to the top 3-5 most potent, recent statements to prevent list bloat. Replace older, weaker statements rather than just appending.
- `cat_change_talk`: Current active mobilizing change talk (Commitment, Activation, Taking Steps). Distill to the top 3-5 most potent, recent statements to prevent list bloat. Replace older, weaker statements rather than just appending.
- `sustain_talk_themes`: Current active resistance or reasons the patient gives *not* to change. Limit to the top 3-5 core themes to prevent list bloat. Replace older or resolved themes rather than just appending.
- `change_plan`: Agreed upon steps and their execution status. 
- `current_clinical_focus`: The overarching clinical priority for the current moment or next session.

- `memory_update_reasoning`: Explicit rationale for every change made. Required for evaluation.
</task>

<output_format>
Respond ONLY with a valid JSON object matching the requested schema. No markdown outside the JSON.
All angle-bracket values below are PLACEHOLDERS — replace every one with your own generated content.

{
  "memory_update_reasoning": "<Explicit rationale for every change made to this memory record.>",
  "biographical_facts": ["<Hard facts about the patient's life (Accumulative)>"],
  "core_values": ["<Deeply held values the patient expresses (Accumulative)>"],
  "target_behavior": "<The specific behavior targeted for change (Consolidate)>",
  "darn_change_talk": ["<Distilled instances of preparatory change talk (Consolidate)>"],
  "cat_change_talk": ["<Distilled instances of mobilizing change talk (Consolidate)>"],
  "sustain_talk_themes": ["<Current active resistance or reasons not to change (Consolidate)>"],
  "change_plan": [
    {
      "intention": "<What the patient said they would do>",
      "attempt_status": "<not_yet_attempted | in_progress | attempted_succeeded | attempted_failed | abandoned>",
      "outcome_note": "<One-line factual note on outcome, or empty string>"
    }
  ],
  "current_clinical_focus": "<Overarching clinical priority for the current moment or next session (Consolidate)>"
}
</output_format>
"""

def _build_user_message(
    conversation_text: str,
    prior_biographical: List[str],
    prior_values: List[str],
    prior_target: str,
    prior_darn: List[str],
    prior_cat: List[str],
    prior_sustain: List[str],
    prior_plan: List[Dict[str, Any]],
    prior_focus: str,
) -> str:
    def _bullets(items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "(none)"

    def _plan_bullets(items: List[Dict[str, Any]]) -> str:
        if not items: return "(none)"
        return "\n".join(f"- {item.get('intention', '')} (Status: {item.get('attempt_status', '')}) - {item.get('outcome_note', '')}" for item in items)

    return f"""\
<recent_transcript>
{conversation_text}
</recent_transcript>

<prior_memory>
Biographical Facts (accumulate):
{_bullets(prior_biographical)}

Core Values (accumulate):
{_bullets(prior_values)}

Target Behavior (evolving):
{prior_target}

DARN Change Talk (evolving):
{_bullets(prior_darn)}

CAT Change Talk (evolving):
{_bullets(prior_cat)}

Sustain Talk Themes (evolving):
{_bullets(prior_sustain)}

Change Plan (evolving):
{_plan_bullets(prior_plan)}

Current Clinical Focus (evolving):
{prior_focus}
</prior_memory>

Update the memory based on the recent transcript. Do not fabricate information.
"""

def _create_structured_chain(temperature: float):
    primary = (
        get_llm_model(
            temperature=temperature,
            provider=_config.PROVIDER,
            model_name=_config.MODEL_NAME,
            provider_preferences=_config.OPENROUTER_PROVIDER_PREFS,
        )
        .with_structured_output(TherapistRollingMemoryOutput, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    fallback = (
        get_llm_model(
            temperature=temperature,
            provider=_config.PROVIDER,
            model_name=_config.FALLBACK_MODEL_NAME,
            provider_preferences=_config.OPENROUTER_PROVIDER_PREFS,
        )
        .with_structured_output(TherapistRollingMemoryOutput, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    return primary.with_fallbacks([fallback])

@traced_span("therapist_rolling_memory")
def therapist_rolling_memory_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    configurable = config.get("configurable", {})
    session_number = configurable.get("current_session_number", state.get("session_number", 1))

    logger.info("therapist_rolling_memory_node: updating memory for session %s", session_number)

    # Process turns since last update
    max_messages = CONTEXT_WINDOW * 2
    messages = state.get("messages", [])
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    conversation_text = format_conversation_history(recent_messages)    
    if not conversation_text.strip():
        logger.info("therapist_rolling_memory_node: no new messages to consolidate.")
        return {}

    store = runtime.store
    
    # Using specific keys under THERAPIST_MEMORY_NS
    biographical_item = store.get(THERAPIST_MEMORY_NS, "biographical_facts")
    values_item = store.get(THERAPIST_MEMORY_NS, "core_values")
    target_item = store.get(THERAPIST_MEMORY_NS, "target_behavior")
    darn_item = store.get(THERAPIST_MEMORY_NS, "darn_change_talk")
    cat_item = store.get(THERAPIST_MEMORY_NS, "cat_change_talk")
    sustain_item = store.get(THERAPIST_MEMORY_NS, "sustain_talk_themes")
    plan_item = store.get(THERAPIST_MEMORY_NS, "change_plan")
    focus_item = store.get(THERAPIST_MEMORY_NS, "current_clinical_focus")

    prior_biographical = list(biographical_item.value.get("items", [])) if biographical_item else []
    prior_values = list(values_item.value.get("items", [])) if values_item else []
    prior_target = str(target_item.value.get("value", "")) if target_item else ""
    prior_darn = list(darn_item.value.get("items", [])) if darn_item else []
    prior_cat = list(cat_item.value.get("items", [])) if cat_item else []
    prior_sustain = list(sustain_item.value.get("items", [])) if sustain_item else []
    prior_plan = list(plan_item.value.get("items", [])) if plan_item else []
    prior_focus = str(focus_item.value.get("value", "")) if focus_item else ""

    user_message = _build_user_message(
        conversation_text=conversation_text,
        prior_biographical=prior_biographical,
        prior_values=prior_values,
        prior_target=prior_target,
        prior_darn=prior_darn,
        prior_cat=prior_cat,
        prior_sustain=prior_sustain,
        prior_plan=prior_plan,
        prior_focus=prior_focus,
    )

    temperature = _config.get_temperature()
    chain = _create_structured_chain(temperature)
    result: TherapistRollingMemoryOutput = chain.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ],
        config=config,
    )

    # Write back to store
    updated_plans = [item.model_dump() for item in result.change_plan]
    
    store.put(THERAPIST_MEMORY_NS, "biographical_facts", {"items": result.biographical_facts})
    store.put(THERAPIST_MEMORY_NS, "core_values", {"items": result.core_values})
    store.put(THERAPIST_MEMORY_NS, "target_behavior", {"value": result.target_behavior})
    store.put(THERAPIST_MEMORY_NS, "darn_change_talk", {"items": result.darn_change_talk})
    store.put(THERAPIST_MEMORY_NS, "cat_change_talk", {"items": result.cat_change_talk})
    store.put(THERAPIST_MEMORY_NS, "sustain_talk_themes", {"items": result.sustain_talk_themes})
    store.put(THERAPIST_MEMORY_NS, "change_plan", {"items": updated_plans})
    store.put(THERAPIST_MEMORY_NS, "current_clinical_focus", {"value": result.current_clinical_focus})

    logger.info(
        "therapist_rolling_memory_node: memory updated."
    )

    enrich_span(
        biographical_facts_count=len(result.biographical_facts),
        core_values_count=len(result.core_values),
        darn_change_talk_count=len(result.darn_change_talk),
        cat_change_talk_count=len(result.cat_change_talk),
        sustain_talk_themes_count=len(result.sustain_talk_themes),
        change_plan_items_count=len(updated_plans),
        current_clinical_focus=result.current_clinical_focus
    )
    
    return {"therapist_last_rolling_memory_update_turn": state.get("turn_count")}
