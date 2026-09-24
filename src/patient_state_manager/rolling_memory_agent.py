"""
Rolling Memory Agent.

Runs every 12 turns and at the end of the session to consolidate macro shifts 
and update the long-term context fields (facts, motivations, concerns, etc.).
Replaces the former end-of-session consolidation.
"""
import json
import logging
from typing import Dict, Any, List

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from ..utils import get_llm_model
from ..utils.formatting import format_conversation_history
from ..utils.tracing import enrich_span, traced_span
from ..utils.store import (
    PATIENT_MEMORY_NS,
)
from .patient_state_manager_config import (
    PROVIDER,
    MODEL_NAME,
    FALLBACK_MODEL_NAME,
    TEMPERATURE,
    API_MAX_RETRIES,
    OPENROUTER_PROVIDER_PREFS,
)
from .patient_state_manager_dtos import (
    RollingMemoryOutput,
    RollingMemoryWithoutMacroShiftOutput,
)
from .cognitive_simulator import DeterministicCognitiveSimulator
from ..graphs.graph_config import LatentTrajectoryEntry
from ..utils.config import (
    ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES,
    DISABLE_COGNITIVE_INTERPRETER_IMPACT,
    DISABLE_PATIENT_MEMORY,
)

logger = logging.getLogger(__name__)

_MACRO_SHIFT_TASK = """\
**MACRO SHIFT** — classify the overall psychological momentum visible across the window:
- `macro_shift.anger`: Did the patient's hostility or defensiveness noticeably increase or decrease across this window?
- `macro_shift.self_efficacy`: Did the patient's confidence in their ability to change noticeably improve or erode across this window?
- `macro_shift.problem_recognition`: Did the patient's felt conflict between their behavior and their values noticeably sharpen or dull?
- `macro_shift.motivational_readiness`: Did the patient's overall readiness to change advance, retreat, or hold steady?

Calibration: most 12-turn windows produce "No Change" on all four macro-shift variables. "Minor Progress" or "Minor Regression" requires clear patient-language evidence (not just therapist effort). "Major" shifts require a sustained pattern across the window, not a single turn.
"""

_MACRO_SHIFT_OUTPUT = """,\
    "macro_shift": {
        "anger": {
            "reasoning": "Why anger shifted (or did not) across this window.",
            "semantic_impact": "Major Regression | Minor Regression | No Change | Minor Progress | Major Progress"
        },
        "self_efficacy": {
            "reasoning": "Why self-efficacy shifted (or did not) across this window.",
            "semantic_impact": "Major Regression | Minor Regression | No Change | Minor Progress | Major Progress"
        },
        "problem_recognition": {
            "reasoning": "Why problem recognition shifted (or did not) across this window.",
            "semantic_impact": "Major Regression | Minor Regression | No Change | Minor Progress | Major Progress"
        },
        "motivational_readiness": {
            "reasoning": "Why motivational readiness shifted (or did not) across this window.",
            "semantic_impact": "Major Regression | Minor Regression | No Change | Minor Progress | Major Progress"
        }
    }"""

_SYSTEM_PROMPT_TEMPLATE = """\
<role>
You are the patient's internal memory consolidator in a Motivational Enhancement Therapy simulation.
Your task is to update the patient's long-term memory record by extracting durable facts, plans, and emotional threads{psychological_shift_scope} from the recent conversation window.
</role>

<task>
Read the <recent_transcript> and <prior_memory> provided in the user message. For each field below, apply the stated update rule.

**ACCUMULATIVE FIELDS** — carry forward everything from prior memory, then add/update from the transcript. Do not repeat items already listed in <prior_memory>; only emit new additions or modifications:
- `facts`: Specific, verbatim-grounded declarative anchors (names, dates, quantities, stated facts). Each element is a plain string sentence (e.g., "Patient works as an administrative assistant"). Do NOT drop an existing fact unless it is explicitly contradicted in the transcript. Output as a flat JSON array of strings — never as a dict or object.
- `agreed_change_plans`: Action plans or concrete steps the patient has agreed to. Preserve existing plans and update their `attempt_status` based on any new evidence. Add new plans if agreed upon in this window. Status must be one of: `not_yet_attempted`, `in_progress`, `attempted_succeeded`, `attempted_failed`, `abandoned`.
- `implicit_threads`: Topics the patient briefly mentioned or alluded to but did not fully explore — the unfinished emotional business. Accumulate; do not drop threads unless the topic was explicitly resolved.

**REPLACED FIELDS** — fully overwrite with the latest assessment:
- `motivations`: The patient's active Change Talk drivers as of this window (the psychological "pros of change"). Limit to the top 3-5 most potent drivers to prevent list bloat. Replace older, weaker statements rather than just appending. Distill from what the patient has expressed, not what the therapist has said.
- `concerns`: The patient's active Sustain Talk drivers (the psychological "cons of change"). Limit to the top 3-5 core concerns to prevent list bloat. Replace older or resolved concerns rather than just appending. Must be grounded in the patient's language, not inferred from demographics.
- `persona_and_stylistic_baseline`: A 1-2 sentence non-declarative description of the patient's current baseline communication style and emotional register (e.g., "Guarded, uses self-deprecating humor to deflect; tends to minimize with 'I don't know'"). Update if the style has noticeably shifted.
- `memory_update_reasoning`: Explicit rationale for every change made — which facts were added, which threads were surfaced, why motivations/concerns were updated. Required for fidelity evaluation.

{macro_shift_task}
</task>

<output_format>
Respond ONLY with a valid JSON object matching the exact structure below. No markdown, no extra text.

{{
  "memory_update_reasoning": "Explicit rationale for every change made to this memory record.",
  "facts": ["Plain string sentence.", "Another plain string sentence."],
  "agreed_change_plans": [
    {{
      "intention": "What the patient said they would do (verbatim or close paraphrase).",
      "attempt_status": "not_yet_attempted | in_progress | attempted_succeeded | attempted_failed | abandoned",
      "outcome_note": "One-line factual note on outcome, or empty string if not_yet_attempted."
    }}
  ],
  "implicit_threads": ["A deferred topic the patient hinted at.", "Another unresolved thread."],
  "motivations": ["A Change Talk driver expressed by the patient."],
  "concerns": ["A Sustain Talk driver expressed by the patient."],
    "persona_and_stylistic_baseline": "1-2 sentence description of the patient's baseline communication style."{macro_shift_output}
}}
</output_format>
"""


def _build_system_prompt() -> str:
        """Build the rolling-memory prompt for the active cognition configuration."""
        return _SYSTEM_PROMPT_TEMPLATE.format(
            psychological_shift_scope=(
                ""
                if DISABLE_COGNITIVE_INTERPRETER_IMPACT
                else ", and macro-level psychological shifts"
            ),
                macro_shift_task=(
                        "" if DISABLE_COGNITIVE_INTERPRETER_IMPACT else _MACRO_SHIFT_TASK
                ),
                macro_shift_output=(
                        "" if DISABLE_COGNITIVE_INTERPRETER_IMPACT else _MACRO_SHIFT_OUTPUT
                ),
        )

def _build_user_message(
    conversation_text: str,
    prior_facts: List[str],
    prior_plans: List[Dict[str, Any]],
    prior_threads: List[str],
    prior_motivations: List[str],
    prior_concerns: List[str],
) -> str:
    def _bullets(items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "(none)"
        
    def _plan_bullets(items: List[Dict[str, Any]]) -> str:
        if not items: return "(none)"
        return "\n".join(f"- {item['intention']} (Status: {item['attempt_status']}) - {item.get('outcome_note', '')}" for item in items)

    return f"""\
<recent_transcript>
{conversation_text}
</recent_transcript>

<prior_memory>
Facts (accumulate/update):
{_bullets(prior_facts)}

Agreed Change Plans (accumulate/update):
{_plan_bullets(prior_plans)}

Implicit Threads (accumulate):
{_bullets(prior_threads)}

Motivations (replace):
{_bullets(prior_motivations)}

Concerns (replace):
{_bullets(prior_concerns)}
</prior_memory>

Update the memory based on the recent transcript. Do not fabricate information.
"""

def _create_structured_chain():
    """Create a structured-output chain with retry + primary → fallback routing."""
    output_type = (
        RollingMemoryWithoutMacroShiftOutput
        if DISABLE_COGNITIVE_INTERPRETER_IMPACT
        else RollingMemoryOutput
    )
    primary = (
        get_llm_model(
            temperature=TEMPERATURE,
            provider=PROVIDER,
            model_name=MODEL_NAME,
            provider_preferences=OPENROUTER_PROVIDER_PREFS,
        )
        .with_structured_output(output_type, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    fallback = (
        get_llm_model(
            temperature=TEMPERATURE,
            provider=PROVIDER,
            model_name=FALLBACK_MODEL_NAME,
            provider_preferences=OPENROUTER_PROVIDER_PREFS,
        )
        .with_structured_output(output_type, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    return primary.with_fallbacks([fallback])

@traced_span("patient_rolling_memory")
def rolling_memory_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    """
    Consolidates the recent conversational turns (up to 12) into the rolling memory.
    """
    if DISABLE_PATIENT_MEMORY:
        logger.info(
            "Ablation Mode active: DISABLE_PATIENT_MEMORY (rolling memory node bypassed)"
        )
        return {}

    configurable = config.get("configurable", {})
    session_number = configurable.get("current_session_number", state.get("session_number", 1))

    logger.info("rolling_memory_node: updating memory for session %s", session_number)
    if DISABLE_COGNITIVE_INTERPRETER_IMPACT:
        logger.info(
            "Ablation Mode active: DISABLE_COGNITIVE_INTERPRETER_IMPACT "
            "(rolling memory uses the memory-only prompt and output schema)"
        )

    # Process only the turns since the last memory update.
    from ..patient.patient_config import PATIENT_CONTEXT_WINDOW_TURNS

    # For simplicity, we grab the last N turns (approx 2 messages per turn).
    max_messages = PATIENT_CONTEXT_WINDOW_TURNS * 2
    messages = state.get("messages", [])
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    conversation_text = format_conversation_history(recent_messages)    
    if not conversation_text.strip():
        logger.info("rolling_memory_node: no new messages to consolidate.")
        return {}

    # Read prior memory items
    store = runtime.store
    facts_item = store.get(PATIENT_MEMORY_NS, "facts")
    plans_item = store.get(PATIENT_MEMORY_NS, "agreed_change_plans")
    threads_item = store.get(PATIENT_MEMORY_NS, "implicit_threads")
    motivations_item = store.get(PATIENT_MEMORY_NS, "motivations")
    concerns_item = store.get(PATIENT_MEMORY_NS, "concerns")

    current_facts = list(facts_item.value.get("items", [])) if facts_item else []
    current_plans = list(plans_item.value.get("items", [])) if plans_item else []
    current_threads = list(threads_item.value.get("items", [])) if threads_item else []
    current_motivations = list(motivations_item.value.get("items", [])) if motivations_item else []
    current_concerns = list(concerns_item.value.get("items", [])) if concerns_item else []

    user_message = _build_user_message(
        conversation_text=conversation_text,
        prior_facts=current_facts,
        prior_plans=current_plans,
        prior_threads=current_threads,
        prior_motivations=current_motivations,
        prior_concerns=current_concerns,
    )

    chain = _create_structured_chain()
    result = chain.invoke(
        [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content=user_message),
        ],
        config=config,
    )

    # Write back to store
    updated_plans = [item.model_dump() for item in result.agreed_change_plans]
    
    store.put(PATIENT_MEMORY_NS, "facts", {"items": result.facts})
    store.put(PATIENT_MEMORY_NS, "agreed_change_plans", {"items": updated_plans})
    store.put(PATIENT_MEMORY_NS, "implicit_threads", {"items": result.implicit_threads})
    
    store.put(PATIENT_MEMORY_NS, "motivations", {"items": result.motivations})
    store.put(PATIENT_MEMORY_NS, "concerns", {"items": result.concerns})
    store.put(PATIENT_MEMORY_NS, "persona_and_stylistic_baseline", {"value": result.persona_and_stylistic_baseline})

    logger.info(
        "rolling_memory_node: memory updated. "
        "facts=%d, plans=%d, threads=%d, motivations=%d, concerns=%d",
        len(result.facts),
        len(updated_plans),
        len(result.implicit_threads),
        len(result.motivations),
        len(result.concerns)
    )

    enrich_span(
        shared_facts_count=len(result.facts),
        change_plan_items_count=len(updated_plans),
        implicit_threads_count=len(result.implicit_threads),
        core_motivations_count=len(result.motivations),
        core_concerns_count=len(result.concerns),
    )

    # Apply Macro-Shift
    if (
        ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES
        or DISABLE_COGNITIVE_INTERPRETER_IMPACT
    ):
        logger.info(
            "Ablation Mode: preserving latent variables unchanged in rolling memory "
            "(static_hidden_variables=%s, disable_cognitive_interpreter_impact=%s)",
            ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES,
            DISABLE_COGNITIVE_INTERPRETER_IMPACT,
        )
        new_latent_vars = state.get("patient_latent_variables", {}).copy()
    else:
        if not isinstance(result, RollingMemoryOutput):
            raise TypeError("Rolling memory result is missing required macro shifts.")
        latent_vars = state.get("patient_latent_variables", {})
        anger = latent_vars.get("anger", 0.0)
        self_efficacy = latent_vars.get("self_efficacy", 0.0)
        problem_recognition = latent_vars.get("problem_recognition", 0.0)
        motivational_readiness = latent_vars.get("motivational_readiness", 0.0)

        new_latent_vars = latent_vars.copy()
        shift = result.macro_shift

        anger_delta = DeterministicCognitiveSimulator._get_macro_shift_delta(shift.anger.semantic_impact)
        se_delta = DeterministicCognitiveSimulator._get_macro_shift_delta(shift.self_efficacy.semantic_impact)
        pr_delta = DeterministicCognitiveSimulator._get_macro_shift_delta(shift.problem_recognition.semantic_impact)
        mr_delta = DeterministicCognitiveSimulator._get_macro_shift_delta(shift.motivational_readiness.semantic_impact)

        new_latent_vars["anger"] = DeterministicCognitiveSimulator._bounded_update(
            float(anger), anger_delta
        )
        new_latent_vars["self_efficacy"] = DeterministicCognitiveSimulator._bounded_update(
            float(self_efficacy), se_delta
        )
        new_latent_vars["problem_recognition"] = DeterministicCognitiveSimulator._bounded_update(
            float(problem_recognition), pr_delta
        )
        new_latent_vars["motivational_readiness"] = DeterministicCognitiveSimulator._bounded_update(
            float(motivational_readiness), mr_delta
        )

    output_dict: Dict[str, Any] = {
        "patient_latent_variables": new_latent_vars,
        "patient_last_rolling_memory_update_turn": state.get("turn_count")
    }

    if not (ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES or DISABLE_COGNITIVE_INTERPRETER_IMPACT):
        macro_shift_details = {
            "macro_shift": {
                "anger": shift.anger.semantic_impact.value,
                "self_efficacy": shift.self_efficacy.semantic_impact.value,
                "problem_recognition": shift.problem_recognition.semantic_impact.value,
                "motivational_readiness": shift.motivational_readiness.semantic_impact.value,
            }
        } if isinstance(result, RollingMemoryOutput) and result.macro_shift else None

        output_dict["macro_shift_details"] = macro_shift_details

        if "session_latent_trajectory" in state:
            session_num = state.get("session_number", 1)
            turn_num = state.get("turn_count", 0)
            trajectory_entry: LatentTrajectoryEntry = {
                "step_index": -1,
                "component": "memory",
                "session_number": session_num,
                "turn_number": turn_num,
                "anger": float(new_latent_vars["anger"]),
                "self_efficacy": float(new_latent_vars["self_efficacy"]),
                "problem_recognition": float(new_latent_vars["problem_recognition"]),
                "motivational_readiness": float(new_latent_vars["motivational_readiness"]),
                "details": macro_shift_details,
            }
            sess_traj = list(state.get("session_latent_trajectory", []))
            sess_traj.append(trajectory_entry)
            output_dict["session_latent_trajectory"] = sess_traj

    return output_dict
