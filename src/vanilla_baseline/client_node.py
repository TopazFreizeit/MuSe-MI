"""
Vanilla Baseline Patient Node.

Implements a static persona in-context role-play patient following the
RoleLLM paradigm (Wang et al., ACL 2024). The patient is conditioned on a
frozen profile read from the shared ConsistentClient profiles JSONL and the
raw conversation history for the current session. No latent variable updates,
no cognitive action selection, and no belief/persona depletion occur. The
only cross-session information is a chronological rolling summary injected
from the store.
"""
import logging
from typing import Dict, Any, Type

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.utils import get_llm_model
from src.utils.tracing import enrich_span, traced_span
from src.utils.formatting import format_speaker_utterance
from src.utils.constants import PATIENT_PREFIX
from src.utils.store import PATIENT_MEMORY_NS
from src.patient.patient_dtos import PatientProfileData
import src.patient.patient_config as patient_config

logger = logging.getLogger(__name__)


class VanillaClientOutput(BaseModel):
    reasoning: str = Field(
        description=(
            "Brief internal reasoning: what emotional state you are in, what the counselor "
            "just said, and why you are responding the way you are. 1–2 sentences."
        )
    )
    response: str = Field(
        description=(
            "Your exact spoken words as the client. Natural, colloquial, 1–3 sentences. "
            "No therapy-speak. Do not resolve your ambivalence prematurely."
        )
    )


def _build_model_with_fallback(output_schema: Type[BaseModel], temperature: float):
    """Build a structured-output model chain (primary + fallback) for the given schema."""
    _primary_base = get_llm_model(
        temperature=temperature,
        provider=patient_config._config.PROVIDER,
        model_name=patient_config._config.MODEL_NAME,
        provider_preferences=patient_config._config.OPENROUTER_PROVIDER_PREFS,
    )
    _fallback_base = get_llm_model(
        temperature=temperature,
        provider=patient_config._config.PROVIDER,
        model_name=patient_config._config.FALLBACK_MODEL_NAME,
        provider_preferences=patient_config._config.OPENROUTER_PROVIDER_PREFS,
    )
    if _primary_base is None:
        raise RuntimeError("_build_model_with_fallback: get_llm_model returned None for primary model.")
    if _fallback_base is None:
        raise RuntimeError("_build_model_with_fallback: get_llm_model returned None for fallback model.")
    primary = (
        _primary_base
        .with_structured_output(output_schema, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=patient_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    fallback = (
        _fallback_base
        .with_structured_output(output_schema, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=patient_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    return primary.with_fallbacks([fallback])


def _build_system_prompt(profile: PatientProfileData, rolling_summary: str) -> str:
    """
    Construct the static system prompt from PatientProfileData.

    The persona is fixed for the entire simulation run. The only dynamic
    component is the inter-session rolling summary injected here at the start
    of each new session.
    """
    personas_block = "\n".join(f"- {p}" for p in profile.personas)
    beliefs_block = "\n".join(f"- {b}" for b in profile.beliefs)
    motivation_block = "\n".join(f"- {m}" for m in profile.motivation)
    plans_block = "\n".join(f"- {p}" for p in profile.acceptable_plans)

    memory_section = (
        f"\n## Memory from Previous Sessions\n{rolling_summary}\n"
        if rolling_summary
        else ""
    )

    return f"""You are roleplaying as a client in a motivational interviewing counseling session.

## Your Situation
- **Behavior you are engaging in**: {profile.behavior}
- **Change goal the counselor is working toward with you**: {profile.topic}

## Your Personas
{personas_block}

## Your Core Beliefs
{beliefs_block}

## What Motivates You
{motivation_block}

## Plans You Would Consider
{plans_block}
{memory_section}
## Response Guidelines
- Stay fully in character as the client at all times.
- Your responses must be natural, colloquial, and no longer than 2–3 sentences.
- Do NOT use therapy language or sound overly reflective. Be realistic and human.
- Do NOT break character or refer to yourself as "the client."
- Your ambivalence and resistance are genuine — they shift only when the conversation genuinely earns it.

## Output Format
You MUST respond with a valid JSON object containing exactly two keys:
{{"reasoning": "<your brief internal reasoning>", "response": "<your spoken words as the client>"}}
Do NOT include any other keys or any text outside the JSON object.
"""


@traced_span("vanilla_client")
def vanilla_client_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    """
    Vanilla baseline patient node.

    Reads a frozen PatientProfileData from config and generates a response
    conditioned solely on the static persona prompt and the raw conversation
    history for the current session. An optional chronological rolling summary
    from the previous session is injected when available.

    No cognitive state updates, action selection, or latent variable tracking
    are performed.
    """
    logger.info("vanilla_client_node invoked")

    configurable = config.get("configurable", {})
    patient_profile_data_dict = configurable.get("patient_profile_data")
    if not patient_profile_data_dict:
        raise ValueError("vanilla_client_node: patient_profile_data missing from configurable.")
    profile = PatientProfileData.model_validate(patient_profile_data_dict)

    # Read rolling summary written by vanilla_client_memory from the previous session.
    store = runtime.store
    if store is None:
        raise RuntimeError("vanilla_client_node: runtime.store is None — store was not passed to the graph.")
    memory_item = store.get(PATIENT_MEMORY_NS, "vanilla_memory")
    rolling_summary = memory_item.value.get("summary", "") if memory_item else ""

    enrich_span(
        node_name="vanilla_client",
        description="Generate vanilla baseline patient response",
        session_number=state.get("session_number", 1),
    )

    system_prompt = _build_system_prompt(profile, rolling_summary)

    # Append the full conversation history — no windowing for this baseline.
    all_messages = state.get("messages", [])

    lc_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    for msg in all_messages:
        name = getattr(msg, "name", None)
        if name == "Therapist":
            lc_messages.append(HumanMessage(content=f"Counselor: {msg.content}"))
        elif name == "Patient":
            lc_messages.append(AIMessage(content=f"Client: {msg.content}"))

    if not all_messages or getattr(all_messages[-1], "name", None) != "Therapist":
        logger.warning("vanilla_client_node: last message is not from the therapist — check graph routing.")

    temperature = patient_config._config.get_temperature()
    chain = _build_model_with_fallback(VanillaClientOutput, temperature)
    result: VanillaClientOutput = chain.invoke(lc_messages, config=config)  # type: ignore[assignment]

    if not getattr(result, "response", None):
        raise ValueError(f"vanilla_client_node: structured output missing 'response' field: {result}")

    reasoning = result.reasoning
    response = result.response

    patient_message = AIMessage(content=response, name="Patient")
    summary_line = format_speaker_utterance(response, PATIENT_PREFIX)
    logger.info(f"\n\nVanilla Client reasoning:\n{reasoning}\n\nVanilla Client response:\n\n{summary_line}\n\n")

    current_turns = state.get("current_session_turns", [])
    turn_record = {
        "turn_number": len(current_turns) + 1,
        "speaker": "patient",
        "volley": response,
        "reasoning": reasoning,
    }

    return {
        "messages": state.get("messages", []) + [patient_message],
        "current_speaker": "patient",
        "turn_count": state.get("turn_count", 0) + 1,
        "current_session_turns": current_turns + [turn_record],
    }
