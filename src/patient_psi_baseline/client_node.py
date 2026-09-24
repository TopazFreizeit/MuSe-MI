"""
Patient-Ψ Baseline Client Node.

Implements the Patient-Ψ simulated patient architecture (Wang et al., ACL 2024).
Conditioned on Judith Beck's Cognitive Conceptualization Diagram (CCD) loaded from
the pre-generated profile repository in ``patient_psi_profiles/{idx}.json``, paired
with 6 configurable conversational styles ('plain', 'upset', 'verbose', 'reserved',
'tangent', 'pleasing').
"""
import logging
import os
from typing import Dict, Any, Type, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel

from src.utils import get_llm_model
from src.utils.tracing import enrich_span, traced_span
from src.utils.formatting import format_speaker_utterance
from src.utils.constants import PATIENT_PREFIX
from src.utils.store import PATIENT_MEMORY_NS
from src.patient.patient_dtos import PatientProfileData
import src.patient.patient_config as patient_config

from .dtos import PatientPsiOutput
from .profile_loader import load_patient_psi_profile, build_patient_psi_system_prompt

logger = logging.getLogger(__name__)


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
        raise RuntimeError("patient_psi_baseline: get_llm_model returned None for primary model.")
    if _fallback_base is None:
        raise RuntimeError("patient_psi_baseline: get_llm_model returned None for fallback model.")

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


@traced_span("patient_psi_client")
def patient_psi_client_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """
    Patient-Ψ simulated client node in the single-session graph.
    """
    logger.info("patient_psi_client_node invoked")

    configurable = config.get("configurable", {})
    patient_profile_data_dict = configurable.get("patient_profile_data") or state.get("patient_profile_data")
    if not patient_profile_data_dict:
        raise ValueError("patient_psi_client_node: patient_profile_data missing from configurable or state.")
    profile = PatientProfileData.model_validate(patient_profile_data_dict)

    store = runtime.store
    if store is None:
        raise RuntimeError("patient_psi_client_node: runtime.store is None — store was not passed to graph.")

    # Load cross-session memory (rolling summary from previous session)
    memory_item = store.get(PATIENT_MEMORY_NS, "patient_psi_memory")
    rolling_summary = memory_item.value.get("summary", "") if memory_item and memory_item.value else ""

    # Load pre-generated CCD profile
    ccd = load_patient_psi_profile(profile.idx)

    # Resolve style from environment or configurable
    style_name = configurable.get("patient_psi_style") or os.getenv("PATIENT_PSI_STYLE", "plain")

    session_id = state.get("session_number", 1)

    enrich_span(
        node_name="patient_psi_client",
        description="Generate Patient-Ψ baseline response",
        session_number=session_id,
        patient_psi_style=style_name,
    )

    system_prompt = build_patient_psi_system_prompt(
        ccd=ccd,
        style_name=style_name,
        rolling_summary=rolling_summary,
    )

    all_messages = state.get("messages", [])
    lc_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    for msg in all_messages:
        name = getattr(msg, "name", None)
        content_val = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        cleaned = content_val.strip()
        if name == "Therapist" or isinstance(msg, HumanMessage):
            lc_messages.append(HumanMessage(content=cleaned))
        elif name == "Patient" or isinstance(msg, AIMessage):
            lc_messages.append(AIMessage(content=cleaned))

    if not all_messages or getattr(all_messages[-1], "name", None) != "Therapist":
        logger.warning("patient_psi_client_node: last message is not from therapist — check graph routing.")

    temperature = patient_config._config.get_temperature()
    chain = _build_model_with_fallback(PatientPsiOutput, temperature)
    result: PatientPsiOutput = chain.invoke(lc_messages, config=config)  # type: ignore[assignment]

    if not getattr(result, "response", None):
        raise ValueError(f"patient_psi_client_node: structured output missing 'response' field: {result}")

    reasoning = result.reasoning.strip()
    response = result.response.strip()

    patient_message = AIMessage(content=response, name="Patient")
    summary_line = format_speaker_utterance(response, PATIENT_PREFIX)
    logger.info(summary_line)

    current_turns = list(state.get("current_session_turns", []))
    turn_record = {
        "turn_number": len(current_turns) + 1,
        "speaker": "patient",
        "volley": response,
        "reasoning": reasoning,
        "patient_psi_style": style_name,
    }

    return {
        "messages": state.get("messages", []) + [patient_message],
        "current_speaker": "patient",
        "turn_count": state.get("turn_count", 0) + 1,
        "current_session_turns": current_turns + [turn_record],
    }
