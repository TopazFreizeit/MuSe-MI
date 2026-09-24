"""
Patient-Ψ Baseline Inter-Session Memory Pipeline Nodes.

Generates a chronological rolling summary of the completed session transcript
from the patient's perspective, stored under ``"patient_psi_memory"`` in
``PATIENT_MEMORY_NS`` and injected into the Patient-Ψ client's system prompt at
the start of the following session.
"""
import logging
from typing import Dict, Any, Type

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel

from src.utils import get_llm_model
from src.utils.formatting import format_conversation_history
from src.utils.tracing import traced_span
import src.patient.patient_config as patient_config
from src.utils.store import PATIENT_MEMORY_NS

from .dtos import PatientPsiMemoryOutput

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
        raise RuntimeError("patient_psi_memory_node: get_llm_model returned None for primary model.")
    if _fallback_base is None:
        raise RuntimeError("patient_psi_memory_node: get_llm_model returned None for fallback model.")

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


@traced_span("patient_psi_session_summary")
def patient_psi_session_summary_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """
    Summarise the completed session transcript and persist the result for the
    next session.
    """
    logger.info("patient_psi_session_summary_node invoked")

    messages = state.get("current_session", {}).get("messages", [])
    if not messages:
        logger.warning("patient_psi_session_summary_node: no messages found, skipping memory generation.")
        return {}

    conversation_text = format_conversation_history(messages)

    system_prompt = """You are tasked with summarizing the therapy session from the perspective of the Client.
Read the transcript and write a concise, one-paragraph summary focusing on:
1. What was discussed.
2. Any plans, realizations, or agreements made.
3. The Client's feelings or stance towards the Counselor and the topic at the end of the session.

OUTPUT FORMAT:
You must strictly return a valid JSON object containing exactly one key named "summary".
Do NOT include any other keys or text.
Example Format:
{
  "summary": "Your one paragraph summary here..."
}
"""

    chain = _build_model_with_fallback(PatientPsiMemoryOutput, temperature=0.3)
    result: PatientPsiMemoryOutput = chain.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Transcript:\n{conversation_text}"),
        ],
        config=config,
    )  # type: ignore[assignment]

    if not getattr(result, "summary", None):
        raise ValueError("patient_psi_session_summary_node: LLM returned empty summary.")

    store = runtime.store
    if store is None:
        raise RuntimeError("patient_psi_session_summary_node: runtime.store is None.")

    store.put(PATIENT_MEMORY_NS, "patient_psi_memory", {"summary": result.summary.strip()})
    logger.info(f"patient_psi_session_summary_node: summary stored: {result.summary[:100]}...")
    return {"patient_psi_session_summary": result.summary.strip()}


@traced_span("patient_psi_inter_sessions")
def patient_psi_inter_sessions_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """Structural placeholder node for Phoenix multi-session tracing hierarchy."""
    logger.info("patient_psi_inter_sessions_node invoked")
    return {}
