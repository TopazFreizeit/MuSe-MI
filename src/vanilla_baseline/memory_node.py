"""
Vanilla Baseline Inter-Session Memory Nodes.

Implements chronological summarisation of the completed session transcript
from a neutral perspective. The summary is stored under the key
``"vanilla_memory"`` in ``PATIENT_MEMORY_NS`` and injected into the vanilla
client's system prompt at the start of the following session.

This mirrors the inter-session memory structure used by the ConsistentClient
baseline while remaining intentionally minimal: no psychological inference,
no belief extraction, no motivational analysis.
"""
import logging
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.utils import get_llm_model
from src.utils.formatting import format_conversation_history
from src.utils.tracing import traced_span
import src.patient.patient_config as patient_config
from src.utils.store import PATIENT_MEMORY_NS

logger = logging.getLogger(__name__)


class VanillaMemoryOutput(BaseModel):
    summary: str = Field(
        description=(
            "A concise, chronological, factual paragraph summarising what was "
            "discussed in the session: topics raised, any agreements or disagreements, "
            "and where things stood at the end."
        )
    )


@traced_span("vanilla_client_memory")
def vanilla_client_memory_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """
    Summarise the completed session transcript and persist the result for the
    next session. Uses a single LLM call to produce a plain chronological
    summary — no psychological framing, no behavioral inference.
    """
    logger.info("vanilla_client_memory_node invoked")

    messages = state.get("current_session", {}).get("messages", [])
    if not messages:
        logger.warning("vanilla_client_memory_node: no messages found, skipping memory generation.")
        return {}

    conversation_text = format_conversation_history(messages)

    system_prompt = """You are tasked with summarizing the therapy session from the perspective of the Client. 
Read the transcript and write a concise, one-paragraph summary focusing on:
1. What was discussed.
2. Any plans or agreements made.
3. The Client's feelings or stance towards the Counselor and the topic at the end of the session.

OUTPUT FORMAT:
You must strictly return a valid JSON object containing exactly one key named "summary".
Do NOT include any other keys. 
Example Format:
{
  "summary": "Your one paragraph summary here..."
}
"""

    _primary_base = get_llm_model(
        temperature=0.3,
        provider=patient_config._config.PROVIDER,
        model_name=patient_config._config.MODEL_NAME,
        provider_preferences=patient_config._config.OPENROUTER_PROVIDER_PREFS,
    )
    _fallback_base = get_llm_model(
        temperature=0.3,
        provider=patient_config._config.PROVIDER,
        model_name=patient_config._config.FALLBACK_MODEL_NAME,
        provider_preferences=patient_config._config.OPENROUTER_PROVIDER_PREFS,
    )
    if _primary_base is None:
        raise RuntimeError("vanilla_client_memory_node: get_llm_model returned None for primary model.")
    if _fallback_base is None:
        raise RuntimeError("vanilla_client_memory_node: get_llm_model returned None for fallback model.")

    primary = (
        _primary_base
        .with_structured_output(VanillaMemoryOutput, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=patient_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    fallback = (
        _fallback_base
        .with_structured_output(VanillaMemoryOutput, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=patient_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    chain = primary.with_fallbacks([fallback])
    result: VanillaMemoryOutput = chain.invoke(  # type: ignore[assignment]
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Transcript:\n{conversation_text}"),
        ],
        config=config,
    )
    if not getattr(result, "summary", None):
        raise ValueError("vanilla_client_memory_node: LLM returned empty summary.")

    store = runtime.store
    if store is None:
        raise RuntimeError("vanilla_client_memory_node: runtime.store is None.")
    store.put(PATIENT_MEMORY_NS, "vanilla_memory", {"summary": result.summary})
    logger.info("vanilla_client_memory_node: summary stored.")
    return {}


@traced_span("vanilla_client_inter_sessions")
def vanilla_client_inter_sessions_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """
    Structural placeholder node for graph consistency.

    The vanilla baseline reads its rolling summary directly from the store in
    ``vanilla_client_node``, so no state transfer is required here. This node
    exists to maintain symmetry with the ConsistentClient inter-session
    architecture and to provide a named span in Phoenix traces.
    """
    logger.info("vanilla_client_inter_sessions_node invoked")
    return {}
