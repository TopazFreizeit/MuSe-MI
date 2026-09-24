import logging
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.utils import get_llm_model
from src.utils.formatting import format_conversation_history
from src.utils.tracing import traced_span
from src.patient.patient_config import PatientConfigClass
from src.utils.store import PATIENT_MEMORY_NS

logger = logging.getLogger(__name__)

class ConsistentMemoryOutput(BaseModel):
    summary: str = Field(description="A concise summary of the session from the patient's perspective, capturing key topics discussed, agreements made, and the patient's current attitude.")

@traced_span("consistent_client_memory")
def consistent_client_memory_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    logger.info("consistent_client_memory_node invoked")
    
    current_session = state.get("current_session", {})
    messages = current_session.get("messages", [])
    if not messages:
        logger.warning("No messages found in current_session, skipping memory generation")
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
    
    config_cls = PatientConfigClass()
    model = get_llm_model(
        temperature=0.7,
        provider=config_cls.PROVIDER,
        model_name=config_cls.MODEL_NAME,
        provider_preferences=config_cls.OPENROUTER_PROVIDER_PREFS
    ).with_structured_output(ConsistentMemoryOutput, method="json_mode")
    
    fallback = get_llm_model(
        temperature=0.7,
        provider=config_cls.PROVIDER,
        model_name=config_cls.FALLBACK_MODEL_NAME,
        provider_preferences=config_cls.OPENROUTER_PROVIDER_PREFS
    ).with_structured_output(ConsistentMemoryOutput, method="json_mode")
    
    chain = model.with_fallbacks([fallback])
    
    result: ConsistentMemoryOutput = chain.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{conversation_text}")
    ])
    
    # Store it in memory for the next session
    runtime.store.put(PATIENT_MEMORY_NS, "consistent_memory", {"summary": result.summary})
    
    logger.info("consistent_client_memory_node completed")
    return {}

@traced_span("consistent_client_inter_sessions")
def consistent_client_inter_sessions_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    # In this node we could transfer fields or perform checks, but since memory_node writes directly to the store
    # and the client_node reads directly from the store, we don't strictly need to modify MultiSessionState here.
    # We include it for graph structural consistency.
    logger.info("consistent_client_inter_sessions_node invoked")
    return {}
