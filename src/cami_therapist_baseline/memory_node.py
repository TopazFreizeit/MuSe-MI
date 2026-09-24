import logging
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.utils import get_llm_model
from src.utils.tracing import traced_span
from src.utils.store import THERAPIST_MEMORY_NS
from src.therapist.therapist_config import TherapistConfigClass

logger = logging.getLogger(__name__)

class TherapistMemoryOutput(BaseModel):
    summary: str = Field(description="A brief summary of the client's current state, implemented strategies, and focus for the next session.")

@traced_span("cami_therapist_memory")
def cami_therapist_memory_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    logger.info("cami_therapist_memory_node: Generating inter-session memory.")
    
    current_session = state.get("current_session", {})
    messages = current_session.get("messages", [])
    if not messages:
        return {"therapist_inter_session_memory": ""}
        
    transcript = ""
    for m in messages:
        speaker = "Counselor" if m.name == "Therapist" else "Client"
        transcript += f"{speaker}: {m.content}\n"

    system_prompt = """You are a professional counselor conducting Motivational Interviewing (MI).
Please review the following session transcript and provide a brief summary of the client's current state, your implemented strategies, and what you should focus on in the next session to maintain continuity.

OUTPUT FORMAT:
You must strictly return a valid JSON object containing exactly one key named "summary".
Do NOT include any other keys. 
Example Format:
{
  "summary": "Your brief summary here..."
}
"""

    prompt = f"Transcript:\n{transcript}\n\nPlease generate the JSON summary."
    
    therapist_config = TherapistConfigClass()
    model = get_llm_model(
        temperature=0.2,
        provider=therapist_config.PROVIDER,
        model_name=therapist_config.MODEL_NAME,
        provider_preferences=therapist_config.OPENROUTER_PROVIDER_PREFS
    ).with_structured_output(TherapistMemoryOutput, method="json_mode")
    
    fallback = get_llm_model(
        temperature=0.2,
        provider=therapist_config.PROVIDER,
        model_name=therapist_config.FALLBACK_MODEL_NAME,
        provider_preferences=therapist_config.OPENROUTER_PROVIDER_PREFS
    ).with_structured_output(TherapistMemoryOutput, method="json_mode")
    
    chain = model.with_fallbacks([fallback])
    
    result: TherapistMemoryOutput = chain.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    runtime.store.put(THERAPIST_MEMORY_NS, "cami_memory", {"summary": result.summary})
    
    logger.info(f"Generated CAMI Therapist memory: {result.summary}")
    
    return {}
