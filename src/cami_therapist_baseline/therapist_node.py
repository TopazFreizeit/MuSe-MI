import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from src.utils.store import THERAPIST_MEMORY_NS
from src.utils.tracing import enrich_span, traced_span
from src.patient.patient_dtos import PatientProfileData
from .adapter import CAMI

logger = logging.getLogger(__name__)

@traced_span("cami_therapist")
def cami_therapist_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    logger.info("cami_therapist_node invoked")
    
    configurable = config.get("configurable", {})
    current_session_number = configurable.get("current_session_number", 1)
    turn_number = state.get("turn_count", 0) + 1
    
    enrich_span(
        node_name="cami_therapist",
        description=f"CAMI Therapist responds in session {current_session_number}, turn {turn_number}",
        session_number=current_session_number,
        turn_number=turn_number,
        speaker="therapist",
        phase="conversation",
    )
    
    patient_profile_data_dict = configurable.get("patient_profile_data")
    profile_data = PatientProfileData.model_validate(patient_profile_data_dict)
    
    # We pass 'dummy' for model since it's overridden by _call_llm configurations in adapter
    # CAMI initializes empty. We need to feed state into it.
    cami = CAMI(goal=profile_data.topic, behavior=profile_data.behavior, model="dummy")
    
    store = runtime.store
    memory_item = store.get(THERAPIST_MEMORY_NS, "cami_memory")
    memory_prompt = memory_item.value.get("summary", "") if memory_item else ""
    
    if memory_prompt:
        # inject memory to the system prompt
        cami.messages[0]["content"] += f"\n\nMemory from previous sessions:\n{memory_prompt}" if memory_item else ""
    
    # Reset default mocked conversation if we have real state messages
    all_messages = state.get("messages", [])
    if len(all_messages) > 0:
        cami.messages = [cami.messages[0]]
        cami.conversation = []

        # We must replay the conversion history into cami.messages and cami.conversation
        for msg in all_messages:
            if msg.name == "Therapist" or msg.name in ["cami_therapist", "counselor"]:
                content = msg.content if msg.content.startswith("Counselor: ") else f"Counselor: {msg.content}"
                cami.messages.append({"role": "assistant", "content": content})
                cami.conversation.append(content)
            elif msg.name == "Patient" or msg.name == "Client":
                content = msg.content if msg.content.startswith("Client: ") else f"Client: {msg.content}"
                cami.messages.append({"role": "user", "content": content})
                cami.conversation.append(content)
                
    cami.initialized = len(all_messages) > 0
    # Also initialize explored topics based on past? It only explores new topics sequentially.
    # We can fetch state to maintain it.
    cami_dict = state.get("cami_therapist_state", {})
    cami.explored_topics = cami_dict.get("explored_topics", [])
    
    if len(all_messages) == 0:
        cami.initialized = False
        logger.info(f"CAMI Therapist initialized for first turn. Topic: {profile_data.topic}")
        
    logic_metadata, raw_response = cami.reply()

    # raw_response always starts with "Counselor: " — strip it so the patient sees plain text
    COUNSELOR_PREFIX = "Counselor: "
    if raw_response.startswith(COUNSELOR_PREFIX):
        clean_response = raw_response[len(COUNSELOR_PREFIX):]
    else:
        clean_response = raw_response

    logger.info(f"\n\nCAMI Therapist response:\n\n{clean_response}\n\n")

    ai_message = AIMessage(content=clean_response, name="Therapist")

    current_turns = state.get("current_session_turns", [])
    turn_record = {
        "turn_number": state.get("turn_count", 0) + 1,
        "speaker": "therapist",
        "volley": clean_response,
        "logic_metadata": logic_metadata,
    }
    
    new_cami_dict = {
        "explored_topics": cami.explored_topics
    }

    return {
        "messages": all_messages + [ai_message],
        "current_speaker": "therapist",
        "turn_count": state.get("turn_count", 0) + 1,
        "current_session_turns": current_turns + [turn_record],
        "cami_therapist_state": new_cami_dict
    }

