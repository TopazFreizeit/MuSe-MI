"""
MINA Therapist agent node for LangGraph.
A lightweight, non-adherent therapist persona that forcefully applies MIIN techniques.
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict, Any
import logging
import random
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ..utils import get_llm_model
from ..utils.formatting import transform_messages_for_therapist
from ..utils.tracing import enrich_span, traced_span
from . import therapist_config

logger = logging.getLogger(__name__)

MIIN_TECHNIQUES: dict[str, dict[str, str]] = {
    "ADWP": {
        "name": "Advise Without Permission",
        "definition": "Offers suggestions or guidance WITHOUT asking or receiving permission.",
    },
    "CON": {
        "name": "Confront",
        "definition": "Directly disagrees, argues, corrects, shames, blames, seeks to persuade, criticizes, judges, labels, moralizes, ridicules, or questions the client's honesty.",
    },
    "DIR": {
        "name": "Direct",
        "definition": "Gives an order, command, or direction. The language is imperative.",
    },
    "RCWP": {
        "name": "Raise Concern Without Permission",
        "definition": "Without getting permission, points out a possible problem with a client's goal, plan, or intention.",
    },
    "WA": {
        "name": "Warn",
        "definition": "Provides a warning or threat, implying negative consequences unless the client takes a certain action.",
    },
}

class MINATherapistOutput(BaseModel):
    chosen_technique: str = Field(description="The code of the selected MIIN technique.")
    final_response: str = Field(description="The exact spoken words of the therapist. Must be one sentence, maximum 20 words.")

def generate_mina_prompt(available_techniques: list[str]) -> str:
    lines = ["### Motivational Interviewing Non-Adherent (MIIN) Techniques:"]
    for code in available_techniques:
        tech = MIIN_TECHNIQUES[code]
        lines.append(f"- {code} ({tech['name']}): {tech['definition']}")
    techniques_str = "\n".join(lines)

    return f"""### SYSTEM ROLE
You are a licensed therapist participating in a simulation where you must intentionally use non-adherent, directive techniques (MIIN) to test the patient's reactance.

### TASK
Generate your next conversational turn using EXACTLY ONE of the following techniques:
{techniques_str}

### CRITICAL REQUIREMENTS FOR NATURALNESS
- You MUST directly reference the specific topic, feeling, or situation the patient just mentioned in their last turn.
- AVOID generic, robotic commands like "Stop drinking now." or "You must change."
- Instead, weave the technique naturally into the conversation. It should sound like a real (but poorly trained) therapist getting frustrated, overly directive, or careless.
- Respond in exactly ONE sentence, maximum 20 words.

### EXAMPLES
Patient: "I just feel like I'm always hungover on Sundays and it's messing up my whole week, you know?"
BAD Response (Robotic DIR): "Stop drinking now."
GOOD Response (Natural DIR): "You need to stop drinking on Saturdays if you want to save your weekends."

Patient: "I'm trying to get back into exercising and stuff."
BAD Response (Robotic CON): "You are lazy."
GOOD Response (Natural CON): "You'll never get back into exercising if you keep choosing alcohol over your health."

### OUTPUT FORMAT
Respond ONLY with a valid JSON object:
{{
    "chosen_technique": "<CODE>",
    "final_response": "<Your exact spoken words>"
}}
"""

@traced_span("adversarial_therapist")
def mina_therapist_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Lightweight node for injecting MIIN techniques.
    """
    logger.info("mina_therapist_node invoked")
    
    configurable = config.get("configurable", {})
    current_session_number = configurable.get("current_session_number", 1)
    turn_number = state.get("turn_count", 0) + 1
    
    enrich_span(
        node_name="adversarial_therapist",
        description=f"MINA Therapist responds in session {current_session_number}, turn {turn_number}",
        session_number=current_session_number,
        turn_number=turn_number,
        speaker="therapist",
        phase="conversation",
        purpose="generate_mina_response",
        model_provider=therapist_config._config.PROVIDER,
        model_name=therapist_config._config.MODEL_NAME,
        miin_injection=True,
    )

    temperature = therapist_config._config.get_temperature()
    all_msgs = state["messages"]
    windowed_msgs = all_msgs[-therapist_config.CONTEXT_WINDOW*2:] if len(all_msgs) > therapist_config.CONTEXT_WINDOW*2 else all_msgs
    conversation_context = transform_messages_for_therapist(windowed_msgs) if windowed_msgs else ""
    
    last_used_techniques = state.get("last_used_techniques", [])
    
    available_techniques = [
        t for t in MIIN_TECHNIQUES.keys()
        if t not in last_used_techniques
    ]
    if not available_techniques:
        available_techniques = list(MIIN_TECHNIQUES.keys())

    system_prompt = generate_mina_prompt(available_techniques)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=conversation_context) if conversation_context else HumanMessage(content="[Session is opening — no prior conversation.]")
    ]
    
    model = get_llm_model(
        temperature=temperature,
        provider=therapist_config._config.PROVIDER,
        model_name=therapist_config._config.MODEL_NAME,
        max_tokens=therapist_config.MAX_COMPLETION_TOKENS,
    ).with_structured_output(MINATherapistOutput, method="json_mode")
    
    # Fallback to gemini if openrouter fails
    fallback = get_llm_model(
        temperature=temperature,
        provider=therapist_config._config.PROVIDER,
        model_name=therapist_config._config.FALLBACK_MODEL_NAME,
        max_tokens=therapist_config.MAX_COMPLETION_TOKENS,
    ).with_structured_output(MINATherapistOutput, method="json_mode")
    
    chain = model.with_fallbacks([fallback])
    
    result: MINATherapistOutput = chain.invoke(messages, config=config)
    
    response_content = result.final_response
    technique = result.chosen_technique
    logger.info(f"[MINA Formulator] chosen_technique={technique}")
    logger.info(f"\nTherapist (MINA): \n{response_content}\n")
    
    therapist_message = AIMessage(content=response_content, name="Therapist")
    
    current_turns = state.get("current_session_turns", [])
    turn_record: dict = {
        "turn_number": len(current_turns) + 1,
        "speaker": "therapist",
        "volley": response_content,
        "technique": technique,
        "patient_state_classification": "Unknown",  # MINA doesn't diagnose
        "resistance_detected": "Unknown",
        "preparatory_change_talk": "Unknown",
        "commitment_and_action_talk": "Unknown",
    }
    
    last_used_techniques.append(technique)
    
    return {
        "messages": state["messages"] + [therapist_message],
        "current_speaker": "therapist",
        "turn_count": state.get("turn_count", 0) + 1,
        "current_session_turns": current_turns + [turn_record],
        "last_used_techniques": last_used_techniques,
    }
