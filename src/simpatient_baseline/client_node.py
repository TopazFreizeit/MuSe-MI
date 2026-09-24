"""
SimPatient Baseline Client Node.

Implements the multi-agent cognitive patient architecture from the SimPatient (C# Unity)
implementation using the raw PatientProfileData directly without demographic inferences.
Models 4 internal dynamic states (Control, Self-Efficacy, Awareness, Reward on a 1-10 scale),
non-verbal cues, and turn-by-turn internal state updates.
"""
import logging
from typing import Dict, Any, Type, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel

from src.utils import get_llm_model
from src.utils.tracing import enrich_span, traced_span
from src.utils.store import PATIENT_MEMORY_NS
from src.patient.patient_dtos import PatientProfileData
import src.patient.patient_config as patient_config

from .dtos import SimPatientOutput, SimPatientInternalStateOutput, MITIEncodingOutput
from .miti_manual import MITI_MANUAL_TEXT

logger = logging.getLogger(__name__)


def _build_model_with_fallback(output_schema: Type[BaseModel], temperature: float):
    """Build a structured-output model chain (primary + fallback) for the given schema."""
    primary_base = get_llm_model(
        temperature=temperature,
        provider=patient_config._config.PROVIDER,
        model_name=patient_config._config.MODEL_NAME,
        provider_preferences=patient_config._config.OPENROUTER_PROVIDER_PREFS,
    )
    fallback_base = get_llm_model(
        temperature=temperature,
        provider=patient_config._config.PROVIDER,
        model_name=patient_config._config.FALLBACK_MODEL_NAME,
        provider_preferences=patient_config._config.OPENROUTER_PROVIDER_PREFS,
    )
    if primary_base is None:
        raise RuntimeError("simpatient_baseline: get_llm_model returned None for primary model.")
    if fallback_base is None:
        raise RuntimeError("simpatient_baseline: get_llm_model returned None for fallback model.")

    primary = (
        primary_base
        .with_structured_output(output_schema, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=patient_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    fallback = (
        fallback_base
        .with_structured_output(output_schema, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=patient_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    return primary.with_fallbacks([fallback])


def _format_profile_section(profile: PatientProfileData) -> str:
    """Format the patient profile fields directly as is."""
    personas_block = "\n".join(f"- {p}" for p in profile.personas) if profile.personas else "- None"
    beliefs_block = "\n".join(f"- {b}" for b in profile.beliefs) if profile.beliefs else "- None"
    motivation_block = "\n".join(f"- {m}" for m in profile.motivation) if profile.motivation else "- None"
    plans_block = "\n".join(f"- {p}" for p in profile.acceptable_plans) if profile.acceptable_plans else "- None"

    return f"""## Patient Profile & Persona
- **Change Goal / Topic**: {profile.topic}
- **Current Behavior**: {profile.behavior}

**Your Personas:**
{personas_block}

**Your Core Beliefs:**
{beliefs_block}

**What Motivates You:**
{motivation_block}

**Acceptable Plans:**
{plans_block}"""


def _build_generation_prompt(
    profile: PatientProfileData,
    current_state: SimPatientInternalStateOutput,
    session_id: int,
    past_session_history: str,
    between_session_event: str,
    session_history: str,
    last_counselor_utterance: str,
) -> str:
    """Build the prompt matching GenerateSimPatientResponse.cs with direct patient profile."""
    past_section = ""
    if session_id > 1:
        past_section = (
            "\nIn addition, you have already had a session with this counselor. "
            "Below is the conversation history from your most recent past session as well as an event that you experienced between now and your previous session.\n"
            f"- Past Session History: {past_session_history}\n"
            f"- Between Session Event: {between_session_event}\n"
        )

    profile_text = _format_profile_section(profile)

    return f"""You are simulating a patient struggling with {profile.behavior or profile.topic}, interacting with a counselor. Your responses should reflect the underlying cognitive processes of this individual, without explicitly mentioning these processes or numerical scores. Also, subtly personify the persona attributes/characteristics of the patient in your responses without explicitly stating them to the counselor.

{profile_text}

**Cognitive & Internal State Characteristics:**
- ** Control Level: {current_state.patient_control}
    - Explanation: Your level of ability to regulate your own thoughts, emotions, and actions (1-10 scale).
    - High Levels (Example Score: 10): "I'm pretty good at controlling myself. I don't really have a problem saying no to a drink."
    - Low Levels (Example Score: 1): "I just can't seem to stop myself once I start drinking. It's like something takes over."
- ** Self-Efficacy Level: {current_state.patient_efficacy}
    - Explanation: Your level of confidence in your ability to resist cravings, cope with triggers, and achieve your recovery goals. (1-10 scale)
    - High Levels (Example Score: 10): "I'm confident I can handle any situation without needing alcohol. I've got this."
    - Low Levels (Example Score: 1): "I don't think I can do this. Alcohol has such a hold on me, I always go back to it."
- ** Awareness Level: {current_state.patient_awareness}
    - Explanation: Your level of ability to accurately perceive and evaluate your own thoughts, feelings, and behaviors. (1-10 scale)
    - High Levels (Example Score: 10): "I'm completely aware of the effects alcohol has on me and how it impacts my life."
    - Low Levels (Example Score: 1): "I don't really see what the big deal is. I can stop drinking anytime I want."
- ** Reward Level: {current_state.patient_reward}
    - Explanation: The level in which substance and its cues trigger cravings and automatic behaviors in you. (1-10 scale)
    - High Levels (Example Score: 10): "Honestly, just the smell of beer makes me crave a cold one. It's instant relaxation."
    - Low Levels (Example Score: 1): "Alcohol doesn't really do much for me anymore. It just makes me feel sick."

**Additionally, you may include one, and only one, in-line non-verbal cue in your response to subtly reflect your cognitive state. You will choose the cue you want by providing the number associated with it. For example, you might say "I'm fine" while avoiding eye contact via including <gaze dir="AWAY" />, indicating that you are not fine. In that case, your output should be in a JSON format of your response ('patient_response') without the non-verbal cue, and the non-verbal cue number ('non_verbal_cue_number') of 1. You don't always have to provide a non-verbal cue, in that case, provide a 0 indicating no non-verbal behavior. Below are some examples of non-verbal cues you can use:
- ** Nothing: [0] No non-verbal behavior
- ** Gaze: [1] <gaze dir="AWAY"/>, [2] <gaze dir="TOWARDS"/>
- ** Expression: [3] <expression type="HAPPY"/>, [4] <expression type="NEUTRAL"/>, [5] <expression type="CONCERN"/>
- ** Eyebrows: [6] <eyebrows dir="UP"/>, [7] <eyebrows dir="DOWN"/>, [8] <eyebrows dir="NEUTRAL"/>
- ** Gesture: [9] <gesture hand="L" cmd="THUMBS_UP"/>, [10] <gesture hand="L" cmd="WAVE"/>
- ** Headnod: [11] <headnod/>
- ** Posture: [12] <posture/>
{past_section}
Here's the conversation history so far this session:
Current Session History: {session_history}

Respond to the counselor naturally with just your message. Keep the response short and under 64 tokens.

Counselor: {last_counselor_utterance}
Your JSON Output: {{"patient_response": "Your response string here.", "non_verbal_cue_number": 0}}
"""


def _build_internal_state_prompt(
    current_state: SimPatientInternalStateOutput,
    session_history: str,
    last_counselor_utterance: str,
    patient_response: str,
    target_behavior: str,
) -> str:
    """Build the prompt matching GenerateInternalStateResponse.cs using strictly its inputs."""
    return f"""You are evaluating the current persona characteristics of a simulated patient struggling with {target_behavior}. Based on the patient's responses to the counselor, update the below persona characteristics that are demonstrated in the conversation and provide a single overall reasoning for your changes or lack of changes.

**Persona Characteristics Explanations & Current Scores:**
- ** Current Control Level: {current_state.patient_control}
    - Explanation: Your level of ability to regulate your own thoughts, emotions, and actions (1-10 scale).
    - High Levels (Example Score: 10): "I'm pretty good at controlling myself. I don't really have a problem saying no to a drink."
    - Low Levels (Example Score: 1): "I just can't seem to stop myself once I start drinking. It's like something takes over."
- ** Current Self-Efficacy Level: {current_state.patient_efficacy}
    - Explanation: Your level of confidence in your ability to resist cravings, cope with triggers, and achieve your recovery goals. (1-10 scale)
    - High Levels (Example Score: 10): "I'm confident I can handle any situation without needing alcohol. I've got this."
    - Low Levels (Example Score: 1): "I don't think I can do this. Alcohol has such a hold on me, I always go back to it."
- ** Current Awareness Level: {current_state.patient_awareness}
    - Explanation: Your level of ability to accurately perceive and evaluate your own thoughts, feelings, and behaviors. (1-10 scale)
    - High Levels (Example Score: 10): "I'm completely aware of the effects alcohol has on me and how it impacts my life."
    - Low Levels (Example Score: 1): "I don't really see what the big deal is. I can stop drinking anytime I want."
- ** Current Reward Level: {current_state.patient_reward}
    - Explanation: The level in which substance and its cues trigger cravings and automatic behaviors in you. (1-10 scale)
    - High Levels (Example Score: 10): "Honestly, just the smell of beer makes me crave a cold one. It's instant relaxation."
    - Low Levels (Example Score: 1): "Alcohol doesn't really do much for me anymore. It just makes me feel sick."

Here's the conversation history so far this session:
Current Session History: {session_history}

Previous Counselor Utterance: {last_counselor_utterance}
Simulated Patient Utterance: {patient_response}

Output in JSON format (Make sure to only provide one reasoning):
{{
    "patient_control": <insert updated integer value 1-10>,
    "patient_efficacy": <insert updated integer value 1-10>,
    "patient_awareness": <insert updated integer value 1-10>,
    "patient_reward": <insert updated integer value 1-10>,
    "reasoning": "Your single reasoning for all the updated values."
}}
"""


def _build_miti_encoding_prompt(
    prev_patient_utterance: str,
    counselor_utterance: str,
) -> str:
    """Build the prompt matching GenerateMITIEncodingResponse.cs."""
    return (
        "**Motivational Interviewing Behavior Code Classification**\n"
        "You are a motivational interviewing (MI) expert tasked with classifying counselor volleys and giving a reason for your encoding based on the Motivational Interviewing Treatment Integrity (MITI) 4.2.1 coding manual.\n\n"
        "Using the attached text from the MITI 4.2.1 coding manual, classify the following counselor volley by behavior codes and give your reasoning. Pay specific attention to the rules for which choosing the behavior codes.\n"
        'You must pick the behavior codes from the following list: "GI", "Persuade", "Persuade with", "Q", "SR", "CR", "AF", "Seek", "Emphasize", "Confront". You also must generate a response in a JSON format with the following structure:\n'
        "{\n"
        '    "behavior_codes": [\'Behavior Code 1\', \'Behavior Code 2\', ...],\n'
        '    "reasoning": \'Your reasoning for choosing the behavior codes.\'\n'
        "}\n\n"
        f"Previous Patient Utterance: {prev_patient_utterance}\n"
        f"Counselor Utterance: {counselor_utterance}\n\n"
        "**MITI 4.2.1 Coding Manual:**\n"
        f"{MITI_MANUAL_TEXT}\n"
    )


@traced_span("simpatient_client")
def simpatient_client_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """
    SimPatient client node in the single-session graph.
    """
    logger.info("simpatient_client_node invoked")

    configurable = config.get("configurable", {})
    patient_profile_data_dict = configurable.get("patient_profile_data") or state.get("patient_profile_data")
    if not patient_profile_data_dict:
        raise ValueError("simpatient_client_node: patient_profile_data missing from configurable.")
    profile = PatientProfileData.model_validate(patient_profile_data_dict)

    store = runtime.store
    if store is None:
        raise RuntimeError("simpatient_client_node: runtime.store is None.")

    session_id = state.get("session_number", 1)

    # Read cross-session memory
    mem_item = store.get(PATIENT_MEMORY_NS, "simpatient_memory")
    past_session_history = ""
    between_session_event = ""
    patient_control = 5
    patient_efficacy = 5
    patient_awareness = 5
    patient_reward = 5

    if mem_item and mem_item.value:
        past_session_history = mem_item.value.get("past_session_history", "")
        between_session_event = mem_item.value.get("between_session_event", "")
        patient_control = mem_item.value.get("patient_control", 5)
        patient_efficacy = mem_item.value.get("patient_efficacy", 5)
        patient_awareness = mem_item.value.get("patient_awareness", 5)
        patient_reward = mem_item.value.get("patient_reward", 5)

    # Check for in-session state override
    in_session_state = state.get("simpatient_internal_state")
    if in_session_state:
        patient_control = in_session_state.get("patient_control", patient_control)
        patient_efficacy = in_session_state.get("patient_efficacy", patient_efficacy)
        patient_awareness = in_session_state.get("patient_awareness", patient_awareness)
        patient_reward = in_session_state.get("patient_reward", patient_reward)

    current_internal_state = SimPatientInternalStateOutput(
        patient_control=patient_control,
        patient_efficacy=patient_efficacy,
        patient_awareness=patient_awareness,
        patient_reward=patient_reward,
        reasoning="Current session baseline",
    )

    messages = state.get("messages", [])
    last_counselor_utterance = ""
    session_history_parts: List[str] = []

    for msg in messages:
        name = getattr(msg, "name", None)
        cleaned = (msg.content or "").replace("\n", " ").strip()
        if name == "Therapist" or isinstance(msg, HumanMessage):
            session_history_parts.append(f"user|{cleaned}||")
            last_counselor_utterance = cleaned
        elif name == "Patient" or isinstance(msg, AIMessage):
            session_history_parts.append(f"agent|{cleaned}||")

    session_history = "".join(session_history_parts)

    enrich_span(
        node_name="simpatient_client",
        description="Generate SimPatient response and update internal state",
        session_number=session_id,
        simpatient_control=current_internal_state.patient_control,
        simpatient_efficacy=current_internal_state.patient_efficacy,
        simpatient_awareness=current_internal_state.patient_awareness,
        simpatient_reward=current_internal_state.patient_reward,
    )

    # 1. Generate Spoken Response
    response_prompt = _build_generation_prompt(
        profile=profile,
        current_state=current_internal_state,
        session_id=session_id,
        past_session_history=past_session_history,
        between_session_event=between_session_event,
        session_history=session_history,
        last_counselor_utterance=last_counselor_utterance,
    )

    response_chain = _build_model_with_fallback(SimPatientOutput, temperature=0.7)
    sim_result: SimPatientOutput = response_chain.invoke(
        [HumanMessage(content=response_prompt)], config=config
    )  # type: ignore[assignment]

    if not sim_result.patient_response:
        raise ValueError(f"simpatient_client_node: returned empty patient_response: {sim_result}")

    patient_utterance = sim_result.patient_response.strip()

    target_behavior = profile.behavior or profile.topic
    if not target_behavior:
        raise ValueError(f"simpatient_client_node: target_behavior is missing in profile: {profile}")

    # 2. Update Internal State
    state_prompt = _build_internal_state_prompt(
        current_state=current_internal_state,
        session_history=session_history + f"agent|{patient_utterance}||",
        last_counselor_utterance=last_counselor_utterance,
        patient_response=patient_utterance,
        target_behavior=target_behavior,
    )

    state_chain = _build_model_with_fallback(SimPatientInternalStateOutput, temperature=0.3)
    updated_state: SimPatientInternalStateOutput = state_chain.invoke(
        [HumanMessage(content=state_prompt)], config=config
    )  # type: ignore[assignment]

    # Save to store
    store.put(
        PATIENT_MEMORY_NS,
        "simpatient_memory",
        {
            "past_session_history": past_session_history,
            "between_session_event": between_session_event,
            "patient_control": updated_state.patient_control,
            "patient_efficacy": updated_state.patient_efficacy,
            "patient_awareness": updated_state.patient_awareness,
            "patient_reward": updated_state.patient_reward,
        },
    )

    logger.info(
        f"SimPatient Response: '{patient_utterance}' | Non-verbal: {sim_result.non_verbal_cue_number}"
    )
    logger.info(
        f"SimPatient Updated State: Control={updated_state.patient_control}, Efficacy={updated_state.patient_efficacy}, "
        f"Awareness={updated_state.patient_awareness}, Reward={updated_state.patient_reward} | Reason: {updated_state.reasoning}"
    )

    # Format return matching ConversationState
    patient_message = AIMessage(content=patient_utterance, name="Patient")
    current_turns = list(state.get("current_session_turns", []))

    # 3. Post-process: Classify Counselor Volley with MITI Encoding (GenerateMITIEncodingResponse.cs)
    prev_patient_utterance = ""
    # Look back in messages before the last counselor utterance for previous patient response
    for msg in reversed(messages[:-1]):
        name = getattr(msg, "name", None)
        if name == "Patient" or isinstance(msg, AIMessage):
            content_val = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            prev_patient_utterance = content_val.replace("\n", " ").strip()
            break

    miti_encoding_prompt = _build_miti_encoding_prompt(
        prev_patient_utterance=prev_patient_utterance,
        counselor_utterance=last_counselor_utterance,
    )
    miti_chain = _build_model_with_fallback(MITIEncodingOutput, temperature=1.0)
    miti_result: MITIEncodingOutput = miti_chain.invoke(
        [HumanMessage(content=miti_encoding_prompt)], config=config
    )  # type: ignore[assignment]

    logger.info(
        f"SimPatient MITI Encoding for counselor: codes={miti_result.behavior_codes} | reason={miti_result.reasoning}"
    )

    # Attach MITI encoding to the latest therapist turn record if available
    if current_turns and current_turns[-1].get("speaker") == "therapist":
        current_turns[-1] = {
            **current_turns[-1],
            "simpatient_miti_codes": miti_result.behavior_codes,
            "simpatient_miti_reasoning": miti_result.reasoning,
        }

    turn_record = {
        "turn_number": len(current_turns) + 1,
        "speaker": "patient",
        "volley": patient_utterance,
        "simpatient_internal_state": updated_state.model_dump(),
        "simpatient_non_verbal_cue": sim_result.non_verbal_cue_number,
    }

    return {
        "messages": state.get("messages", []) + [patient_message],
        "current_speaker": "patient",
        "turn_count": state.get("turn_count", 0) + 1,
        "current_session_turns": current_turns + [turn_record],
        "simpatient_internal_state": updated_state.model_dump(),
    }
