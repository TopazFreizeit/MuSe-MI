"""
SimPatient Inter-Session Memory Pipeline Nodes.

Implements the multi-agent inter-session flow matching the SimPatient C# architecture:
1. GenerateGlobalScoresResponse.cs -> simpatient_global_scores_node
2. GenerateSessionSummaryResponse.cs -> simpatient_session_summary_node (past session memory)
3. GenerateBetweenSessionEventResponse.cs -> simpatient_between_session_event_node
"""
import logging
from typing import Dict, Any, List, Type

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel

from src.utils import get_llm_model
from src.utils.formatting import format_conversation_history
from src.utils.tracing import traced_span
import src.patient.patient_config as patient_config
from src.patient.patient_dtos import PatientProfileData
from src.utils.store import PATIENT_MEMORY_NS

from .dtos import (
    GlobalScoresOutput,
    SessionSummaryOutput,
    SimPatientBetweenSessionEventOutput,
)
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
        raise RuntimeError("simpatient_memory: get_llm_model returned None for primary model.")
    if fallback_base is None:
        raise RuntimeError("simpatient_memory: get_llm_model returned None for fallback model.")

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


def _build_pipe_session_history(messages: List[Any]) -> str:
    """Format session history as pipe-separated string matching SimPatient SESSION_HISTORY."""
    parts: List[str] = []
    for msg in messages:
        name = getattr(msg, "name", None)
        cleaned = (msg.content or "").replace("\n", " ").strip()
        cleaned = cleaned.replace("'", "").replace('"', "")
        if name == "Therapist":
            parts.append(f"user|{cleaned}||")
        elif name == "Patient":
            parts.append(f"agent|{cleaned}||")
        else:
            if getattr(msg, "type", "") == "human":
                parts.append(f"user|{cleaned}||")
            else:
                parts.append(f"agent|{cleaned}||")
    return "".join(parts)


def _build_session_history_for_summary(current_session_turns: List[Dict[str, Any]]) -> str:
    """
    Format annotated session transcript including MITI codes and internal state updates
    matching SimPatient SESSION_HISTORY_FOR_SUMMARY.
    """
    lines: List[str] = []
    for t in current_session_turns:
        speaker = t.get("speaker", "")
        volley = (t.get("volley", "") or "").replace("\n", " ").strip()
        if speaker == "therapist":
            codes = t.get("simpatient_miti_codes", [])
            reasoning = t.get("simpatient_miti_reasoning", "")
            codes_str = "|".join(codes) if codes else "''"
            lines.append(f"User: {volley} [MITI Codes: {codes_str}] [Reasoning: {reasoning}]")
        elif speaker == "patient":
            state_data = t.get("simpatient_internal_state", {})
            ctrl = state_data.get("patient_control", "")
            eff = state_data.get("patient_efficacy", "")
            awa = state_data.get("patient_awareness", "")
            rew = state_data.get("patient_reward", "")
            reason = state_data.get("reasoning", "")
            lines.append(
                f"SimPatient: {volley} [Control: {ctrl}, Efficacy: {eff}, Awareness: {awa}, Reward: {rew}] [Reasoning: {reason}]"
            )
    return "\n".join(lines)


def _compute_behavior_code_metrics(current_session_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate MITI behavior code counts, percentages, and adherence totals."""
    counts = {
        "GI": 0,
        "Persuade": 0,
        "Persuade with": 0,
        "Q": 0,
        "SR": 0,
        "CR": 0,
        "AF": 0,
        "Seek": 0,
        "Emphasize": 0,
        "Confront": 0,
    }
    for t in current_session_turns:
        if t.get("speaker") == "therapist":
            codes = t.get("simpatient_miti_codes", [])
            for c in codes:
                c_clean = c.strip()
                if c_clean in counts:
                    counts[c_clean] += 1

    cr = counts["CR"]
    sr = counts["SR"]
    q = counts["Q"]
    percent_cr = (cr / (cr + sr) * 100.0) if (cr + sr) > 0 else 0.0
    reflect_q_ratio = ((cr + sr) / q) if q > 0 else float(cr + sr)

    total_adherent = counts["AF"] + counts["Seek"] + counts["Emphasize"] + counts["Persuade with"]
    total_non_adherent = counts["Confront"] + counts["Persuade"]

    return {
        "counts": counts,
        "percent_cr": round(percent_cr, 2),
        "reflect_question_ratio": round(reflect_q_ratio, 2),
        "total_adherent": total_adherent,
        "total_non_adherent": total_non_adherent,
    }


# ===========================================================================
# 1. Global Scores Node (GenerateGlobalScoresResponse.cs)
# ===========================================================================
@traced_span("simpatient_global_scores")
def simpatient_global_scores_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """
    Analyze the full counseling transcript and output global scores on cultivating change talk,
    softening sustain talk, partnership, and empathy based on MITI 4.2.1.
    """
    logger.info("simpatient_global_scores_node invoked")

    current_session = state.get("current_session", {})
    messages = current_session.get("messages", [])
    if not messages:
        logger.warning("simpatient_global_scores_node: no messages in current_session.")
        return {}

    session_history = _build_pipe_session_history(messages)

    prompt = (
        "**Motivational Interviewing Global Scoring**\n"
        "You are a motivational interviewing (MI) expert tasked with analyzing a counseling transcript and providing global scores on cultivating change talk, softening sustain talk, partnership, and empathy based on the Motivational Interviewing Treatment Integrity (MITI) 4.2.1 coding manual.\n\n"
        "Using the attached text from the MITI 4.2.1 coding manual to help you, step through the counseling transcript to generate final 1(low)-5(high) scores on cultivating change talk, softening sustain talk, partnership, and empathy. Pay specific attention to the rules for scoring and provide your reasoning for each score.\n"
        "You also must generate a response in a JSON format with the following structure:\n"
        "{\n"
        '    "cultivating_change_talk": {\n'
        '        "score": <int score>,\n'
        '        "reasoning": "Your reasoning for the score."\n'
        "    },\n"
        '    "softening_sustain_talk": {\n'
        '        "score": <int score>,\n'
        '        "reasoning": "Your reasoning for the score."\n'
        "    },\n"
        '    "partnership": {\n'
        '        "score": <int score>,\n'
        '        "reasoning": "Your reasoning for the score."\n'
        "    },\n"
        '    "empathy": {\n'
        '        "score": <int score>,\n'
        '        "reasoning": "Your reasoning for the score."\n'
        "    }\n"
        "}\n\n"
        f"**Counseling Transcript:**\n{session_history}\n\n"
        f"**MITI 4.2.1 Coding Manual:**\n{MITI_MANUAL_TEXT}\n"
    )

    chain = _build_model_with_fallback(GlobalScoresOutput, temperature=1.0)
    result: GlobalScoresOutput = chain.invoke(
        [HumanMessage(content=prompt)], config=config
    )  # type: ignore[assignment]

    cct = result.cultivating_change_talk.score
    sst = result.softening_sustain_talk.score
    part = result.partnership.score
    emp = result.empathy.score

    technical_global = round((cct + sst) / 2.0, 2)
    relational_global = round((part + emp) / 2.0, 2)

    global_scores_data = {
        "cultivating_change_talk": {"score": cct, "reasoning": result.cultivating_change_talk.reasoning},
        "softening_sustain_talk": {"score": sst, "reasoning": result.softening_sustain_talk.reasoning},
        "partnership": {"score": part, "reasoning": result.partnership.reasoning},
        "empathy": {"score": emp, "reasoning": result.empathy.reasoning},
        "technical_global": technical_global,
        "relational_global": relational_global,
    }

    logger.info(
        f"SimPatient Global Scores: CCT={cct}, SST={sst}, Part={part}, Emp={emp} | "
        f"Tech={technical_global}, Rel={relational_global}"
    )

    # Persist in store
    store = runtime.store
    if store is not None:
        mem_item = store.get(PATIENT_MEMORY_NS, "simpatient_memory")
        existing_val = mem_item.value.copy() if mem_item and mem_item.value else {}
        existing_val["global_scores"] = global_scores_data
        store.put(PATIENT_MEMORY_NS, "simpatient_memory", existing_val)

    return {"simpatient_global_scores": global_scores_data}


# ===========================================================================
# 2. Session Summary Node (GenerateSessionSummaryResponse.cs)
# ===========================================================================
@traced_span("simpatient_session_summary")
def simpatient_session_summary_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """
    Summarize the session using MITI global scores, behavior counts, and annotated transcript.
    The resulting summary is stored as `past_session_history` (the true cross-session memory).
    """
    logger.info("simpatient_session_summary_node invoked")

    current_session = state.get("current_session", {})
    turns = current_session.get("current_session_turns", [])
    global_scores_data = state.get("simpatient_global_scores")

    store = runtime.store
    if store is None:
        raise RuntimeError("simpatient_session_summary_node: runtime.store is None.")

    # Fallback to store if state doesn't have global scores
    if not global_scores_data:
        mem_item = store.get(PATIENT_MEMORY_NS, "simpatient_memory")
        if mem_item and mem_item.value:
            global_scores_data = mem_item.value.get("global_scores", {})

    if not global_scores_data:
        global_scores_data = {
            "cultivating_change_talk": {"score": 3, "reasoning": "Standard baseline"},
            "softening_sustain_talk": {"score": 3, "reasoning": "Standard baseline"},
            "partnership": {"score": 3, "reasoning": "Standard baseline"},
            "empathy": {"score": 3, "reasoning": "Standard baseline"},
            "technical_global": 3.0,
            "relational_global": 3.0,
        }

    metrics = _compute_behavior_code_metrics(turns)
    counts = metrics["counts"]
    transcript_annotated = _build_session_history_for_summary(turns)

    cct_score = global_scores_data.get("cultivating_change_talk", {}).get("score", 0)
    cct_reason = global_scores_data.get("cultivating_change_talk", {}).get("reasoning", "")
    sst_score = global_scores_data.get("softening_sustain_talk", {}).get("score", 0)
    sst_reason = global_scores_data.get("softening_sustain_talk", {}).get("reasoning", "")
    part_score = global_scores_data.get("partnership", {}).get("score", 0)
    part_reason = global_scores_data.get("partnership", {}).get("reasoning", "")
    emp_score = global_scores_data.get("empathy", {}).get("score", 0)
    emp_reason = global_scores_data.get("empathy", {}).get("reasoning", "")

    tech_score = global_scores_data.get("technical_global", 0.0)
    rel_score = global_scores_data.get("relational_global", 0.0)

    prompt = (
        "**Summary of Session**\n"
        "You are a motivational interviewing (MI) expert tasked with summarizing the session based on the data collected. Provide a concise summary of the session, highlighting key points, insights, and recommendations for future sessions to help the counselor/trainee improve their MI skills. Use the MITI 4.2.1 Coding Manual to help understand what all the codes mean. You do not have to reference the scores below in your summary, but just provide a general overview of the session based off of them and the conversation history.\n\n"
        "The summary should be direct and too the point, without any sections or bullet points. The summary should be a paragraph or two long. Keep it under 512 tokens.\n\n"
        "Below are MI scores for evaluating the effectiveness of the counselor/trainee conducting MI and reasoning behind the scores:\n"
        f"- **Cultivating Change Talk Score:** {cct_score}\n"
        f"- **Softening Sustain Talk Score:** {sst_score}\n"
        f"- **Partnership Score:** {part_score}\n"
        f"- **Empathy Score:** {emp_score}\n\n"
        "**Reasoning:**\n"
        f"- **Cultivating Change Talk Reasoning:** {cct_reason}\n"
        f"- **Softening Sustain Talk Reasoning:** {sst_reason}\n"
        f"- **Partnership Reasoning:** {part_reason}\n"
        f"- **Empathy Reasoning:** {emp_reason}\n\n"
        "**Behavior Code Counts:**\n"
        f"- **CR Count:** {counts['CR']}\n"
        f"- **SR Count:** {counts['SR']}\n"
        f"- **Q Count:** {counts['Q']}\n"
        f"- **GI Count:** {counts['GI']}\n"
        f"- **Persuade Count:** {counts['Persuade']}\n"
        f"- **Persuade with Count:** {counts['Persuade with']}\n"
        f"- **AF Count:** {counts['AF']}\n"
        f"- **Seek Count:** {counts['Seek']}\n"
        f"- **Emphasize Count:** {counts['Emphasize']}\n"
        f"- **Confront Count:** {counts['Confront']}\n\n"
        "**Behavior Code Percentages:**\n"
        f"- **Percent CR:** {metrics['percent_cr']}%\n"
        f"- **Reflect Question Ratio:** {metrics['reflect_question_ratio']}%\n"
        f"- **Total MI Adherent:** {metrics['total_adherent']}\n"
        f"- **Total MI Non-Adherent:** {metrics['total_non_adherent']}\n\n"
        f"**Technical Global Score:** {tech_score}\n"
        f"**Relational Global Score:** {rel_score}\n\n"
        "**Counseling Transcript with Reasoning for MITI Behavior Encoding & Persona Characteristics (User = Participant/Clinician; SimPatient = SimPatient/Client):**\n"
        f"{transcript_annotated}\n\n"
        f"**MITI 4.2.1 Coding Manual:**\n{MITI_MANUAL_TEXT}\n\n"
        "Output JSON Format:\n"
        '{"summary": "<Your 1-2 paragraph session summary here>"}'
    )

    chain = _build_model_with_fallback(SessionSummaryOutput, temperature=1.0)
    result: SessionSummaryOutput = chain.invoke(
        [HumanMessage(content=prompt)], config=config
    )  # type: ignore[assignment]

    summary_text = (result.summary or "").strip()
    logger.info(f"SimPatient Session Summary generated: {summary_text[:150]}...")

    # Write as past_session_history into runtime.store
    mem_item = store.get(PATIENT_MEMORY_NS, "simpatient_memory")
    existing_val = mem_item.value.copy() if mem_item and mem_item.value else {}
    existing_val["past_session_history"] = summary_text
    existing_val["behavior_counts"] = metrics
    store.put(PATIENT_MEMORY_NS, "simpatient_memory", existing_val)

    return {"simpatient_session_summary": summary_text}


# ===========================================================================
# 3. Between-Session Event Node (GenerateBetweenSessionEventResponse.cs)
# ===========================================================================
@traced_span("simpatient_between_session_event")
def simpatient_between_session_event_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """
    Simulate a between-session life event using the direct patient profile, internal states,
    and previous session transcript.
    """
    logger.info("simpatient_between_session_event_node invoked")

    configurable = config.get("configurable", {})
    patient_profile_data_dict = state.get("patient_profile_data") or configurable.get("patient_profile_data")
    if not patient_profile_data_dict:
        raise ValueError("simpatient_between_session_event_node: patient_profile_data missing.")
    profile = PatientProfileData.model_validate(patient_profile_data_dict)

    current_session = state.get("current_session", {})
    messages = current_session.get("messages", [])
    conversation_text = format_conversation_history(messages)

    store = runtime.store
    if store is None:
        raise RuntimeError("simpatient_between_session_event_node: runtime.store is None.")

    mem_item = store.get(PATIENT_MEMORY_NS, "simpatient_memory")
    patient_control = 5
    patient_efficacy = 5
    patient_awareness = 5
    patient_reward = 5

    if mem_item and mem_item.value:
        patient_control = mem_item.value.get("patient_control", 5)
        patient_efficacy = mem_item.value.get("patient_efficacy", 5)
        patient_awareness = mem_item.value.get("patient_awareness", 5)
        patient_reward = mem_item.value.get("patient_reward", 5)

    profile_text = _format_profile_section(profile)
    target_behavior = profile.behavior or profile.topic or "substance misuse"

    prompt = f"""Simulate a Between-Session Event for a Patient struggling with {target_behavior}.

{profile_text}

**Cognitive & Internal State Characteristics:**
- ** Control Level: {patient_control}
    - Explanation: Your level of ability to regulate your own thoughts, emotions, and actions (1-10 scale).
    - High Levels (Example Score: 10): "I'm pretty good at controlling myself. I don't really have a problem saying no to a drink."
    - Low Levels (Example Score: 1): "I just can't seem to stop myself once I start drinking. It's like something takes over."
- ** Self-Efficacy Level: {patient_efficacy}
    - Explanation: Your level of confidence in your ability to resist cravings, cope with triggers, and achieve your recovery goals. (1-10 scale)
    - High Levels (Example Score: 10): "I'm confident I can handle any situation without needing alcohol. I've got this."
    - Low Levels (Example Score: 1): "I don't think I can do this. Alcohol has such a hold on me, I always go back to it."
- ** Awareness Level: {patient_awareness}
    - Explanation: Your level of ability to accurately perceive and evaluate your own thoughts, feelings, and behaviors. (1-10 scale)
    - High Levels (Example Score: 10): "I'm completely aware of the effects alcohol has on me and how it impacts my life."
    - Low Levels (Example Score: 1): "I don't really see what the big deal is. I can stop drinking anytime I want."
- ** Reward Level: {patient_reward}
    - Explanation: The level in which substance and its cues trigger cravings and automatic behaviors in you. (1-10 scale)
    - High Levels (Example Score: 10): "Honestly, just the smell of beer makes me crave a cold one. It's instant relaxation."
    - Low Levels (Example Score: 1): "Alcohol doesn't really do much for me anymore. It just makes me feel sick."

## Previous Session Conversation:
{conversation_text}

## Event Description:
Based on the patient's profile and the content of their last therapy session, describe a realistic and probable event that could have happened to the patient since then. The event should be related to their {target_behavior} and recovery journey.

Make sure to keep it simple and to the point. The event description should be under 128 tokens.

Output JSON Format:
{{"event": "<between session event text>"}}
"""

    chain = _build_model_with_fallback(SimPatientBetweenSessionEventOutput, temperature=1.0)
    result: SimPatientBetweenSessionEventOutput = chain.invoke(
        [HumanMessage(content=prompt)], config=config
    )  # type: ignore[assignment]

    event_text = result.event.strip() if getattr(result, "event", None) else ""
    logger.info(f"SimPatient Between-Session Event generated: {event_text}")

    # Persist in store for subsequent session
    existing_val = mem_item.value.copy() if mem_item and mem_item.value else {}
    existing_val["between_session_event"] = event_text
    store.put(PATIENT_MEMORY_NS, "simpatient_memory", existing_val)

    return {"simpatient_between_session_event": event_text}


@traced_span("simpatient_inter_sessions")
def simpatient_inter_sessions_node(
    state: Dict[str, Any], config: RunnableConfig, runtime: Runtime
) -> Dict[str, Any]:
    """Pass-through node to align with Phoenix multi-session tracing hierarchy."""
    logger.info("simpatient_inter_sessions_node invoked")
    return {}
