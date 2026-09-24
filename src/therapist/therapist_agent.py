"""
Therapist agent node for LangGraph — 3-agent sequential pipeline.

Clinical Analyzer → Clinical Strategist → Formulator
Each agent runs sequentially (each depends on the prior output) and appears
as a separate named CHAIN span in Phoenix via @traced_span.
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from typing import Dict, Any, List, Optional
import logging
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from ..utils import get_llm_model
from ..utils.constants import THERAPIST_PREFIX
from ..utils.formatting import (
    format_speaker_utterance,
    transform_messages_for_therapist,
)
from ..utils.tracing import enrich_span, traced_span
from ..utils.store import THERAPIST_MEMORY_NS
from ..graphs.graph_config import MAX_TURNS
from . import therapist_config
from .therapist_dtos import (
    ClinicalAnalyzerOutput,
    ClinicalStrategistOutput,
    FormulatorOutput,
)
from .therapist_planner_prompts import (
    generate_clinical_analyzer_prompt,
    generate_strategist_prompt,
    generate_formulator_prompt,
    get_session_objective,
    MI_TECHNIQUES,
)

logger = logging.getLogger(__name__)

TECHNIQUE_RECENT_RESET_SIZE = 9


def _build_model_with_fallback(temperature: float, dto):
    """Build a structured-output model chain (primary + fallback) for the given DTO."""
    primary = (
        get_llm_model(
            temperature=temperature,
            provider=therapist_config._config.PROVIDER,
            model_name=therapist_config._config.MODEL_NAME,
            provider_preferences=therapist_config._config.OPENROUTER_PROVIDER_PREFS,
            max_tokens=therapist_config.MAX_COMPLETION_TOKENS,
        )
        .with_structured_output(dto, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=therapist_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    fallback = (
        get_llm_model(
            temperature=temperature,
            provider=therapist_config._config.PROVIDER,
            model_name=therapist_config._config.FALLBACK_MODEL_NAME,
            max_tokens=therapist_config.MAX_COMPLETION_TOKENS,
        )
        .with_structured_output(dto, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=therapist_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    return primary.with_fallbacks([fallback])


@traced_span("therapist_clinical_analyzer")
def _run_clinical_analyzer(
    state: Dict[str, Any],
    messages: List[BaseMessage],
    temperature: float,
    config: RunnableConfig,
) -> ClinicalAnalyzerOutput:
    """Diagnoses patient state and extracts new memory from the last exchange."""
    model = _build_model_with_fallback(temperature, dto=ClinicalAnalyzerOutput)
    return model.invoke(messages, config=config)


@traced_span("therapist_strategist")
def _run_strategist(
    state: Dict[str, Any],
    messages: List[BaseMessage],
    temperature: float,
    config: RunnableConfig,
) -> ClinicalStrategistOutput:
    """Selects the optimal MI technique and produces a macro instruction."""
    model = _build_model_with_fallback(temperature, dto=ClinicalStrategistOutput)
    return model.invoke(messages, config=config)


@traced_span("therapist_formulator")
def _run_formulator(
    state: Dict[str, Any],
    messages: List[BaseMessage],
    temperature: float,
    config: RunnableConfig,
) -> FormulatorOutput:
    """Translates the strategist's macro instruction into natural spoken language."""
    model = _build_model_with_fallback(temperature, dto=FormulatorOutput)
    return model.invoke(messages, config=config)


@traced_span("therapist")
def therapist_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    """
    Therapist node — runs the 3-agent pipeline:
      1. Clinical Analyzer (turns 2+): diagnoses patient state, updates working memory.
      2. Clinical Strategist: selects MI technique and produces macro instruction.
      3. Formulator: renders the macro instruction as natural therapist speech.
    """
    try:
        logger.info("therapist_node invoked")

        configurable = config.get("configurable", {})
        current_session_number = configurable.get("current_session_number", 1)

        turn_number = state.get("turn_count", 0) + 1
        is_first = len(state["messages"]) == 0

        enrich_span(
            node_name="therapist",
            description=f"Therapist {'opens' if is_first else 'responds in'} session {current_session_number}, turn {turn_number}",
            session_number=current_session_number,
            turn_number=turn_number,
            speaker="therapist",
            phase="conversation",
            purpose="generate_response",
            model_provider=therapist_config._config.PROVIDER,
            model_name=therapist_config._config.MODEL_NAME,
        )

        temperature = therapist_config._config.get_temperature()
        last_used_techniques = state.get("last_used_techniques", [])

        # ── Read therapist working memory from store ─────────────────────────
        biographical_item = runtime.store.get(THERAPIST_MEMORY_NS, "biographical_facts")
        values_item = runtime.store.get(THERAPIST_MEMORY_NS, "core_values")
        target_item = runtime.store.get(THERAPIST_MEMORY_NS, "target_behavior")
        darn_item = runtime.store.get(THERAPIST_MEMORY_NS, "darn_change_talk")
        cat_item = runtime.store.get(THERAPIST_MEMORY_NS, "cat_change_talk")
        sustain_item = runtime.store.get(THERAPIST_MEMORY_NS, "sustain_talk_themes")
        plan_item = runtime.store.get(THERAPIST_MEMORY_NS, "change_plan")
        focus_item = runtime.store.get(THERAPIST_MEMORY_NS, "current_clinical_focus")

        biographical_facts = list(biographical_item.value.get("items", [])) if biographical_item else []
        core_values = list(values_item.value.get("items", [])) if values_item else []
        target_behavior = str(target_item.value.get("value", "")) if target_item else "Unknown"
        darn_change_talk = list(darn_item.value.get("items", [])) if darn_item else []
        cat_change_talk = list(cat_item.value.get("items", [])) if cat_item else []
        sustain_talk_themes = list(sustain_item.value.get("items", [])) if sustain_item else []
        change_plan = list(plan_item.value.get("items", [])) if plan_item else []
        current_clinical_focus = str(focus_item.value.get("value", "")) if focus_item else "None"

        all_msgs = state["messages"]
        windowed_msgs = all_msgs[-therapist_config.CONTEXT_WINDOW*2:] if len(all_msgs) > therapist_config.CONTEXT_WINDOW*2 else all_msgs

        if is_first:
            conversation_context = "No prior history before the current exchange."
        else:
            conversation_context = transform_messages_for_therapist(windowed_msgs)

        # ---------------------------------------------------------------
        # Step 1: Clinical Analyzer (skip on the opening turn — no prior
        # patient response to analyze yet)
        # ---------------------------------------------------------------
        analyzer_result: Optional[ClinicalAnalyzerOutput] = None
        if is_first or len(state["messages"]) < 2:
            if turn_number == 1:
                logger.info("Opening turn of the very first session — no prior messages to analyze for Clinical Analyzer.")
            else:
                logger.error("Insufficient message history for Clinical Analyzer on turn %d: only %d messages found.",
                             turn_number, len(state["messages"]))
                raise ValueError("Clinical Analyzer requires at least one prior therapist message and one prior patient message to analyze.")
        else:
            analyzer_system_prompt = generate_clinical_analyzer_prompt()
            
            analyzer_messages = [
                SystemMessage(content=analyzer_system_prompt),
                HumanMessage(content=conversation_context),
            ]
            analyzer_result = _run_clinical_analyzer(state, analyzer_messages, temperature, config)

            logger.info(
                f"[Analyzer] stage={analyzer_result.patient_diagnosis.stage_of_change}"
            )

        # ---------------------------------------------------------------
        # Step 2: Dynamic instructions (beginning / end of session)
        # ---------------------------------------------------------------
        beginning_strategist_alert, beginning_formulator_alert, beginning_technique = get_beginning_session_instructions(
            current_session_number, turn_number,
        )
        end_strategist_alert, end_formulator_alert, end_technique = get_end_session_instructions(turn_number, current_session_number)

        # End-session overrides take priority over beginning-session overrides.
        forced_technique: Optional[str] = end_technique or beginning_technique

        # ---------------------------------------------------------------
        # Step 3: Clinical Strategist (skipped when a technique is forced)
        # ---------------------------------------------------------------
        strategist_result: Optional[ClinicalStrategistOutput] = None

        if forced_technique is not None:
            logger.info(
                "[Strategist] SKIPPED — using pre-determined technique=%s (beginning=%s, end=%s)",
                forced_technique, beginning_technique, end_technique,
            )
            # Create a dummy strategist result to satisfy the Formulator constraints
            strategist_result = ClinicalStrategistOutput(
                strategist_reasoning_trace=[f"Forced technique override: {forced_technique}"],
                current_mi_phase="Engaging" if is_first else "Planning",
                macro_intent="Execute forced technique to manage session boundaries.",
                allowed_techniques=[forced_technique]
            )
        else:
            current_session_turns_for_count: list = state.get("current_session_turns", [])
            mobilizing_ct_count_this_session = sum(
                1 for t in current_session_turns_for_count
                if t.get("speaker") == "therapist"
                and t.get("commitment_and_action_talk")
                and t.get("commitment_and_action_talk").lower() != "none"
            )

            available_techniques = [
                t for t in MI_TECHNIQUES.keys()
            ]

            # Ensure GI is available if we are waiting for follow-through on advice
            if last_used_techniques and last_used_techniques[-1] in ("ADP", "RCP") and "GI" not in available_techniques:
                available_techniques.append("GI")

            strategist_system_prompt = generate_strategist_prompt(
                session_number=current_session_number,
                biographical_facts=biographical_facts,
                core_values=core_values,
                target_behavior=target_behavior,
                darn_change_talk=darn_change_talk,
                cat_change_talk=cat_change_talk,
                sustain_talk_themes=sustain_talk_themes,
                change_plan=change_plan,
                current_clinical_focus=current_clinical_focus,
                patient_diagnosis=analyzer_result,
                mobilizing_ct_count_this_session=mobilizing_ct_count_this_session,
                available_techniques=available_techniques,
                previous_technique=last_used_techniques[-1] if last_used_techniques else None,
            )

            strategist_messages: List[BaseMessage] = [
                SystemMessage(content=strategist_system_prompt),
                SystemMessage(content=get_session_objective(current_session_number)),
            ]
            
            if is_first:
                human_content = "[Session is opening — formulate opening strategy.]"
            else:
                human_content = f"{conversation_context}\n\nFormulate the macro clinical strategy based on the Patient Diagnosis, Memory Context, and the Recent Exchange."
                
            if beginning_strategist_alert:
                human_content += f"\n\n### SYSTEM ALERT:\n{beginning_strategist_alert}"
            if end_strategist_alert:
                human_content += f"\n\n### SYSTEM ALERT:\n{end_strategist_alert}"
                
            strategist_messages.append(HumanMessage(content=human_content))

            strategist_result = _run_strategist(state, strategist_messages, temperature, config)
            logger.info(
                "[Strategist] MI Phase: %s",
                strategist_result.current_mi_phase,
            )

        # ---------------------------------------------------------------
        # Step 4: Formulator
        # ---------------------------------------------------------------
        formulator_system_prompt = generate_formulator_prompt(
            macro_intent=strategist_result.macro_intent,
            allowed_techniques=strategist_result.allowed_techniques,
            biographical_facts=biographical_facts,
            core_values=core_values,
            target_behavior=target_behavior,
            darn_change_talk=darn_change_talk,
            cat_change_talk=cat_change_talk,
            sustain_talk_themes=sustain_talk_themes,
            change_plan=change_plan,
            current_clinical_focus=current_clinical_focus,
            patient_diagnosis=analyzer_result,
        )

        if is_first:
            human_content = "[Session is opening — no prior conversation.]"
        else:
            human_content = conversation_context
            
        if beginning_formulator_alert:
            logger.info("[DYNAMIC INSTRUCTION] BEGINNING SESSION active for Formulator — session %d, turn %d",
                        current_session_number, turn_number)
            human_content += f"\n\n### SYSTEM ALERT:\n{beginning_formulator_alert}"
        if end_formulator_alert:
            logger.info("[DYNAMIC INSTRUCTION] END SESSION active for Formulator — session %d, turn %d, turns_remaining=%d",
                        current_session_number, turn_number, MAX_TURNS - turn_number)
            human_content += f"\n\n### SYSTEM ALERT:\n{end_formulator_alert}"

        formulator_messages: List[BaseMessage] = [
            SystemMessage(content=formulator_system_prompt),
            SystemMessage(content=get_session_objective(current_session_number)),
            HumanMessage(content=human_content),
        ]

        formulator_result = _run_formulator(state, formulator_messages, temperature, config)
        response_content = formulator_result.final_response
        technique = formulator_result.chosen_technique
        logger.info("[Formulator] chosen_technique=%s", technique)

        # ---------------------------------------------------------------
        # Build turn record
        # ---------------------------------------------------------------
        therapist_message = AIMessage(content=response_content, name="Therapist")

        summary_line = format_speaker_utterance(response_content, THERAPIST_PREFIX)
        logger.info(f"\n{summary_line}")

        current_turns = state.get("current_session_turns", [])
        turn_record: dict = {
            "turn_number": len(current_turns) + 1,
            "speaker": "therapist",
            "volley": response_content,
            "technique": technique,
        }

        if analyzer_result:
            diag = analyzer_result.patient_diagnosis
            turn_record["patient_state_classification"] = diag.stage_of_change
            turn_record["resistance_detected"] = diag.resistance_detected
            turn_record["preparatory_change_talk"] = diag.preparatory_change_talk
            turn_record["commitment_and_action_talk"] = diag.commitment_and_action_talk
            turn_record["classification_reasoning"] = diag.classification_reasoning
            turn_record["patient_is_stuck"] = diag.patient_is_stuck
            turn_record["ready_for_planning"] = diag.ready_for_planning
            turn_record["patient_requests_guidance_or_info"] = diag.patient_requests_guidance_or_info

        if strategist_result:
            turn_record["current_mi_phase"] = strategist_result.current_mi_phase
            turn_record["macro_intent"] = strategist_result.macro_intent
            turn_record["allowed_techniques"] = strategist_result.allowed_techniques

        turn_record["formulator_reasoning"] = formulator_result.reasoning_chain if formulator_result else []
        turn_record["strategic_reasoning"] = strategist_result.strategist_reasoning_trace if strategist_result else [f"Technique pre-determined by turn number: {turn_number}, instruction: {forced_technique}"]

        # ---------------------------------------------------------------
        # Update technique tracking
        # ---------------------------------------------------------------
        last_used_techniques = last_used_techniques + [technique]
        if len(last_used_techniques) > TECHNIQUE_RECENT_RESET_SIZE:
            last_used_techniques = last_used_techniques[-TECHNIQUE_RECENT_RESET_SIZE:]

        # ---------------------------------------------------------------
        # Build return dict
        # ---------------------------------------------------------------
        update_dict = {
            "messages": state["messages"] + [therapist_message],
            "current_speaker": "therapist",
            "turn_count": state.get("turn_count", 0) + 1,
            "current_session_turns": current_turns + [turn_record],
            "miin_injection_active": state.get("miin_injection_active", False),
            "miin_injection_turn_numbers": state.get("miin_injection_turn_numbers"),
            "last_used_techniques": last_used_techniques,
        }
        return update_dict
    except Exception as e:
        logger.exception(f"Therapist node failed: {e}")
        raise


def get_end_session_instructions(turn_number: int, session_number: int) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns (strategist_alert_text, formulator_alert_text, technique_override).

    Fires exactly once per end-session phase:
      0 < turns_remaining <= 4: Alert Strategist to wrap up gracefully. Alert Formulator to append a time warning.
      turns_remaining == 0: Alert both to close the session gracefully.
    """
    turns_remaining = MAX_TURNS - turn_number

    if turns_remaining == 0:
        last_sentence = (
            "- Tell the patient you look forward to seeing them next time (without referencing specific session numbers or dates to avoid confusion in later sessions).\n"
            if session_number != 4
            else "- Acknowledge that this is your final session together and express appreciation for the work you've done together.\n"
        )
        return (
            (
                "\n## LAST TURN\n"
                "- Instruct the Formulator to end the session gracefully and empathetically.\n"
                "- Remind the Formulator NOT to end with a question.\n"
            ),
            (
                "\n## LAST TURN\n"
                "- End the session gracefully and empathetically.\n"
                "- You must explicitly state that the session is now over as part of your response.\n"
                f"{last_sentence}"
            ),
            None,  # Do not force a technique
        )

    if 0 < turns_remaining <= 4:
        return (
            (
                f"\n## SESSION CLOSING WARNING ({turns_remaining} turns left)\n"
                "- The session is ending soon. Do NOT open new topics for exploration.\n"
                "- Instruct the Formulator to give a verbal time warning.\n"
                "- Find a natural moment to use Summarize (SM) to consolidate progress or Affirm (AF) to reinforce change talk.\n"
                "- Do not rush; let the patient finish their thought. You are managing the clock.\n"
            ),
            (
                f"\n## SESSION CLOSING WARNING ({turns_remaining} turns left)\n"
                "- The session is ending soon. Do NOT open new topics for exploration.\n"
                "- Find a natural moment to signal the patient that the session is nearing its end.\n"
            ),
            None,  # Strategist remains free to choose
        )

    return None, None, None


def get_beginning_session_instructions(
    session_number: int,
    turn_number: int,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns (strategist_alert_text, formulator_alert_text, technique_override)."""

    if turn_number > 3:
        return None, None, None

    if session_number == 1:
        alert = (
            "## Session 1 — Opening\n"
            "This is the very first turn of the entire therapy program.\n"
            "1. Introduce yourself.\n"
            "2. Briefly name the structure: four sessions together, focused on exploring "
            "the patient's relationship with the target behavior at their own pace.\n"
            "3. Invite the patient to start by telling you a bit about themselves."
        )
        return (alert, alert, None)

    if session_number == 4:
        alert = (
            f"## Session 4 — Opening & Final Review\n"
            "You are opening the final session of the therapy program, exactly one week after your last meeting.\n"
            "1. Welcome the patient back warmly.\n"
            "2. Explicitly acknowledge that this is your final session together.\n"
            "3. Ask an open question to check in on how their week went.\n"        )
        return (alert, alert, None)

    # Sessions 2 and 3 
    alert = (
        f"## Session {session_number} — Opening (Temporal Check-in)\n"
        f"You are opening Session {session_number}, exactly one week after your last meeting.\n"
        "1. Welcome the patient back warmly in one short sentence.\n"
        "2. Explicitly acknowledge the passage of time.\n"
        "3. Reference something specific they shared last session to show clinical continuity.\n"
        "4. Ask an open question about how their week has been.\n"
    )
    return (alert, alert, None)