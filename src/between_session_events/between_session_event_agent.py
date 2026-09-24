"""
Between-Session Event Generator Agent.

Generates a realistic life event that occurs between therapy sessions and appends
a narrative summary to the patient's between_session_occurrences in the graph state.
Also outputs a MacroShift to update the patient's latent variables.

Runs inter-session (after sessions 1–3 only), before the next session starts.
"""
import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from ..utils import get_llm_model
from ..utils.store import PATIENT_MEMORY_NS
from ..utils.formatting import format_conversation_history
from ..utils.tracing import enrich_span, traced_span
from ..patient.patient_dtos import PatientProfileData
from .between_session_dtos import BetweenSessionEventChunk, LatentStateSnapshot
from .between_sessions_event_prompt import build_between_session_event_prompt
from ..graphs.graph_config import LatentTrajectoryEntry
from .between_session_config import (
    PROVIDER,
    MODEL_NAME,
    FALLBACK_MODEL_NAME,
    TEMPERATURE,
    API_MAX_RETRIES,
    OPENROUTER_PROVIDER_PREFS,
)
from ..patient_state_manager.cognitive_simulator import DeterministicCognitiveSimulator
from ..utils.config import ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES

logger = logging.getLogger(__name__)


def _create_structured_chain():
    """Create a structured-output chain with retry + primary -> fallback routing."""
    primary = (
        get_llm_model(temperature=TEMPERATURE, provider=PROVIDER, model_name=MODEL_NAME, provider_preferences=OPENROUTER_PROVIDER_PREFS)
        .with_structured_output(BetweenSessionEventChunk, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    fallback = (
        get_llm_model(temperature=TEMPERATURE, provider=PROVIDER, model_name=FALLBACK_MODEL_NAME, provider_preferences=OPENROUTER_PROVIDER_PREFS)
        .with_structured_output(BetweenSessionEventChunk, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    return primary.with_fallbacks([fallback])


def _get_allowed_trigger_classes(self_efficacy: float, motivational_readiness: float, problem_recognition: float, anger: float) -> list[str]:
    # Categorize each variable into 0 (Low), 1 (Medium), 2 (High)
    # Higher is better for SE, MR, PR. Lower is better for Anger.
    def categorize_positive(val: float) -> int:
        if val <= 33.3: return 0
        if val <= 66.6: return 1
        return 2
        
    def categorize_negative(val: float) -> int:
        if val <= 33.3: return 2
        if val <= 66.6: return 1
        return 0

    se_cat = categorize_positive(self_efficacy)
    mr_cat = categorize_positive(motivational_readiness)
    pr_cat = categorize_positive(problem_recognition)
    anger_cat = categorize_negative(anger)

    categories = [se_cat, mr_cat, pr_cat, anger_cat]
    
    # Count frequencies
    counts = {0: 0, 1: 0, 2: 0}
    for c in categories:
        counts[c] += 1
        
    # Find dominant category, resolving ties towards the higher (more positive) category
    dominant_cat = 0
    max_count = 0
    for cat in [0, 1, 2]: # Iterating 0 to 2 means if there's a tie (e.g. 2 for 0, 2 for 2), the higher category (2) wins because it overwrites max_count if count >= max_count
        if counts[cat] >= max_count:
            max_count = counts[cat]
            dominant_cat = cat

    if dominant_cat == 0:
        return ["relapse", "partial_setback"]
    elif dominant_cat == 1:
        return ["partial_setback", "mundane_stressor", "near_miss"]
    else:
        return ["success", "mundane_stressor"]

@traced_span("life_event_simulator")
def between_session_event_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    """
    Generate a between-session life event for the patient.

    Reads the completed session transcript and patient profile, invokes the
    Life Simulator LLM, writes the event_summary to between_session_occurrences.
    Calculates and applies the MacroShift to the patient's latent state.

    Args:
        state: Parent MultiSessionState.
        config: RunnableConfig with configurable context.
        runtime: LangGraph Runtime providing access to the compiled Store.

    Returns:
        State update dict with past_event_titles, and updated patient_latent_variables.
    """
    current_session_number = state["current_session_number"]
    upcoming_session_number = current_session_number + 1

    logger.info(
        f"between_session_event_node: generating event between session "
        f"{current_session_number} and session {upcoming_session_number}"
    )

    enrich_span(
        node_name="life_event_simulator",
        description=f"Generate between-session event before session {upcoming_session_number}",
        session_number=current_session_number,
        phase="memory",
        purpose="between_session_event_generation",
        model_provider=PROVIDER,
        model_name=MODEL_NAME,
    )

    demographics = PatientProfileData.model_validate(state["patient_profile_data"])
        
    latent_vars = state["patient_latent_variables"]
    self_efficacy = latent_vars["self_efficacy"]
    problem_recognition = latent_vars["problem_recognition"]
    motivational_readiness = latent_vars["motivational_readiness"]
    anger = latent_vars["anger"]

    occurrences_item = runtime.store.get(PATIENT_MEMORY_NS, "between_session_occurrences")
    titles_item = runtime.store.get(PATIENT_MEMORY_NS, "past_event_titles")
    motivations_item = runtime.store.get(PATIENT_MEMORY_NS, "motivations")
    concerns_item = runtime.store.get(PATIENT_MEMORY_NS, "concerns")
    
    past_event_titles: list[str] = titles_item.value.get("titles", []) if titles_item else []
    current_occurrences: list[str] = occurrences_item.value.get("items", []) if occurrences_item else []
    
    motivations = motivations_item.value.get("items", []) if motivations_item else []
    concerns = concerns_item.value.get("items", []) if concerns_item else []
    evolving_decisional_balance = {
        "beliefs": "; ".join(concerns) if concerns else "(none recorded)",
        "motivation": "; ".join(motivations) if motivations else "(none recorded)"
    }
    
    conversation_text = format_conversation_history(state["current_session"]["messages"])

    allowed_trigger_classes = _get_allowed_trigger_classes(
        float(self_efficacy), float(motivational_readiness), float(problem_recognition), float(anger)
    )

    system_prompt = build_between_session_event_prompt(
        profile_data=demographics,
        past_event_titles=past_event_titles,
        between_session_occurrences=current_occurrences,
        evolving_decisional_balance=evolving_decisional_balance,
        anger=float(anger),
        self_efficacy=float(self_efficacy),
        problem_recognition=float(problem_recognition),
        motivational_readiness=float(motivational_readiness),
        allowed_trigger_classes=allowed_trigger_classes
    )
    
    chain = _create_structured_chain()
    
    # Run the prompt
    result: BetweenSessionEventChunk = chain.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Session Transcript:\n{conversation_text}"},
        ],
        config=config,
    )

    if not result.event_title:
        raise ValueError("Between-session event agent returned empty 'event_title'.")
    if not result.event_summary:
        raise ValueError("Between-session event agent returned empty 'event_summary'.")
    if not result.trigger_class:
        raise ValueError("Between-session event agent returned empty 'trigger_class'.")

    pre_shift_snapshot = LatentStateSnapshot(
        self_efficacy=float(self_efficacy),
        problem_recognition=float(problem_recognition),
        anger=float(anger),
        motivational_readiness=float(motivational_readiness),
    )
    bse_audit_record = {
        "session_pair": f"{current_session_number}->{upcoming_session_number}",
        "from_session": current_session_number,
        "to_session": upcoming_session_number,
        "trigger_class": result.trigger_class,
        "event_title": result.event_title,
        "latent_state_snapshot": pre_shift_snapshot.model_dump(),
    }

    # 1. Update Memory (Occurrences and Titles)
    event_summary = (
        f"[Between sessions {current_session_number} and {upcoming_session_number} — {result.event_title}]: "
        f"{result.event_summary}"
    )
    
    # between_session_occurrences is a replaced field. It only tracks the most recent event.
    updated_occurrences = current_occurrences + [event_summary]
    updated_titles = past_event_titles + [result.event_title]
    # Write back to store
    runtime.store.put(PATIENT_MEMORY_NS, "between_session_occurrences", {"items": updated_occurrences})
    runtime.store.put(PATIENT_MEMORY_NS, "past_event_titles", {"titles": updated_titles})

    # 2. Apply Macro-Shift to Latent Variables using Bounded Updates
    if ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES:
        logger.info("Ablation Mode: Skipping latent variable updates in between-session events")
        new_latent_vars = latent_vars.copy()
    else:
        new_latent_vars = latent_vars.copy()
        shift = result.macro_shift
        
        anger_delta = -DeterministicCognitiveSimulator._get_macro_shift_delta(shift.anger.semantic_impact)
        se_delta = DeterministicCognitiveSimulator._get_macro_shift_delta(shift.self_efficacy.semantic_impact)
        pr_delta = DeterministicCognitiveSimulator._get_macro_shift_delta(shift.problem_recognition.semantic_impact)
        mr_delta = DeterministicCognitiveSimulator._get_macro_shift_delta(shift.motivational_readiness.semantic_impact)

        new_latent_vars["anger"] = DeterministicCognitiveSimulator._bounded_update(
            float(anger), anger_delta
        )
        new_latent_vars["self_efficacy"] = DeterministicCognitiveSimulator._bounded_update(
            float(self_efficacy), se_delta
        )
        new_latent_vars["problem_recognition"] = DeterministicCognitiveSimulator._bounded_update(
            float(problem_recognition), pr_delta
        )
        new_latent_vars["motivational_readiness"] = DeterministicCognitiveSimulator._bounded_update(
            float(motivational_readiness), mr_delta
        )

        logger.info(
            f"Between-session event generated: '{result.event_title}' "
            f"(trigger_class={result.trigger_class}). "
            f"Macro-Shift Applied. Anger delta: {anger_delta}, SE delta: {se_delta}, PR delta: {pr_delta}, MR delta: {mr_delta}."
        )

    enrich_span(
        trigger_class=result.trigger_class,
        bse_self_efficacy=pre_shift_snapshot.self_efficacy,
        bse_motivational_readiness=pre_shift_snapshot.motivational_readiness,
    )

    # Append the new audit record to the existing list
    prior_bse_trigger_classes = state.get("bse_trigger_classes", [])
    updated_bse_trigger_classes = prior_bse_trigger_classes + [bse_audit_record]

    prior_trajectory = list(state.get("latent_variable_trajectory", []))
    from ..utils.config import DISABLE_COGNITIVE_INTERPRETER_IMPACT
    if not (ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES or DISABLE_COGNITIVE_INTERPRETER_IMPACT):
        bse_trajectory_entry: LatentTrajectoryEntry = {
            "step_index": -1,
            "component": "bse",
            "session_before": current_session_number,
            "session_after": upcoming_session_number,
            "anger": float(new_latent_vars["anger"]),
            "self_efficacy": float(new_latent_vars["self_efficacy"]),
            "problem_recognition": float(new_latent_vars["problem_recognition"]),
            "motivational_readiness": float(new_latent_vars["motivational_readiness"]),
            "details": {
                "trigger_class": result.trigger_class,
                "event_title": result.event_title,
                "macro_shift": {
                    "anger": shift.anger.semantic_impact.value,
                    "self_efficacy": shift.self_efficacy.semantic_impact.value,
                    "problem_recognition": shift.problem_recognition.semantic_impact.value,
                    "motivational_readiness": shift.motivational_readiness.semantic_impact.value,
                } if not ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES and result.macro_shift else None,
            },
        }
        prior_trajectory.append(bse_trajectory_entry)

    # Return to MultiSessionState
    return {
        "past_event_titles": updated_titles,
        "patient_latent_variables": new_latent_vars,
        "bse_trigger_classes": updated_bse_trigger_classes,
        "latent_variable_trajectory": prior_trajectory,
    }
