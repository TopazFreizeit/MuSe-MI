"""
Unified Latent State Updater Agent.

Handles evaluating the patient's latent variables in a single LLM call.
Produces deltas that update the patient's state before they speak.
"""

import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage

from ..utils import get_llm_model
from ..utils.constants import THERAPIST_PREFIX, PATIENT_PREFIX
from ..utils.tracing import enrich_span, traced_span
from .patient_state_manager_config import (
    PROVIDER,
    MODEL_NAME,
    FALLBACK_MODEL_NAME,
    TEMPERATURE,
    API_MAX_RETRIES,
    OPENROUTER_PROVIDER_PREFS,
)
from .patient_state_manager_dtos import (
    PatientStateUpdateDTO,
)
from ..patient.patient_dtos import PatientProfileData
from ..graphs.graph_config import LatentTrajectoryEntry
from .unified_state_updater_prompt import generate_unified_state_updater_prompt
from .cognitive_simulator import DeterministicCognitiveSimulator

logger = logging.getLogger(__name__)

def _create_structured_chain():
    """Create a structured-output chain for unified latent state updates."""
    primary = (
        get_llm_model(
            temperature=TEMPERATURE,
            provider=PROVIDER,
            model_name=MODEL_NAME,
            provider_preferences=OPENROUTER_PROVIDER_PREFS
        )
        .with_structured_output(PatientStateUpdateDTO, method="json_mode")
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    return primary

@traced_span("cognitive_interpreter")
def unified_state_update_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Evaluates the most recent therapist intervention to output delta shifts for
    all latent variables. Applies the Deterministic Cognitive Simulator mathematical models.
    """
    try:
        # Extract needed context from config. Fail fast if missing.
        configurable = config["configurable"]
        
        raw_profile = configurable["patient_profile_data"]
        patient_profile_data = PatientProfileData(**raw_profile) if isinstance(raw_profile, dict) else raw_profile

        prior_latent_variables = state["patient_latent_variables"]

        # Convert prior latent variables to dict format expected by the prompt
        prior_lv_dict = prior_latent_variables.model_dump() if hasattr(prior_latent_variables, "model_dump") else prior_latent_variables

        # Collect the system prompt
        prompt = generate_unified_state_updater_prompt(
            prior_latent_variables=prior_lv_dict,
            profile_data=patient_profile_data,
        )

        updated_latent_variables = prior_lv_dict
        reasoning_snapshot = None

        # Get the recent conversational exchange (Patient -> Therapist)
        messages = state["messages"]
        if len(messages) < 3:
            logger.info("Not enough messages for a patient->therapist exchange; returning unshifted state.")
        else:
            from ..patient.patient_config import PATIENT_CONTEXT_WINDOW_TURNS

            # Limit to the last N turns (1 turn = approx 2 messages)
            max_messages = PATIENT_CONTEXT_WINDOW_TURNS * 2
            recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

            history_text = ""
            for msg in recent_messages[:-2]:
                prefix = PATIENT_PREFIX if msg.name == "Patient" else THERAPIST_PREFIX
                history_text += f"{prefix}{msg.content}\n"
                
            recent_exchange = ""
            # Grab the last 2 messages (Previous Patient turn + Latest Therapist turn)
            patient_msg = messages[-2]
            therapist_msg = messages[-1]
            recent_exchange += f"{PATIENT_PREFIX}{patient_msg.content}\n"
            recent_exchange += f"{THERAPIST_PREFIX}{therapist_msg.content}\n"

            chain = _create_structured_chain()

            # Invoke LLM (Semantic Sensor)
            messages_to_send = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"<conversation_history>\n{history_text.strip()}\n</conversation_history>\n\n<latest_exchange>\n{recent_exchange.strip()}\n</latest_exchange>")
            ]
            
            dto: PatientStateUpdateDTO = chain.invoke(messages_to_send)

            updated_latent_variables = DeterministicCognitiveSimulator.apply_math(
                prior_state=prior_lv_dict,
                dto=dto,
            )

            logger.info(f"Latent State Update - Prior: {prior_lv_dict} | New: {updated_latent_variables}")

            reasoning_snapshot = {
                "general": dto.reasoning_general,
                "anger":                  {"reasoning": dto.anger.reasoning,                  "impact": dto.anger.semantic_impact.value},
                "self_efficacy":          {"reasoning": dto.self_efficacy.reasoning,          "impact": dto.self_efficacy.semantic_impact.value},
                "problem_recognition":    {"reasoning": dto.problem_recognition.reasoning,    "impact": dto.problem_recognition.semantic_impact.value},
                "motivational_readiness": {"reasoning": dto.motivational_readiness.reasoning, "impact": dto.motivational_readiness.semantic_impact.value},
            }

            enrich_span(
                anger_impact=dto.anger.semantic_impact.value,
                se_impact=dto.self_efficacy.semantic_impact.value,
                pr_impact=dto.problem_recognition.semantic_impact.value,
                mr_impact=dto.motivational_readiness.semantic_impact.value,
            )

        # Snapshot of all latent scores after this update (analytics)
        turn_number = state.get("turn_count", 0)
        snapshot = {
            "turn_number": turn_number,
            "anger": updated_latent_variables["anger"],
            "self_efficacy": updated_latent_variables["self_efficacy"],
            "problem_recognition": updated_latent_variables["problem_recognition"],
            "motivational_readiness": updated_latent_variables["motivational_readiness"],
        }
        
        snapshots = list(state.get("latent_variable_logs", []))
        snapshots.append(snapshot)

        session_number = state.get("session_number", 1)
        impact_details = None
        if reasoning_snapshot is not None:
            impact_details = {
                "impacts": {
                    "anger": dto.anger.semantic_impact.value,
                    "self_efficacy": dto.self_efficacy.semantic_impact.value,
                    "problem_recognition": dto.problem_recognition.semantic_impact.value,
                    "motivational_readiness": dto.motivational_readiness.semantic_impact.value,
                }
            }

        trajectory_entry: LatentTrajectoryEntry = {
            "step_index": -1,
            "component": "cognitive_interpreter",
            "session_number": session_number,
            "turn_number": turn_number,
            "anger": float(updated_latent_variables["anger"]),
            "self_efficacy": float(updated_latent_variables["self_efficacy"]),
            "problem_recognition": float(updated_latent_variables["problem_recognition"]),
            "motivational_readiness": float(updated_latent_variables["motivational_readiness"]),
            "details": impact_details,
        }
        session_trajectory = list(state.get("session_latent_trajectory", []))
        session_trajectory.append(trajectory_entry)

        result = {
            "patient_latent_variables": updated_latent_variables,
            "latent_variable_logs": snapshots,
            "session_latent_trajectory": session_trajectory,
        }
        
        if reasoning_snapshot is not None:
            result["patient_state_reasoning"] = reasoning_snapshot

        return result

    except Exception as e:
        logger.error(f"Failed to generate unified state update: {e}", exc_info=True)
        raise
