"""
State persistence for multi-session therapy runs.
Saves final state with metadata for later analysis.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from ..patient.patient_config import PatientConfigClass
from ..therapist.therapist_config import TherapistConfigClass
from ..therapist import therapist_config as therapist_config_module
from ..utils.config import (
    ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES, 
    DISABLE_COGNITIVE_INTERPRETER_IMPACT,
    DISABLE_PATIENT_MEMORY,
    ABLATE_LIFE_EVENT_SIMULATOR, 
    CONSISTENT_CLIENT_BASELINE, 
    CAMI_THERAPIST_BASELINE, 
    VANILLA_BASELINE,
    SIMPATIENT_BASELINE,
    PATIENT_PSI_BASELINE,
    PATIENT_PSI_STYLE,
    SHIFT_DELTA_MULTIPLIER,
    CAUSAL_RECORD_BASELINE,
    CAUSAL_SHUFFLED_ROLLOUT,
    CAUSAL_ROLLOUT_SEED,
)

logger = logging.getLogger(__name__)


def save_final_state(state: Dict[str, Any], output_dir: str = "data/runs") -> str:
    """
    Save final MultiSessionState with metadata for later analysis.
    
    Args:
        state: MultiSessionState dictionary from graph execution
        output_dir: Directory to save state files
    
    Returns:
        Path to saved JSON file
    """
    # Get patient and therapist config for filename
    patient_config = PatientConfigClass()
    therapist_config = TherapistConfigClass()
    patient_temp = patient_config.get_temperature()
    therapist_temp = therapist_config.get_temperature()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_id = state.get("therapy_program_trace_id", timestamp)
    
    # Create simple filename with trace id
    filename = f"{trace_id}.json"
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data with metadata per module
    data = {
        "metadata": {
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "patient": {
                "provider": patient_config.PROVIDER,
                "model": patient_config.MODEL_NAME,
                "temperature": patient_temp,
            },
            "therapist": {
                "provider": therapist_config.PROVIDER,
                "model": therapist_config.MODEL_NAME,
                "temperature": therapist_temp
            },
            "miin_injection": {
                "enabled": therapist_config_module.MIIN_INJECTION_ENABLED,
                "sessions": therapist_config_module.MIIN_INJECTION_SESSIONS,
                "min_max_turn": therapist_config_module.MIIN_INJECTION_MIN_MAX_TURN,
                "count": therapist_config_module.MIIN_INJECTION_COUNT,
                "schedule": state.get("miin_injection_schedule", {})
            },
            "ablation_study": {
                "static_hidden_variables": ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES,
                "disable_cognitive_interpreter_impact": DISABLE_COGNITIVE_INTERPRETER_IMPACT,
                "disable_patient_memory": DISABLE_PATIENT_MEMORY,
                "remove_life_event_simulator": ABLATE_LIFE_EVENT_SIMULATOR,
                "consistent_client_baseline": CONSISTENT_CLIENT_BASELINE,
                "vanilla_baseline": VANILLA_BASELINE,
                "simpatient_baseline": SIMPATIENT_BASELINE,
                "patient_psi_baseline": PATIENT_PSI_BASELINE,
                "patient_psi_style": PATIENT_PSI_STYLE,
                "cami_therapist_baseline": CAMI_THERAPIST_BASELINE,
                "shift_delta_multiplier": SHIFT_DELTA_MULTIPLIER,
                "causal_record_baseline": CAUSAL_RECORD_BASELINE,
                "causal_shuffled_rollout": CAUSAL_SHUFFLED_ROLLOUT,
                "causal_rollout_seed": CAUSAL_ROLLOUT_SEED,
            },
        },
        "state": _serialize_state(state)
    }
    
    # Save to file
    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved final state to {filepath}")
    return str(filepath)


def _normalize_trajectory(trajectory: list) -> list:
    """Normalize step indices of latent variable trajectory to continuous 0-based sequence."""
    normalized = []
    for idx, item in enumerate(trajectory):
        if isinstance(item, dict):
            entry = dict(item)
            entry["step_index"] = idx
            normalized.append(entry)
        else:
            normalized.append(item)
    return normalized


def _serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize MultiSessionState, converting BaseMessage objects to dicts.
    
    Args:
        state: MultiSessionState dictionary
    
    Returns:
        JSON-serializable state dictionary
    """
    serialized = {}
    
    for key, value in state.items():
        if key == "current_session" and isinstance(value, dict):
            # Serialize nested ConversationState
            serialized[key] = _serialize_conversation_state(value)
        elif key == "all_session_transcripts":
            # Keep transcripts as-is (already strings)
            serialized[key] = value
        elif key == "latent_variable_trajectory" and isinstance(value, list):
            serialized[key] = _normalize_trajectory(value)
        else:
            serialized[key] = value
    
    return serialized


def _serialize_conversation_state(conv_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize ConversationState, handling BaseMessage objects.
    
    Args:
        conv_state: ConversationState dictionary
    
    Returns:
        JSON-serializable conversation state
    """
    serialized = {}
    
    for key, value in conv_state.items():
        if key == "messages":
            # Convert BaseMessage objects to dicts
            serialized[key] = [
                {
                    "content": msg.content,
                    "type": msg.__class__.__name__,
                    "name": getattr(msg, "name", None)
                }
                for msg in value
            ]
        else:
            serialized[key] = value
    
    return serialized
