"""
Patient initialization from the unified profile extraction JSON.

Loads all patient data from consistent_client_profiles/profiles.jsonl
and constructs typed DTOs for use throughout the simulation.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from src.utils.config import CONSISTENT_CLIENT_BASELINE

from .patient_dtos import (
    PatientLatentVariables,
    PatientProfileData,
)

logger = logging.getLogger(__name__)

CONSISTENT_CLIENTS_DIR = Path(__file__).resolve().parent.parent.parent / "consistent_client_profiles"


def _normalize_score(raw_score: float, min_val: float, max_val: float) -> float:
    """Normalize a raw clinical score to a 1-100 continuous scale using linear interpolation."""
    normalized = ((raw_score - min_val) / (max_val - min_val)) * 99.0 + 1.0
    return max(1.0, min(100.0, normalized))


@dataclass
class PatientInitResult:
    """All patient profile data initialized from the unified profile JSONL."""
    latent_variables: PatientLatentVariables
    profile_data: PatientProfileData


def _load_profile(patient_idx: int) -> dict:
    """Load and return the profile dict from the unified JSONL."""
    filepath = CONSISTENT_CLIENTS_DIR / "profiles.jsonl"
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("idx") == patient_idx:
                return data
    raise KeyError(f"Expected to find 'idx': {patient_idx} in {filepath}")


def _build_latent_variables(profile: dict) -> PatientLatentVariables:
    """Build PatientLatentVariables by normalizing raw scores from the profile."""
    if CONSISTENT_CLIENT_BASELINE:
        # In the SOTA baseline, latent variables aren't normalized, they use the exact values,
        # or we just pass the raw data in the dict and handle the read in client_node.py.
        # But we still gotta build the DTO. We'll pass the raw raw_mr if it exists.
        return PatientLatentVariables(
            anger=float(profile.get("anger_score", 25.0)),
            problem_recognition=float(profile.get("problem_recognition_score", 21.0)),
            motivational_readiness=float(profile.get("motivational_readiness_score", 6.0)),
            self_efficacy=float(profile.get("self_efficacy_score", 3.0)),
            # Actually, the original node just uses float(latent_variables.get("motivational_readiness", 50.0))
        )
    
    # Defaults are roughly population means if missing
    raw_anger = float(profile.get("anger_score", 25.0))
    raw_pr = float(profile.get("problem_recognition_score", 21.0))
    raw_mr = float(profile.get("motivational_readiness_score", 6.0))
    raw_se = float(profile.get("self_efficacy_score", 3.0))

    return PatientLatentVariables(
        anger=_normalize_score(raw_anger, min_val=10.0, max_val=40.0),
        problem_recognition=_normalize_score(raw_pr, min_val=7.0, max_val=35.0),
        motivational_readiness=_normalize_score(raw_mr, min_val=-2.0, max_val=14.0),
        self_efficacy=_normalize_score(raw_se, min_val=1.0, max_val=5.0),
    )


def _build_profile_data(profile: dict) -> PatientProfileData:
    """Build PatientProfileData from the loaded JSONL dict."""
    return PatientProfileData(
        idx=profile["idx"],
        topic=profile["topic"],
        personas=profile.get("Personas", []),
        acceptable_plans=profile.get("Acceptable Plans", []),
        beliefs=profile.get("Beliefs", []),
        motivation=profile.get("Motivation", []),
        behavior=profile.get("Behavior", ""),
        suggestibilities=profile["suggestibilities"],
        initial_state=profile["states"][0]
    )


_patient_profile_dict_cache: dict[int, dict] = {}


def build_patient_profile_dict(patient_idx: int) -> dict:
    """
    Build a flat string-valued patient profile dict for prompt embedding.
    """
    if patient_idx in _patient_profile_dict_cache:
        return _patient_profile_dict_cache[patient_idx]

    result = initialize_patient(patient_idx)
    p = result.profile_data

    profile_dict = {
        "topic": p.topic,
        "personas": "\n".join(f"- {x}" for x in p.personas),
        "acceptable_plans": "\n".join(f"- {x}" for x in p.acceptable_plans),
        "beliefs": "\n".join(f"- {x}" for x in p.beliefs),
        "motivation": "\n".join(f"- {x}" for x in p.motivation),
        "behavior": p.behavior,
    }
    _patient_profile_dict_cache[patient_idx] = profile_dict
    return profile_dict

def initialize_patient(patient_idx: int) -> PatientInitResult:
    """
    Initialize all patient profile DTOs from the unified profiles JSONL.

    Args:
        patient_idx: Numeric patient identifier (idx).

    Returns:
        PatientInitResult dataclass with all typed profile DTOs.
    """
    logger.info(f"Initializing patient profile for patient_idx={patient_idx}")

    profile = _load_profile(patient_idx)

    latent_variables = _build_latent_variables(profile)
    profile_data = _build_profile_data(profile)

    logger.info(
        f"Patient {patient_idx} initialized – "
        f"latent_vars(PR={latent_variables.problem_recognition:.1f}, "
        f"Anger={latent_variables.anger:.1f}, MR={latent_variables.motivational_readiness:.1f}, SE={latent_variables.self_efficacy:.1f})"
    )

    return PatientInitResult(
        latent_variables=latent_variables,
        profile_data=profile_data,
    )
