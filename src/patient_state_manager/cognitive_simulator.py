"""
Deterministic Cognitive Simulator.

Step 3 simplification:
- Normalized all latent variables to 1-100 scale.
- Removed bucket-fill readiness transfer; all variables use bounded linear updates.
- Adapted for new variables: anger, problem_recognition, motivational_readiness, self_efficacy.
"""

import os
import logging
from typing import Dict, Any
from .patient_state_manager_dtos import SemanticImpact, MacroShiftSemanticImpact
from ..utils.config import SHIFT_DELTA_MULTIPLIER

logger = logging.getLogger(__name__)

class DeterministicCognitiveSimulator:
    """
    Encapsulates the mathematical rules for updating a patient's latent variables
    from a per-variable semantic-impact classification produced by the upstream LLM.
    """
    ANGER_DELTA = float(os.getenv("ANGER_DELTA", "2.0"))
    SE_DELTA = float(os.getenv("SE_DELTA", "2.0"))
    PR_DELTA = float(os.getenv("PR_DELTA", "2.0"))
    MR_DELTA = float(os.getenv("MR_DELTA", "2.0"))

    @staticmethod
    def _get_semantic_multiplier(impact: SemanticImpact) -> float:
        """3-category rubric: SUCCESS / NEUTRAL / RESISTANCE. Raises on unknown values."""
        if impact == SemanticImpact.SUCCESS: return 1.0
        if impact == SemanticImpact.NEUTRAL: return 0.0
        if impact == SemanticImpact.RESISTANCE: return -1.0
        raise ValueError(f"Unrecognized SemanticImpact value: {impact!r}")

    @staticmethod
    def _get_macro_shift_delta(impact: MacroShiftSemanticImpact) -> float:
        """Converts a qualitative macro-shift semantic impact into a raw float delta. Raises on unknown values."""
        base_delta = 0.0
        if impact == MacroShiftSemanticImpact.MAJOR_PROGRESS: base_delta = 15.0
        elif impact == MacroShiftSemanticImpact.MINOR_PROGRESS: base_delta = 5.0
        elif impact == MacroShiftSemanticImpact.NO_CHANGE: base_delta = 0.0
        elif impact == MacroShiftSemanticImpact.MINOR_REGRESSION: base_delta = -5.0
        elif impact == MacroShiftSemanticImpact.MAJOR_REGRESSION: base_delta = -15.0
        else: raise ValueError(f"Unrecognized MacroShiftSemanticImpact value: {impact!r}")
        return base_delta * SHIFT_DELTA_MULTIPLIER

    @staticmethod
    def _bounded_update(current_val: float, delta: float) -> float:
        """
        Simple linear update constrained to [1.0, 100.0].
        Replaces the complex asymptotic leaky integrator.
        """
        return max(1.0, min(100.0, current_val + delta))

    @classmethod
    def apply_math(
        cls,
        prior_state: Dict[str, Any],
        dto: Any,
    ) -> Dict[str, Any]:
        """
        Calculates new latent state from per-variable semantic_impact.
        """
        # Per-variable deltas: pure function of semantic_impact.
        # Anger goes DOWN on success, UP on resistance (sign inverted).
        base_anger_delta = -cls._get_semantic_multiplier(dto.anger.semantic_impact) * cls.ANGER_DELTA * SHIFT_DELTA_MULTIPLIER
        new_anger = cls._bounded_update(float(prior_state["anger"]), base_anger_delta)

        base_se_delta = cls._get_semantic_multiplier(dto.self_efficacy.semantic_impact) * cls.SE_DELTA * SHIFT_DELTA_MULTIPLIER
        base_pr_delta = cls._get_semantic_multiplier(dto.problem_recognition.semantic_impact) * cls.PR_DELTA * SHIFT_DELTA_MULTIPLIER
        base_mr_delta = cls._get_semantic_multiplier(dto.motivational_readiness.semantic_impact) * cls.MR_DELTA * SHIFT_DELTA_MULTIPLIER

        new_se = cls._bounded_update(float(prior_state["self_efficacy"]), base_se_delta)
        new_pr = cls._bounded_update(float(prior_state["problem_recognition"]), base_pr_delta)
        new_mr = cls._bounded_update(float(prior_state["motivational_readiness"]), base_mr_delta)

        return {
            "anger": float(round(new_anger, 2)),
            "self_efficacy": float(round(new_se, 2)),
            "problem_recognition": float(round(new_pr, 2)),
            "motivational_readiness": float(round(new_mr, 2))
        }
