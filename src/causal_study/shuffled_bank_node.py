r"""
LangGraph Node for Closed-Loop Trajectory Rollout (Pass 3).

In Pass 3, the Cognitive Interpreter is replaced by this node. Instead of calculating
new appraisal directives based on the therapist's latest turn, this node injects the
permuted directive package C_{\pi(t)} from the pre-shuffled bank and updates the patient's
latent state cumulatively using the permuted discrete multipliers.
"""
import logging
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from .metrics import apply_package_math
from .shuffled_bank import get_shuffled_package_for_turn
from ..graphs.graph_config import LatentTrajectoryEntry
from ..utils.config import CAUSAL_ROLLOUT_SEED, CAUSAL_BASELINE_DIR
from ..utils.tracing import enrich_span, traced_span
from ..patient.patient_config import PATIENT_IDX

logger = logging.getLogger(__name__)


@traced_span("causal_shuffled_bank")
def causal_shuffled_bank_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    r"""
    Inject permuted cognitive appraisal package C_{\pi(t)} and update latent state.

    Args:
        state: ConversationState dict.
        config: RunnableConfig with configurable items.

    Returns:
        State update dict with patient_latent_variables, latent_variable_logs, and patient_state_reasoning.
    """
    try:
        configurable = config.get("configurable", {})
        session_number = state.get("session_number", configurable.get("current_session_number", 1))

        # Resolve patient index
        patient_profile_data = configurable.get("patient_profile_data", {})
        if isinstance(patient_profile_data, dict) and "idx" in patient_profile_data:
            patient_id = int(patient_profile_data["idx"])
        else:
            patient_id = PATIENT_IDX

        # Count completed patient turns in this session to index the shuffled bank
        messages = state.get("messages", [])
        patient_turn_index = sum(1 for m in messages if getattr(m, "name", "") == "Patient")

        prior_lv = state["patient_latent_variables"]
        prior_lv_dict = prior_lv.model_dump() if hasattr(prior_lv, "model_dump") else prior_lv.copy()

        # Retrieve permuted package C_{\pi(t)}
        package = get_shuffled_package_for_turn(
            patient_id=patient_id,
            session_number=session_number,
            patient_turn_index=patient_turn_index,
            seed=CAUSAL_ROLLOUT_SEED,
            baseline_dir=CAUSAL_BASELINE_DIR,
        )

        # Apply cumulative mathematical transition along permuted trajectory
        updated_lv = apply_package_math(prior_lv_dict, package)

        logger.info(
            f"[Causal Shuffled Bank | P{patient_id} S{session_number} Turn {patient_turn_index}] "
            f"Prior LV: {prior_lv_dict} -> Updated LV: {updated_lv}"
        )

        # Enrich Phoenix trace span with injected impacts
        enrich_span(
            anger_impact=package.get("anger", {}).get("impact"),
            se_impact=package.get("self_efficacy", {}).get("impact"),
            pr_impact=package.get("problem_recognition", {}).get("impact"),
            mr_impact=package.get("motivational_readiness", {}).get("impact"),
        )

        # Append latent score snapshot to logs
        turn_number = state.get("turn_count", 0)
        snapshot = {
            "turn_number": turn_number,
            "anger": updated_lv["anger"],
            "self_efficacy": updated_lv["self_efficacy"],
            "problem_recognition": updated_lv["problem_recognition"],
            "motivational_readiness": updated_lv["motivational_readiness"],
        }
        snapshots = list(state.get("latent_variable_logs", []))
        snapshots.append(snapshot)

        trajectory_entry: LatentTrajectoryEntry = {
            "step_index": -1,
            "component": "cognitive_interpreter",
            "session_number": session_number,
            "turn_number": turn_number,
            "anger": float(updated_lv["anger"]),
            "self_efficacy": float(updated_lv["self_efficacy"]),
            "problem_recognition": float(updated_lv["problem_recognition"]),
            "motivational_readiness": float(updated_lv["motivational_readiness"]),
            "details": {
                "causal_shuffled": True,
                "impacts": {
                    "anger": package.get("anger", {}).get("impact"),
                    "self_efficacy": package.get("self_efficacy", {}).get("impact"),
                    "problem_recognition": package.get("problem_recognition", {}).get("impact"),
                    "motivational_readiness": package.get("motivational_readiness", {}).get("impact"),
                }
            },
        }
        session_trajectory = list(state.get("session_latent_trajectory", []))
        session_trajectory.append(trajectory_entry)

        return {
            "patient_latent_variables": updated_lv,
            "latent_variable_logs": snapshots,
            "session_latent_trajectory": session_trajectory,
            "patient_state_reasoning": package,
        }

    except Exception as e:
        logger.error(f"Failed in causal_shuffled_bank_node: {e}", exc_info=True)
        raise
