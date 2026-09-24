"""
Baseline Trace Persistence for Causal Desynchronization & Permutation Experiment.

Extracts complete episode traces from Pass 1 (intact baseline run) and saves them
to disk as structured JSON files for offline replay in Pass 2 and Pass 3.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def save_causal_baseline_trace(
    final_state: Dict[str, Any],
    patient_id: int,
    output_dir: str = "data/causal_study/baseline_traces",
    store: Optional[Any] = None,
) -> str:
    """
    Save the complete baseline trace for a patient across all sessions.

    Extracts:
      - Initial latent state V_{k,0} for each session
      - Dialogue history H_{k,t} before each patient turn
      - Latest therapist turn u_{k,t}^T
      - Patient utterance u_{k,t}^P
      - Cognitive appraisal package C_{k,t} = (r_{k,t}, I_{k,t})
      - Latent state V_{k,t}
      - Token perplexity and logprob metrics for u_{k,t}^P
      - Memory context and profile data for standalone replay

    Args:
        final_state: MultiSessionState dictionary returned by graph execution.
        patient_id: Numeric patient index.
        output_dir: Directory where trace JSON files are stored.
        store: Optional LangGraph Store to extract memory items.

    Returns:
        String path to the saved trace file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thread_id = final_state.get("thread_id", "")
    profile_data = final_state.get("patient_profile_data", {})
    all_session_turns: List[List[Dict[str, Any]]] = final_state.get("all_session_turns", [])
    all_lv_logs: List[List[Dict[str, Any]]] = final_state.get("all_latent_variable_logs", [])

    # Extract store memory items if store is provided
    memory_context = {}
    if store is not None:
        try:
            from ..utils.store import PATIENT_MEMORY_NS
            for key in ["facts", "agreed_change_plans", "implicit_threads", "motivations", "concerns", "persona_and_stylistic_baseline", "between_session_occurrences"]:
                item = store.get(PATIENT_MEMORY_NS, key)
                if item:
                    memory_context[key] = item.value
        except Exception as e:
            logger.debug(f"Could not extract memory store items: {e}")

    sessions_data = {}
    for s_idx, session_turns in enumerate(all_session_turns):
        session_num = s_idx + 1

        # Determine initial latent state for session
        initial_state = {}
        if s_idx < len(all_lv_logs) and all_lv_logs[s_idx]:
            initial_state = all_lv_logs[s_idx][0]
        elif session_turns:
            # Fallback to patient_latent_variables of first patient turn
            for t in session_turns:
                if t.get("speaker") == "patient" and "patient_latent_variables" in t:
                    initial_state = t["patient_latent_variables"]
                    break

        patient_turn_records = []
        turn_index = 1
        for turn in session_turns:
            if turn.get("speaker") != "patient":
                continue

            # Fallback package if not explicitly attached
            package = turn.get("package", {})
            if (not package or not any(package.values())) and turn.get("session_appraisal_package"):
                package = turn.get("session_appraisal_package")
            elif (not package or not any(package.values())) and session_num == len(all_session_turns):
                last_reasoning = final_state.get("current_session", {}).get("patient_state_reasoning")
                if last_reasoning and isinstance(last_reasoning, dict) and any(last_reasoning.values()):
                    package = last_reasoning

            state_snapshot = turn.get("patient_latent_variables", {})

            record = {
                "turn_index": turn_index,
                "turn_number": turn.get("turn_number", turn_index * 2),
                "history": turn.get("history", ""),
                "therapist_turn": turn.get("therapist_turn", ""),
                "patient_turn": turn.get("volley", ""),
                "package": package,
                "state": state_snapshot,
                "current_dominant_stage": turn.get("current_dominant_stage", ""),
                "token_metrics": turn.get("token_metrics", {}),
                "baseline_system_prompt": turn.get("baseline_system_prompt", ""),
                "bse_occurrence": turn.get("bse_occurrence"),
            }
            patient_turn_records.append(record)
            turn_index += 1

        final_therapist_turn = ""
        if session_turns and session_turns[-1].get("speaker") == "therapist":
            final_therapist_turn = session_turns[-1].get("volley", "")

        sessions_data[str(session_num)] = {
            "session_number": session_num,
            "initial_state": initial_state,
            "turn_count": len(patient_turn_records),
            "final_therapist_turn": final_therapist_turn,
            "turns": patient_turn_records,
        }

    # Aggregate all unique, non-empty appraisal packages across all sessions
    all_appraisal_packages: List[Dict[str, Any]] = []
    for s_info in sessions_data.values():
        if s_info.get("appraisal_package") and s_info["appraisal_package"] not in all_appraisal_packages:
            all_appraisal_packages.append(s_info["appraisal_package"])
        for t_rec in s_info.get("turns", []):
            pkg = t_rec.get("package")
            if pkg and isinstance(pkg, dict) and any(pkg.values()) and pkg not in all_appraisal_packages:
                all_appraisal_packages.append(pkg)

    # Check if final session has patient_state_reasoning
    last_reasoning = final_state.get("current_session", {}).get("patient_state_reasoning")
    if last_reasoning and isinstance(last_reasoning, dict) and any(last_reasoning.values()):
        if last_reasoning not in all_appraisal_packages:
            all_appraisal_packages.append(last_reasoning)

    trace_payload = {
        "metadata": {
            "patient_id": patient_id,
            "thread_id": thread_id,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "num_sessions": len(all_session_turns),
            "model": "meta-llama/llama-3.3-70b-instruct",
        },
        "patient_profile_data": profile_data,
        "memory_context": memory_context,
        "past_event_titles": final_state.get("past_event_titles", []),
        "bse_trigger_classes": final_state.get("bse_trigger_classes", []),
        "all_appraisal_packages": all_appraisal_packages,
        "sessions": sessions_data,
    }

    filepath = output_path / f"patient_{patient_id}_trace.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trace_payload, f, indent=2)

    logger.info(f"Saved causal baseline trace to {filepath}")
    return str(filepath)
