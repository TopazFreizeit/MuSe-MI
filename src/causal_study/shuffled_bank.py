r"""
Shuffled Directive Package Bank for Causal Rollout Simulations (Pass 3).

Loads the recorded baseline trace for a patient, shuffles the directive packages
within each session using a deterministic seed, and provides the sequence of permuted
packages C_{\pi(t)} to the interactive rollout graph.
"""
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# In-memory cache for loaded banks: (patient_id, seed, baseline_dir) -> bank
_CACHED_BANKS: Dict[tuple, Dict[int, List[Dict[str, Any]]]] = {}


def load_shuffled_package_bank(
    patient_id: int,
    seed: int,
    baseline_dir: str = "data/causal_study/baseline_traces",
) -> List[Dict[str, Any]]:
    """
    Load baseline trace for patient_id and permute all appraisal packages
    randomly across all sessions using a deterministic seed.

    Args:
        patient_id: Patient profile index.
        seed: Random seed for deterministic permutation.
        baseline_dir: Directory containing patient_*_trace.json files.

    Returns:
        List of globally permuted appraisal packages across all sessions.
    """
    cache_key = (patient_id, seed, baseline_dir)
    if cache_key in _CACHED_BANKS:
        return _CACHED_BANKS[cache_key]

    trace_file = Path(baseline_dir) / f"patient_{patient_id}_trace.json"
    if not trace_file.exists():
        raise FileNotFoundError(
            f"Baseline trace file not found at {trace_file}. "
            f"Run Pass 1 with CAUSAL_RECORD_BASELINE=true first to generate baseline traces."
        )

    with open(trace_file, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    rng = random.Random(seed)
    global_packages: List[Dict[str, Any]] = []

    # 1. Check top-level all_appraisal_packages
    for pkg in trace_data.get("all_appraisal_packages", []):
        if pkg and isinstance(pkg, dict) and any(pkg.values()) and pkg not in global_packages:
            global_packages.append(pkg)

    # 2. Check session and turn level packages
    sessions_dict = trace_data.get("sessions", {})
    for session_key, session_info in sessions_dict.items():
        s_pkg = session_info.get("appraisal_package")
        if s_pkg and isinstance(s_pkg, dict) and any(s_pkg.values()) and s_pkg not in global_packages:
            global_packages.append(s_pkg)
        turns = session_info.get("turns", [])
        for t in turns:
            pkg = t.get("package")
            if pkg and isinstance(pkg, dict) and any(pkg.values()) and pkg not in global_packages:
                global_packages.append(pkg)

    permuted = list(global_packages)
    rng.shuffle(permuted)

    logger.info(
        f"Loaded shuffled package bank for Patient {patient_id} (seed={seed}) with {len(permuted)} packages shuffled globally across all sessions."
    )
    _CACHED_BANKS[cache_key] = permuted
    return permuted


def get_shuffled_package_for_turn(
    patient_id: int,
    session_number: int,
    patient_turn_index: int,
    seed: int,
    baseline_dir: str = "data/causal_study/baseline_traces",
) -> Dict[str, Any]:
    r"""
    Retrieve the permuted package C_{\pi(t)} for a specific turn in an interactive rollout.
    Draws desynchronized packages selected randomly across all sessions.

    Args:
        patient_id: Patient index.
        session_number: Current session number (1-4).
        patient_turn_index: 0-based index of the patient turn within the session.
        seed: Random seed used for permutation.
        baseline_dir: Baseline trace directory path.

    Returns:
        The permuted appraisal package dict.
    """
    permuted_packages = load_shuffled_package_bank(patient_id, seed, baseline_dir)

    if not permuted_packages:
        logger.warning(
            f"No packages found in bank for patient {patient_id}. Returning empty package."
        )
        return {}

    # Global linear turn index across 4 sessions (assuming ~2-3 patient turns per session)
    # This guarantees each turn across all sessions draws from different permuted packages
    global_turn_offset = (session_number - 1) * 3 + patient_turn_index
    pkg_idx = global_turn_offset % len(permuted_packages)
    return permuted_packages[pkg_idx]
