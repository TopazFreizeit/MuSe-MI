"""
This metric tracks the distribution of DARN-CAT planner codes selected by the
patient agent across sessions.

Before each non-Neutral patient turn the DARN-CAT cognitive planner selects
the single most appropriate motivational code given the therapist's last
utterance, the patient's latent psychological variables (readiness,
self-efficacy, perceived discrepancy), and the sampled language type.

The selected code is one of the DARN-CAT taxonomy codes prefixed by polarity:
    Change Talk codes  — Change-Desire, Change-Ability, Change-Reason,
                         Change-Need, Change-Commitment, Change-Activation,
                         Change-TakingSteps
    Sustain Talk codes — Sustain-Desire, Sustain-Ability, Sustain-Reason,
                         Sustain-Need, Sustain-Commitment, Sustain-Activation,
                         Sustain-TakingSteps (and similar)
    Neutral            — no DARN-CAT planner run (language type was Neutral)

DESIRED VALUES
    A well-functioning simulation should show a shift from DARN-type Change
    Talk codes (Desire, Ability, Reason, Need — preparatory) toward CAT-type
    codes (Commitment, Activation, TakingSteps — mobilizing) across sessions,
    consistent with the Transtheoretical Model arc embedded in MET.
"""
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# Colour palette: Change codes get green shades, Sustain codes get red shades,
# Neutral gets grey.
_CHANGE_BASE = "#27ae60"
_SUSTAIN_BASE = "#e74c3c"
_NEUTRAL_COLOR = "#95a5a6"

# Ordered code lists used to produce a deterministic stacking order.
_CHANGE_CODES_ORDERED = [
    "Change-Desire", "Change-Ability", "Change-Reason", "Change-Need",
    "Change-Commitment", "Change-Activation", "Change-TakingSteps",
]
_SUSTAIN_CODES_ORDERED = [
    "Sustain-Desire", "Sustain-Ability", "Sustain-Reason", "Sustain-Need",
    "Sustain-Commitment", "Sustain-Activation", "Sustain-TakingSteps",
]

# Green shades for Change codes (light → dark across DARN → CAT)
_CHANGE_COLORS = ["#a9dfbf", "#7dcea0", "#52be80", "#27ae60",
                  "#1e8449", "#196f3d", "#145a32"]
# Red shades for Sustain codes
_SUSTAIN_COLORS = ["#f5b7b1", "#f1948a", "#ec7063", "#e74c3c",
                   "#cb4335", "#b03a2e", "#922b21"]


def _build_color_map() -> Dict[str, str]:
    color_map: dict[str, str] = {"Neutral": _NEUTRAL_COLOR}
    for code, color in zip(_CHANGE_CODES_ORDERED, _CHANGE_COLORS):
        color_map[code] = color
    for code, color in zip(_SUSTAIN_CODES_ORDERED, _SUSTAIN_COLORS):
        color_map[code] = color
    return color_map


_COLOR_MAP = _build_color_map()


def _extract_darn_cat_code_data(
    all_session_turns: List[List[Dict[str, Any]]],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Return per-session code counts and a sorted list of all observed codes."""
    all_observed: set[str] = set()
    session_counters: list[Counter] = []

    for turns in all_session_turns:
        counter: Counter = Counter()
        for turn in turns:
            if turn.get("speaker") != "patient":
                continue
            code = turn.get("patient_chosen_darn_cat_code")
            if code:
                counter[code] += 1
                all_observed.add(code)
        session_counters.append(counter)

    # Sort: Change codes in order, then Sustain codes in order, then Neutral,
    # then anything unexpected alphabetically.
    def _sort_key(c: str) -> tuple:
        if c in _CHANGE_CODES_ORDERED:
            return (0, _CHANGE_CODES_ORDERED.index(c))
        if c in _SUSTAIN_CODES_ORDERED:
            return (1, _SUSTAIN_CODES_ORDERED.index(c))
        if c == "Neutral":
            return (2, 0)
        return (3, c)

    sorted_codes = sorted(all_observed, key=_sort_key)

    results: list[dict[str, Any]] = []
    for session_idx, counter in enumerate(session_counters):
        row: dict[str, Any] = {"session": session_idx + 1}
        for code in sorted_codes:
            row[code] = counter.get(code, 0)
        row["total"] = sum(counter.values())
        results.append(row)

    return results, sorted_codes


def plot_patient_darn_cat_code_distribution(
    state: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot patient DARN-CAT planner code distribution per session as a stacked bar chart."""
    try:
        all_session_turns: list[list[dict]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping patient DARN-CAT code distribution plot.")
            return

        data, codes = _extract_darn_cat_code_data(all_session_turns)
        if not data or not codes or all(d["total"] == 0 for d in data):
            logger.info("No patient_chosen_darn_cat_code data — skipping plot.")
            return

        sessions = [d["session"] for d in data]
        totals = np.array([d["total"] for d in data], dtype=float)
        x = np.arange(len(sessions))
        bar_width = 0.55

        fig, ax = plt.subplots(figsize=(max(8, len(sessions) * 2.5), 7))

        bottoms = np.zeros(len(sessions), dtype=float)
        for code in codes:
            vals = np.array([d[code] for d in data], dtype=float)
            color = _COLOR_MAP.get(code, "#aaaaaa")
            ax.bar(x, vals, bar_width, bottom=bottoms,
                   label=code, color=color,
                   edgecolor="white", linewidth=0.5)
            # Label segments with count if large enough
            for i, val in enumerate(vals):
                if val >= 1:
                    ax.text(x[i], bottoms[i] + val / 2, f"{int(val)}",
                            ha="center", va="center", fontsize=7,
                            fontweight="bold", color="white" if val >= 2 else "#333")
            bottoms += vals

        # N= total on top of each bar
        for i, total in enumerate(totals):
            if total > 0:
                ax.text(x[i], total + 0.3, f"N={int(total)}",
                        ha="center", va="bottom", fontsize=9, color="#333")

        ax.set_xlabel("Session Number", fontsize=12)
        ax.set_ylabel("Patient Turn Count", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Session {s}" for s in sessions])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title("Patient DARN-CAT Planner Code Distribution per Session",
                     fontsize=14, fontweight="bold")

        # Legend outside the plot on the right
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, fontsize=9, title="DARN-CAT Code")

        ax.text(0.02, 0.98,
                "DARN = Desire/Ability/Reason/Need (Preparatory CT)\n"
                "CAT = Commitment/Activation/TakingSteps (Mobilizing CT)",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

        out_path = output_dir / "34_patient_darn_cat_code_distribution.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Patient DARN-CAT code distribution plot saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to plot patient DARN-CAT code distribution: {e}", exc_info=True)
        plt.close()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.plot_patient_darn_cat_code_distribution <path_to_json>")
        sys.exit(1)

    from src.analytics.analyzer import load_json, create_output_dir

    json_path = sys.argv[1]
    data = load_json(json_path)
    metadata = data.get("metadata", {})
    state = data.get("state", data)
    output_dir = create_output_dir(metadata, json_path)
    plot_patient_darn_cat_code_distribution(state, metadata, output_dir)
