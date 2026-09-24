"""
This metric tracks the distribution of client language types chosen by the
patient agent across sessions.

Before each patient turn the agent samples a language type — Change Talk,
Sustain Talk, or Neutral — from transition probabilities derived from the
represents the intended motivational valence of the patient's upcoming
utterance, making it a direct measure of how therapist behaviour influences
the patient's moment-to-moment motivational orientation.

LANGUAGE TYPES
    Change Talk  — patient is oriented toward change; corresponds to
                   DARN-CAT utterances (D+, AB+, R+, N+, C+, AC+, TS+, O+)
    Sustain Talk — patient is oriented away from change; corresponds to
                   DARN-CAT sustain-talk utterances (D-, AB-, R-, N-, C-,
                   AC-, TS-, O-)
    Neutral      — patient is neither oriented toward nor away from change;
                   typically social exchange or factual content (N)

DESIRED VALUES
    In a well-functioning MET simulation the proportion of Change Talk turns
    should increase across sessions while Sustain Talk turns decline.
    A flat or deteriorating profile across four sessions suggests the
    therapist behaviour is not shifting the patient's motivational orientation
    in the expected therapeutic direction.
"""
import logging
import sys
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

_CHANGE_TALK = "Change Talk"
_SUSTAIN_TALK = "Sustain Talk"
_NEUTRAL = "Neutral"
_COLORS = {
    _CHANGE_TALK: "#27ae60",
    _SUSTAIN_TALK: "#e74c3c",
    _NEUTRAL: "#95a5a6",
}


def _extract_language_type_data(
    all_session_turns: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for session_idx, turns in enumerate(all_session_turns):
        counts = {_CHANGE_TALK: 0, _SUSTAIN_TALK: 0, _NEUTRAL: 0}
        for turn in turns:
            if turn.get("speaker") != "patient":
                continue
            lang = turn.get("patient_chosen_language_type")
            if lang in counts:
                counts[lang] += 1
        total = sum(counts.values())
        results.append({
            "session": session_idx + 1,
            "change_talk": counts[_CHANGE_TALK],
            "sustain_talk": counts[_SUSTAIN_TALK],
            "neutral": counts[_NEUTRAL],
            "total": total,
        })
    return results


def plot_patient_language_type_distribution(
    state: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot patient chosen language type distribution per session as a stacked bar chart."""
    try:
        all_session_turns: list[list[dict]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping patient language type distribution plot.")
            return

        data = _extract_language_type_data(all_session_turns)
        if not data or all(d["total"] == 0 for d in data):
            logger.info("No patient language type data — skipping plot.")
            return

        sessions = [d["session"] for d in data]
        change = np.array([d["change_talk"] for d in data], dtype=float)
        sustain = np.array([d["sustain_talk"] for d in data], dtype=float)
        neutral = np.array([d["neutral"] for d in data], dtype=float)
        totals = np.array([d["total"] for d in data], dtype=float)

        x = np.arange(len(sessions))
        bar_width = 0.55

        fig, ax = plt.subplots(figsize=(max(8, len(sessions) * 2.2), 6))

        bars_change = ax.bar(x, change, bar_width,
                             label=_CHANGE_TALK, color=_COLORS[_CHANGE_TALK],
                             edgecolor="white", linewidth=0.5)
        bars_sustain = ax.bar(x, sustain, bar_width, bottom=change,
                              label=_SUSTAIN_TALK, color=_COLORS[_SUSTAIN_TALK],
                              edgecolor="white", linewidth=0.5)
        ax.bar(x, neutral, bar_width, bottom=change + sustain,
               label=_NEUTRAL, color=_COLORS[_NEUTRAL],
               edgecolor="white", linewidth=0.5)

        # Inline count labels
        for i in range(len(sessions)):
            segments = [
                (change[i], _COLORS[_CHANGE_TALK], 0.0),
                (sustain[i], _COLORS[_SUSTAIN_TALK], change[i]),
                (neutral[i], _COLORS[_NEUTRAL], change[i] + sustain[i]),
            ]
            for val, color, base in segments:
                if val >= 1:
                    ax.text(x[i], base + val / 2, f"{int(val)}",
                            ha="center", va="center", fontsize=9,
                            fontweight="bold",
                            color="white" if val >= 2 else "#333")
            # Total on top
            if totals[i] > 0:
                ax.text(x[i], totals[i] + 0.3, f"N={int(totals[i])}",
                        ha="center", va="bottom", fontsize=9, color="#333")

        ax.set_xlabel("Session Number", fontsize=12)
        ax.set_ylabel("Patient Turn Count", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Session {s}" for s in sessions])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title("Patient Language Type Distribution per Session",
                     fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)

        ax.text(0.02, 0.98,
                "Sampled from Apodaca et al. (2016) transition probabilities\n",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

        out_path = output_dir / "33_patient_language_type_distribution.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Patient language type distribution plot saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to plot patient language type distribution: {e}", exc_info=True)
        plt.close()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.plot_patient_language_type_distribution <path_to_json>")
        sys.exit(1)

    from src.analytics.analyzer import load_json, create_output_dir

    json_path = sys.argv[1]
    data = load_json(json_path)
    metadata = data.get("metadata", {})
    state = data.get("state", data)
    output_dir = create_output_dir(metadata, json_path)
    plot_patient_language_type_distribution(state, metadata, output_dir)
