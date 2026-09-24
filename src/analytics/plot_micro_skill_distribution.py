"""Visualization: Micro-Skill Anchor Distribution per session.

Tracks the frequency of each micro-skill (FA, FI, EC, ST) across therapist
turns per session, derived from the ``micro_skill_anchor`` field on therapist
TurnRecords.  High EC usage may signal the therapist leaning on autonomy
support; ST marks structural transitions.
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

_NONE_VALUES = {"none", "null", "none.", "n/a", ""}
_SKILL_ORDER = ["FA", "FI", "EC", "ST"]
_SKILL_COLORS = {
    "FA": "#16a085",
    "FI": "#2980b9",
    "EC": "#c0392b",
    "ST": "#f39c12",
}
_SKILL_LABELS = {
    "FA": "FA (Facilitate)",
    "FI": "FI (Filler)",
    "EC": "EC (Emphasize Control)",
    "ST": "ST (Structure)",
}


def plot_micro_skill_distribution(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot Micro-Skill Anchor Distribution per session.

    Produces a grouped bar chart (one group per session) showing how many
    times each micro-skill was anchored by the therapist, derived from the
    ``micro_skill_anchor`` field in TurnRecord.  Turns reporting "None" are
    excluded from counts.

    Args:
        state: Full run state containing ``all_session_turns``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_session_turns: List[List[Dict[str, Any]]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping micro-skill distribution plot.")
            return

        session_labels: List[str] = []
        session_counters: List[Counter] = []

        for i, turns in enumerate(all_session_turns):
            therapist_turns = [t for t in turns if t.get("speaker") == "therapist"]
            if not therapist_turns:
                continue
            counter: Counter = Counter()
            for t in therapist_turns:
                raw = (t.get("micro_skill_anchor") or "").strip().upper()
                if not raw or raw in _NONE_VALUES:
                    continue
                # The field may contain e.g. "EC — Let's take a moment…"
                # Extract the code: first word before space or dash
                code = raw.split()[0].rstrip("(-)").strip()
                if code in _SKILL_ORDER:
                    counter[code] += 1
                # Silently skip unrecognised values
            session_labels.append(f"S{i + 1}")
            session_counters.append(counter)

        if not session_labels or all(sum(c.values()) == 0 for c in session_counters):
            logger.info("No micro_skill_anchor data — skipping micro-skill distribution plot.")
            return

        n_sessions = len(session_labels)
        x = np.arange(n_sessions)
        bar_width = 0.18
        fig, ax = plt.subplots(figsize=(max(6, n_sessions * 2.5), 6))

        for j, skill in enumerate(_SKILL_ORDER):
            counts = [session_counters[i].get(skill, 0) for i in range(n_sessions)]
            offset = (j - len(_SKILL_ORDER) / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, counts,
                width=bar_width,
                label=_SKILL_LABELS[skill],
                color=_SKILL_COLORS[skill],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        str(count),
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(session_labels)
        ax.set_xlabel("Session", fontsize=12)
        ax.set_ylabel("Number of Therapist Turns", fontsize=11)
        ax.set_title("Micro-Skill Anchor Distribution per Session", fontsize=14, fontweight="bold")
        ax.legend(title="Micro-Skill", fontsize=9, title_fontsize=10)

        out_path = output_dir / "29_micro_skill_distribution.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Micro-skill distribution plot saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to plot micro-skill distribution: {e}", exc_info=True)
        plt.close()
        raise
