"""Visualization: Session phase (BEGINNING / MIDDLE / END) distribution per session.

Shows whether the therapist is correctly pacing the MET protocol by tracking
how many therapist turns fall in each structural phase across all sessions.
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

PHASES = ["BEGINNING", "MIDDLE", "END"]
_PHASE_COLORS = {
    "BEGINNING": "#3498db",
    "MIDDLE":    "#2ecc71",
    "END":       "#e67e22",
}


def plot_session_phase_distribution(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot therapist turn distribution across session phases per session.

    Produces a grouped/stacked bar chart with one group per session,
    bars coloured by phase (BEGINNING / MIDDLE / END).

    Args:
        state: Full run state containing ``all_session_turns``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_session_turns: List[List[Dict[str, Any]]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping session phase distribution plot.")
            return

        # Count phase occurrences per session (therapist turns only)
        session_labels: List[str] = []
        session_counters: List[Counter] = []
        for i, turns in enumerate(all_session_turns):
            counts: Counter = Counter()
            for turn in turns:
                if turn.get("speaker") == "therapist":
                    phase = (turn.get("phase") or turn.get("session_phase") or "").strip().upper()
                    if phase in PHASES:
                        counts[phase] += 1
            session_labels.append(f"S{i + 1}")
            session_counters.append(counts)

        # Check there is any data
        if not any(c for c in session_counters):
            logger.info("No phase data found — skipping session phase distribution plot.")
            return

        # Build matrix (sessions × phases)
        n_sessions = len(session_labels)
        matrix = np.zeros((n_sessions, len(PHASES)), dtype=int)
        for i, counter in enumerate(session_counters):
            for j, phase in enumerate(PHASES):
                matrix[i, j] = counter.get(phase, 0)

        fig, ax = plt.subplots(figsize=(max(6, n_sessions * 1.8), 6))

        x = np.arange(n_sessions)
        bar_width = 0.55

        # Compute row-wise percentages (100% stacked)
        row_totals = matrix.sum(axis=1, keepdims=True)
        pct_matrix = np.where(row_totals > 0, matrix / row_totals * 100, 0)

        bottoms = np.zeros(n_sessions)
        for j, phase in enumerate(PHASES):
            heights = pct_matrix[:, j]
            bars = ax.bar(
                x,
                heights,
                width=bar_width,
                bottom=bottoms,
                label=phase,
                color=_PHASE_COLORS[phase],
                edgecolor="white",
                linewidth=0.5,
            )
            # Annotate segment centre with raw count if segment is wide enough
            for k, (raw, pct) in enumerate(zip(matrix[:, j], heights)):
                if pct >= 8:
                    ax.text(
                        x[k],
                        bottoms[k] + pct / 2,
                        str(raw),
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold",
                    )
            bottoms += heights

        ax.set_xticks(x)
        ax.set_xticklabels(session_labels)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Session", fontsize=12)
        ax.set_ylabel("% of Therapist Turns", fontsize=12)
        ax.set_title("Session Phase Distribution — Protocol Pacing (% per Session)", fontsize=14, fontweight="bold")
        ax.legend(title="Phase", fontsize=10, loc="upper right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

        plt.savefig(output_dir / "22_session_phase_distribution.png", bbox_inches="tight")
        plt.close()
        logger.info(f"Session phase distribution plot saved to {output_dir / '22_session_phase_distribution.png'}")
    except Exception as e:
        logger.error(f"Failed to plot session phase distribution: {e}", exc_info=True)
        plt.close()
        raise
