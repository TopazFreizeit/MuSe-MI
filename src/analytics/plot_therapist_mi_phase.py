"""Visualization: Therapist's MI Phase (Engaging, Focusing, Evoking, Planning) per turn per session.

Shows the clinical strategist's active MI Phase classification as a continuous trajectory
across turns — one subplot per session.

Expected therapeutic direction generally progresses:
  - Engaging -> Focusing -> Evoking -> Planning

Layout: one subplot per session (stacked vertically).
X-axis: therapist turn number within session.
Y-axis: ordinal phase index (0–3) with categorical labels.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, get_args

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from src.therapist.therapist_dtos import MIPhase

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

PHASE_ORDER = list(get_args(MIPhase))  # Engaging, Focusing, Evoking, Planning

PHASE_COLORS = {
    "Engaging": "#9b59b6",
    "Focusing": "#e67e22",
    "Evoking": "#e74c3c",
    "Planning": "#27ae60",
}

LINE_COLOR = "#2c3e50"


def plot_therapist_mi_phase(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot the therapist's MI Phase classification per turn per session.

    Each session gets its own subplot showing the phase trajectory as a line
    with colored markers matching the phase.

    Args:
        state: Full run state containing ``all_session_turns``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_session_turns: List[List[Dict[str, Any]]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping therapist MI phase plot.")
            return

        sessions_data = []
        for session_idx, turns in enumerate(all_session_turns):
            turn_numbers: List[int] = []
            phase_indices: List[float] = []
            phase_names: List[str] = []
            for t in turns:
                if t.get("speaker") != "therapist":
                    continue
                raw = (t.get("current_mi_phase") or "").strip()
                if raw not in PHASE_ORDER:
                    continue
                turn_numbers.append(t.get("turn_number", 0))
                phase_indices.append(float(PHASE_ORDER.index(raw)))
                phase_names.append(raw)
            if turn_numbers:
                sessions_data.append((session_idx + 1, turn_numbers, phase_indices, phase_names))

        if not sessions_data:
            logger.info("No MI phase data in turn records — skipping plot.")
            return

        n = len(sessions_data)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False, sharex=False)

        for row, (session_num, turn_numbers, phase_indices, phase_names) in enumerate(sessions_data):
            ax = axes[row, 0]
            turns_arr = np.array(turn_numbers, dtype=float)
            idx_arr = np.array(phase_indices, dtype=float)
            point_colors = [PHASE_COLORS[s] for s in phase_names]

            ax.plot(
                turns_arr, idx_arr,
                color=LINE_COLOR,
                linewidth=1.8,
                alpha=0.6,
                zorder=2,
            )
            ax.scatter(
                turns_arr, idx_arr,
                c=point_colors,
                s=70,
                zorder=3,
                edgecolors="white",
                linewidths=0.6,
            )

            ax.set_yticks(range(len(PHASE_ORDER)))
            ax.set_yticklabels(PHASE_ORDER, fontsize=9)
            ax.set_ylim(-0.5, len(PHASE_ORDER) - 0.5)
            ax.set_xlabel("Turn Number", fontsize=10)
            ax.set_title(f"Session {session_num}", fontsize=12, fontweight="bold", pad=6)
            ax.set_xticks(turns_arr)
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        legend_patches = [
            mpatches.Patch(color=PHASE_COLORS[s], label=s)
            for s in PHASE_ORDER
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(PHASE_ORDER),
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.suptitle(
            "Therapist MI Phase: Intra-Session Trajectory",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()

        out_path = output_dir / "36_therapist_mi_phase.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Therapist MI phase plot saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to plot therapist MI phase: {e}", exc_info=True)
        plt.close()
        raise
