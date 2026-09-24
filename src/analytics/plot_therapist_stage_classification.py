"""Visualization: Therapist's TTM stage-of-change classification per turn per session.

Shows the clinical analyzer's patient stage classification (Precontemplation,
Contemplation, Action) as a continuous trajectory across turns —
one subplot per session, similar to the latent variable trajectory plots.

Expected therapeutic direction:
  - Precontemplation ↓  (patient moves out of denial)
  - Contemplation ↑     (patient weighs change)
  - Action ↑            (patient actively changing behavior)

Layout: one subplot per session (stacked vertically).
X-axis: therapist turn number within session.
Y-axis: ordinal stage index (0–2) with categorical labels.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, get_args

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from src.therapist.therapist_dtos import StageOfChange

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

STAGE_ORDER = list(get_args(StageOfChange))  # Precontemplation, Contemplation, Action

STAGE_COLORS = {
    "Precontemplation": "#e74c3c",
    "Contemplation": "#f39c12",
    "Action": "#2ecc71",
}

LINE_COLOR = "#34495e"


def plot_therapist_stage_classification(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot the therapist's TTM stage-of-change classification per turn per session.

    Each session gets its own subplot showing the stage trajectory as a line
    with colored markers matching the stage.

    Args:
        state: Full run state containing ``all_session_turns``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_session_turns: List[List[Dict[str, Any]]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping therapist stage classification plot.")
            return

        sessions_data = []
        for session_idx, turns in enumerate(all_session_turns):
            turn_numbers: List[int] = []
            stage_indices: List[float] = []
            stage_names: List[str] = []
            for t in turns:
                if t.get("speaker") != "therapist":
                    continue
                raw = (t.get("patient_state_classification") or "").strip()
                if raw not in STAGE_ORDER:
                    continue
                turn_numbers.append(t.get("turn_number", 0))
                stage_indices.append(float(STAGE_ORDER.index(raw)))
                stage_names.append(raw)
            if turn_numbers:
                sessions_data.append((session_idx + 1, turn_numbers, stage_indices, stage_names))

        if not sessions_data:
            logger.info("No stage-of-change data in turn records — skipping plot.")
            return

        n = len(sessions_data)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False, sharex=False)

        for row, (session_num, turn_numbers, stage_indices, stage_names) in enumerate(sessions_data):
            ax = axes[row, 0]
            turns_arr = np.array(turn_numbers, dtype=float)
            idx_arr = np.array(stage_indices, dtype=float)
            point_colors = [STAGE_COLORS[s] for s in stage_names]

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

            ax.set_yticks(range(len(STAGE_ORDER)))
            ax.set_yticklabels(STAGE_ORDER, fontsize=9)
            ax.set_ylim(-0.5, len(STAGE_ORDER) - 0.5)
            ax.set_xlabel("Turn Number", fontsize=10)
            ax.set_title(f"Session {session_num}", fontsize=12, fontweight="bold", pad=6)
            ax.set_xticks(turns_arr)
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        legend_patches = [
            mpatches.Patch(color=STAGE_COLORS[s], label=s)
            for s in STAGE_ORDER
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(STAGE_ORDER),
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.suptitle(
            "Therapist TTM Stage-of-Change Classification: Intra-Session Trajectory",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()

        out_path = output_dir / "35_therapist_stage_classification.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Therapist stage classification plot saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to plot therapist stage classification: {e}", exc_info=True)
        plt.close()
        raise
