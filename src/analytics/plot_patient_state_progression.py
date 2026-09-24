"""Visualization: Therapist's TTM Stage-of-Change classification progression through sessions.

Tracks how the clinical analyzer classified the patient's TTM stage
(Precontemplation, Contemplation, Action) at each therapist turn,
plotted as a timeline per session.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, get_args

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from src.therapist.therapist_dtos import StageOfChange

logger = logging.getLogger(__name__)

sns.set_theme(style="white", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

STAGE_ORDER = list(get_args(StageOfChange))  # ["Precontemplation", "Contemplation", "Action"]

STAGE_LABELS = {
    "Precontemplation": "Precontemplation",
    "Contemplation": "Contemplation",
    "Action": "Action",
}

STAGE_COLORS = {
    "Precontemplation": "#e74c3c",   # red
    "Contemplation": "#f39c12",      # amber
    "Action": "#2ecc71",             # green
}


def plot_patient_state_progression(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot the therapist's TTM stage-of-change classification across turns per session.

    Each session gets its own subplot with a step-line showing how the
    clinical analyzer's classification evolved turn by turn.

    Args:
        state: Full run state containing ``all_session_turns``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_session_turns: List[List[Dict[str, Any]]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping stage progression plot.")
            return

        sessions_data: List[Dict[str, List]] = []
        for turns in all_session_turns:
            turn_numbers: List[int] = []
            stage_indices: List[int] = []
            stage_names: List[str] = []
            for t in turns:
                if t.get("speaker") != "therapist":
                    continue
                raw = (t.get("patient_state_classification") or "").strip()
                if raw not in STAGE_ORDER:
                    continue
                turn_numbers.append(t.get("turn_number", 0))
                stage_indices.append(STAGE_ORDER.index(raw))
                stage_names.append(raw)
            if turn_numbers:
                sessions_data.append({
                    "turn_numbers": turn_numbers,
                    "stage_indices": stage_indices,
                    "stage_names": stage_names,
                })

        if not sessions_data:
            logger.info("No stage-of-change classification data — skipping plot.")
            return

        n_sessions = len(sessions_data)
        fig, axes = plt.subplots(
            n_sessions, 1,
            figsize=(10, 3 * n_sessions),
            squeeze=False,
            sharex=False,
        )

        for i, sdata in enumerate(sessions_data):
            ax = axes[i, 0]
            turns = sdata["turn_numbers"]
            indices = sdata["stage_indices"]
            names = sdata["stage_names"]
            colors = [STAGE_COLORS[n] for n in names]

            ax.step(turns, indices, where="mid", color="#34495e", linewidth=1.5, alpha=0.7)
            ax.scatter(turns, indices, c=colors, s=60, zorder=3, edgecolors="white", linewidths=0.5)

            ax.set_yticks(range(len(STAGE_ORDER)))
            ax.set_yticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], fontsize=9)
            ax.set_ylim(-0.5, len(STAGE_ORDER) - 0.5)
            ax.set_xlabel("Turn Number", fontsize=10, labelpad=6)
            ax.set_title(f"Session {i + 1}", fontsize=12, fontweight="bold", pad=8)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        legend_patches = [
            mpatches.Patch(color=STAGE_COLORS[s], label=STAGE_LABELS[s])
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
            "Therapist TTM Stage-of-Change Classification",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()

        plt.savefig(output_dir / "27_patient_state_progression.png", bbox_inches="tight")
        plt.close()
        logger.info(f"Stage progression plot saved to {output_dir / '27_patient_state_progression.png'}")
    except Exception as e:
        logger.error(f"Failed to plot stage progression: {e}", exc_info=True)
        plt.close()
        raise
