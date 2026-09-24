"""Visualization: Technique progression through sessions.

Shows a timeline per session with the specific technique code used at each
therapist turn, displayed as uniformly coloured bars.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="white", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

BAR_COLOR = "#3498db"

_NONE_VALUES = {"none", "null", "none.", "n/a", ""}


def plot_response_mode_technique(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot technique progression across turns per session.

    Each session gets a subplot with a horizontal bar for every therapist
    turn, annotated with the technique code used.

    Args:
        state: Full run state containing ``all_session_turns``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_session_turns: List[List[Dict[str, Any]]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping technique progression plot.")
            return

        # Extract per-session therapist data
        sessions_data: List[Dict[str, List]] = []
        for turns in all_session_turns:
            turn_numbers: List[int] = []
            techniques: List[str] = []
            for t in turns:
                if t.get("speaker") != "therapist":
                    continue
                turn_numbers.append(t.get("turn_number", 0))
                raw_tech = (t.get("technique") or "").strip()
                techniques.append(raw_tech if raw_tech.lower() not in _NONE_VALUES else "")
            if turn_numbers:
                sessions_data.append({
                    "turn_numbers": turn_numbers,
                    "techniques": techniques,
                })

        if not sessions_data:
            logger.info("No technique data — skipping plot.")
            return

        n_sessions = len(sessions_data)
        fig, axes = plt.subplots(
            n_sessions, 1,
            figsize=(max(10, max(len(s["turn_numbers"]) for s in sessions_data) * 0.8 + 2), 2.5 * n_sessions),
            squeeze=False,
            sharex=False,
        )

        for i, sdata in enumerate(sessions_data):
            ax = axes[i, 0]
            turns = sdata["turn_numbers"]
            techniques = sdata["techniques"]
            n = len(turns)

            # Draw uniformly coloured bars
            bars = ax.bar(
                range(n), [1] * n,
                color=BAR_COLOR,
                edgecolor="white",
                linewidth=0.8,
                width=0.9,
            )

            # Annotate technique code inside each bar
            for j, (bar, tech) in enumerate(zip(bars, techniques)):
                if tech:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.5,
                        tech,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="white",
                        rotation=45 if n > 20 else 0,
                    )

            ax.set_xticks(range(n))
            ax.set_xticklabels([str(t) for t in turns], fontsize=11)
            ax.set_yticks([])
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Turn Number", fontsize=13, labelpad=6)
            ax.set_title(f"Session {i + 1}", fontsize=15, fontweight="bold", pad=8)

        fig.suptitle(
            "Technique Progression",
            fontsize=17,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()

        plt.savefig(output_dir / "28_technique_progression.png", bbox_inches="tight")
        plt.close()
        logger.info(f"Technique progression plot saved to {output_dir / '28_technique_progression.png'}")
    except Exception as e:
        logger.error(f"Failed to plot technique progression: {e}", exc_info=True)
        plt.close()
        raise
