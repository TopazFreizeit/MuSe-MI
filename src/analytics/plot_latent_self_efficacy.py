"""Visualization: Self-Efficacy intra-session progress across therapy sessions.

Plots the patient's self-efficacy score (AASE, 0–100) at each latent-variable
update checkpoint within every session.  Entry 0 in each session's snapshot
list is the pre-session baseline (turn 0); subsequent entries are recorded every
X_TURN_UPDATE turns by the perceived_discrepancy_manager_agent (last node in the
latent-variable chain).

Layout: one subplot per session (stacked vertically).
X-axis: turn number within session.
Y-axis: Self-Efficacy score (0–100).
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

_COLOR = "#1565C0"          # deep blue
_COLOR_LIGHT = "#90CAF9"    # light blue for fill
_MARKER = "o"
_LINEWIDTH = 2.0
_MARKERSIZE = 7


def plot_latent_self_efficacy(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot self-efficacy score trajectory within each therapy session.

    Args:
        state: Full run state containing ``all_latent_variable_logs``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_snapshots: List[List[Dict[str, Any]]] = state.get("all_latent_variable_logs", [])
        if not all_snapshots:
            logger.info("No latent variable snapshots — skipping self-efficacy plot.")
            return

        # Filter out sessions with no meaningful data (only the turn-0 placeholder)
        sessions_data = []
        for session_idx, snapshots in enumerate(all_snapshots):
            if not snapshots:
                continue
            turns = [s["turn_number"] for s in snapshots]
            values = [s.get("self_efficacy", np.nan) for s in snapshots]
            if all(np.isnan(v) for v in values):
                continue
            sessions_data.append((session_idx + 1, turns, values))

        if not sessions_data:
            logger.info("Self-efficacy values all NaN — skipping plot.")
            return

        n = len(sessions_data)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), squeeze=False, sharex=False)

        for row, (session_num, turns, values) in enumerate(sessions_data):
            ax = axes[row, 0]
            turns_arr = np.array(turns, dtype=float)
            vals_arr = np.array(values, dtype=float)

            # Shaded area under the line
            ax.fill_between(turns_arr, vals_arr, alpha=0.12, color=_COLOR_LIGHT)

            # Main line + markers
            ax.plot(turns_arr, vals_arr, color=_COLOR, linewidth=_LINEWIDTH,
                    marker=_MARKER, markersize=_MARKERSIZE, zorder=3)

            # Annotate each data point with its value
            for x, y in zip(turns_arr, vals_arr):
                if not np.isnan(y):
                    ax.annotate(
                        f"{y:.0f}",
                        xy=(x, y),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=_COLOR,
                    )

            # Reference lines
            ax.axhline(50, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

            ax.set_ylim(0, 105)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
            ax.set_ylabel("Self-Efficacy (0–100)", fontsize=10)
            ax.set_xlabel("Turn Number", fontsize=10)
            ax.set_title(f"Session {session_num}", fontsize=12, fontweight="bold", pad=6)
            ax.set_xticks(turns_arr)
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=9)

        fig.suptitle(
            "Self-Efficacy: Intra-Session Progression",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()

        out_path = output_dir / "29_latent_self_efficacy.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Self-efficacy plot saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to plot self-efficacy: {e}", exc_info=True)
        plt.close()
        raise
