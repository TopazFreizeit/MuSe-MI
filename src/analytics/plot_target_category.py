"""Visualization: Target category intra-session progress across sessions.

Layout: one subplot per session (stacked vertically), 1 line per subplot.
X-axis: turn number within session.
Y-axis: Target category.
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


def plot_target_category(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot target category trajectories within each therapy session.

    Args:
        state: Full run state containing ``all_session_turns``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_session_turns: List[List[Dict[str, Any]]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping target category plot.")
            return

        sessions_data = []
        for session_idx, turns in enumerate(all_session_turns):
            if not turns:
                continue
            
            p_turns = [t for t in turns if t.get("speaker") == "patient"]
            turn_nums = [t.get("turn_number", i+1) for i, t in enumerate(p_turns)]
            
            categories = []
            for t in p_turns:
                cat = str(t.get("target_category", "")).strip().lower()
                if not cat or cat == "none":
                    categories.append("unknown")
                elif "change_talk" in cat or "change" in cat:
                    categories.append("change_talk")
                elif "sustain_talk" in cat or "sustain" in cat:
                    categories.append("sustain_talk")
                elif "neutral_talk" in cat or "neutral" in cat:
                    categories.append("neutral_talk")
                else:
                    categories.append("unknown")
                    
            if all(c == "unknown" for c in categories):
                continue
                
            sessions_data.append((session_idx + 1, turn_nums, categories))

        if not sessions_data:
            logger.info("Target category values all unknown/empty — skipping plot.")
            return

        n = len(sessions_data)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False, sharex=False)

        cat_order = ["sustain_talk", "neutral_talk", "change_talk", "unknown"]
        y_mapping = {cat: i for i, cat in enumerate(cat_order)}
        display_labels = ["Sustain Talk", "Neutral Talk", "Change Talk", "Unknown"]

        for row, (session_num, turns, categories) in enumerate(sessions_data):
            ax = axes[row, 0]
            turns_arr = np.array(turns, dtype=float)
            y_vals = [y_mapping[c] for c in categories]

            ax.plot(
                turns_arr, y_vals,
                color="#0288D1",
                linewidth=2.0,
                marker="o",
                markersize=6,
                zorder=3,
            )

            ax.set_ylim(-0.5, 3.5)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(display_labels)
            
            ax.set_ylabel("Target Category", fontsize=10)
            ax.set_xlabel("Turn Number", fontsize=10)
            ax.set_title(f"Session {session_num}", fontsize=12, fontweight="bold", pad=6)
            ax.set_xticks(turns_arr)
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=9)

        fig.suptitle(
            "Patient Target Category: Intra-Session Progression",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()

        out_path = output_dir / "38_target_category.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Target category plot saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to plot target category: {e}", exc_info=True)
        plt.close()
        raise