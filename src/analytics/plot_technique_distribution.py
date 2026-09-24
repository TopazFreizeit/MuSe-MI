"""Visualization: MET Technique Distribution per session.

Tracks the frequency of each MET technique used by the therapist per
session, derived from the ``technique`` field on therapist TurnRecords.
Rendered as a heatmap (techniques × sessions) so cross-session patterns and
relative usage are immediately legible regardless of how many distinct
technique labels exist.
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="white", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

_NONE_VALUES = {"none", "null", "none.", "n/a", ""}


def plot_technique_distribution(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot MET Tactic Distribution per session as a heatmap.

    Rows are unique tactic labels (as found in the data), columns are
    sessions.  Cell colour encodes count; the raw count is annotated inside
    each cell.  Tactics are sorted by total usage (most frequent on top)
    so the most important rows are immediately visible.

    Args:
        state: Full run state containing ``all_session_turns``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_session_turns: List[List[Dict[str, Any]]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping technique distribution plot.")
            return

        session_labels: List[str] = []
        session_counters: List[Counter] = []

        for i, turns in enumerate(all_session_turns):
            therapist_turns = [t for t in turns if t.get("speaker") == "therapist"]
            if not therapist_turns:
                continue
            counter: Counter = Counter()
            for t in therapist_turns:
                raw = (t.get("technique") or "").strip()
                if raw and raw.lower() not in _NONE_VALUES:
                    counter[raw] += 1
            session_labels.append(f"S{i + 1}")
            session_counters.append(counter)

        if not session_labels:
            logger.info("No technique data — skipping technique distribution plot.")
            return

        # Build DataFrame: rows = techniques, columns = sessions
        all_techs: List[str] = sorted(
            {k for c in session_counters for k in c},
            key=lambda t: -sum(c.get(t, 0) for c in session_counters),
        )
        data = {
            label: [counter.get(tech, 0) for tech in all_techs]
            for label, counter in zip(session_labels, session_counters)
        }
        df = pd.DataFrame(data, index=all_techs)

        n_techs = len(all_techs)
        n_sessions = len(session_labels)
        cell_w, cell_h = 1.1, 0.55
        fig_w = max(6, n_sessions * cell_w + 3)
        fig_h = max(3, n_techs * cell_h + 1.5)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sns.heatmap(
            df,
            ax=ax,
            cmap="Blues",
            linewidths=0.5,
            linecolor="white",
            annot=True,
            fmt="d",
            annot_kws={"size": 9},
            cbar_kws={"label": "Count", "shrink": 0.6},
        )

        ax.set_xlabel("Session", fontsize=12, labelpad=8)
        ax.set_ylabel("MET Tactic", fontsize=12, labelpad=8)
        ax.set_title(
            "Therapist MET Tactic Usage per Session",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.tick_params(axis="x", rotation=0, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

        plt.savefig(output_dir / "23_technique_distribution.png", bbox_inches="tight")
        plt.close()
        logger.info(f"Technique distribution plot saved to {output_dir / '23_technique_distribution.png'}")
    except Exception as e:
        logger.error(f"Failed to plot technique distribution: {e}", exc_info=True)
        plt.close()
        raise
