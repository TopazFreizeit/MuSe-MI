"""Visualization: Threads Addressed per Session (Metric M5).

Plots the number of change talk threads newly pursued or resolved by the
therapist in each session.  Derived from ``Active_Change_Talk_Threads`` in
therapist session memory snapshots (primary path) with a fallback to the
per-turn ``threads_addressed_this_session`` text field for pre-v3 run files.
High follow-through indicates the therapist is closing open change talk
loops rather than leaving them unresolved.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

_ADDRESSED_STATUSES = {"pursued", "resolved"}


def _load_thread_statuses(mem_json: Any) -> Dict[str, str]:
    """Extract {thread_id: status} dict from a therapist memory snapshot."""
    try:
        mem = json.loads(mem_json) if isinstance(mem_json, str) else mem_json
        threads = mem.get("Active_Change_Talk_Threads", [])
        return {t["id"]: t["status"] for t in threads if "id" in t and "status" in t}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {}


def plot_threads_addressed(state: Dict[str, Any], output_dir: Path) -> None:
    """Plot the number of change talk threads newly addressed per session (M5).

    Primary path: compares consecutive therapist memory snapshots to count
    threads that transitioned from 'open' to 'pursued' or 'resolved'.
    Fallback path (pre-v3 runs): parses the per-turn
    ``threads_addressed_this_session`` free-text field.

    Args:
        state: Full run state containing ``all_therapist_memories`` and/or
               ``all_session_turns``.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_therapist_memories: List[Any] = state.get("all_therapist_memories", [])
        non_empty_memories = [m for m in all_therapist_memories if m]

        if non_empty_memories:
            # Primary path: derive from memory snapshots
            session_labels: List[str] = []
            thread_counts: List[int] = []
            prev_statuses: Dict[str, str] = {}
            for i, mem_json in enumerate(all_therapist_memories):
                if not mem_json:
                    session_labels.append(f"S{i + 1}")
                    thread_counts.append(0)
                    continue
                current_statuses = _load_thread_statuses(mem_json)
                newly_addressed: Set[str] = set()
                for tid, status in current_statuses.items():
                    if status in _ADDRESSED_STATUSES:
                        prev_status = prev_statuses.get(tid)
                        # Count if newly appeared as addressed, or transitioned from open
                        if prev_status is None or prev_status == "open":
                            newly_addressed.add(tid)
                prev_statuses = current_statuses
                session_labels.append(f"S{i + 1}")
                thread_counts.append(len(newly_addressed))
        else:
            # Fallback path: per-turn text field (pre-v3 format)
            _TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")
            _EMPTY_VALUES = {"none", "null", "n/a", "[]", ""}
            all_session_turns: List[List[Dict[str, Any]]] = state.get("all_session_turns", [])
            if not all_session_turns:
                logger.info("No session turns or memories — skipping threads addressed plot.")
                return
            session_labels = []
            thread_counts = []
            for i, turns in enumerate(all_session_turns):
                session_threads: Set[str] = set()
                for turn in turns:
                    if turn.get("speaker") != "therapist":
                        continue
                    raw = (turn.get("threads_addressed_this_session") or "").strip()
                    if raw.lower() in _EMPTY_VALUES:
                        continue
                    session_threads.update(_TOKEN_RE.findall(raw))
                session_labels.append(f"S{i + 1}")
                thread_counts.append(len(session_threads))

        if not any(thread_counts):
            logger.info("No threads addressed data — skipping threads addressed plot.")
            return

        x = np.arange(len(session_labels))
        fig, ax = plt.subplots(figsize=(max(5, len(session_labels) * 1.6), 6))

        bars = ax.bar(
            x, thread_counts,
            color="#8e44ad", edgecolor="white", linewidth=0.5, width=0.55,
        )

        for bar, count in zip(bars, thread_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(session_labels)
        ax.set_xlabel("Session", fontsize=12)
        ax.set_ylabel("Threads Newly Addressed (M5)", fontsize=12)
        ax.set_title("Change Talk Threads Addressed per Session (M5)", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.savefig(output_dir / "24_threads_addressed_m5.png", bbox_inches="tight")
        plt.close()
        logger.info(f"Threads addressed plot saved to {output_dir / '24_threads_addressed_m5.png'}")
    except Exception as e:
        logger.error(f"Failed to plot threads addressed: {e}", exc_info=True)
        plt.close()
        raise
