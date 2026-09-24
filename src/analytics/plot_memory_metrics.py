"""
This graph measures how well the therapist's memory system is working across
three session transitions — from Session 1 to 2, Session 2 to 3, and Session
3 to 4. The dotted green line at the top marks the ideal score of 100%,
meaning perfect performance.

Each group of bars represents one transition between sessions, and the three
colors represent three different measurements.

GREEN BAR — M1: Biographical Fact Retention Rate (BFRR)
    Measures whether the therapist remembered all the basic facts about the
    patient across sessions — things like their job, their family, and their
    drinking habits. In real therapy, a therapist must never forget a patient's
    personal background, because losing that information breaks the continuity
    of care. This bar should always be 100%.

    Clinical basis:
        The MITI-4 coding manual (Moyers et al., 2016, Journal of Substance
        Abuse Treatment) requires that clinicians demonstrate full knowledge
        of the patient's background as a prerequisite for developing
        discrepancy — one of the four core MET principles. Without accurate
        background retention, the therapist cannot connect the patient's
        behavior to their values.

BLUE BAR — M2: Change Talk Thread Persistence Rate (CTPR)
    Measures whether every topic the patient raised in one session was still
    tracked and recorded in the next — none of them silently dropped or
    forgotten. Clinically, every concern a patient raises is potentially
    important change talk, so nothing should disappear from the record.
    This bar should also always be 100%.

    Clinical basis:
        Miller & Rollnick (2013, "Motivational Interviewing: Helping People
        Change", 3rd ed., Guilford Press) describe the therapist's role as
        actively collecting and holding all change talk expressed by the
        client across sessions. Dropping a thread is equivalent to ignoring
        a client's expressed motivation — a direct violation of the evoking
        process in MI.

ORANGE BAR — M3: Thread Follow-up Rate (TFR)
    Measures whether the therapist actually came back to topics the patient
    had opened but not yet resolved. For example: if a patient said "maybe I
    could try talking to a friend," did the therapist return to that idea in
    the very next session? This is the hardest bar to achieve because it
    requires not just storing information, but actively using it to guide
    the next session's agenda. The clinical minimum for this bar is 50%,
    meaning at least half of all open topics should be revisited.

    Clinical basis:
        The SAMHSA Treatment Improvement Protocol (Chapter 3, "Motivational
        Interviewing as a Counseling Style") states that the therapist must
        "reinforce change talk by reflecting it back verbally" and
        "encourage the client to continue exploring the possibility of change
        by asking for elaboration." This reinforcement cannot occur if the
        therapist does not return to previously opened threads.

        Futher, Hardcastle et al. (2017, "Using Motivational Interviewing
        and Brief Action Planning", ScienceDirect) describe the therapeutic
        summary as a "bouquet" that selectively collects and re-presents
        threads of change talk from previous interactions — making
        cross-session thread follow-up a defining technical skill of MET,
        not an optional practice.

        The 50% minimum threshold is derived from MITI-4 proficiency
        standards (Moyers et al., 2016), which define basic competency as
        consistent — not occasional — pursuit of client-generated change
        talk. A rate below 50% indicates the therapist is systematically
        ignoring the majority of the patient's own expressed motivations,
        which contradicts the foundational mechanism of MET as validated in
        Project MATCH (Allen et al., 1997, Journal of Studies on Alcohol).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def _parse_memories(all_therapist_memories: List[str]) -> List[Optional[Dict[str, Any]]]:
    """Parse each therapist memory JSON string into a dict. Returns None for unparseable entries."""
    parsed: list[Optional[dict]] = []
    for i, mem_str in enumerate(all_therapist_memories):
        if not mem_str:
            parsed.append(None)
            continue
        try:
            parsed.append(json.loads(mem_str))
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Could not parse therapist memory for session {i + 1}")
            parsed.append(None)
    return parsed


def _compute_bfrr(mem_n: Dict, mem_n1: Dict) -> Optional[float]:
    """M1: Biographical Fact Retention Rate between two consecutive memories."""
    facts_n = mem_n.get("Static_Patient_Background", [])
    facts_n1 = mem_n1.get("Static_Patient_Background", [])
    if not facts_n:
        return None  # nothing to retain
    retained = sum(1 for fact in facts_n if fact in facts_n1)
    return retained / len(facts_n) * 100


def _compute_ctpr(mem_n: Dict, mem_n1: Dict) -> Optional[float]:
    """M2: Change Talk Thread Persistence Rate between two consecutive memories."""
    threads_n = mem_n.get("Active_Change_Talk_Threads")
    threads_n1 = mem_n1.get("Active_Change_Talk_Threads")
    if threads_n is None or threads_n1 is None:
        return None  # field missing — old schema
    if not threads_n:
        return None  # no threads to track
    ids_n1 = {t["id"] for t in threads_n1 if isinstance(t, dict) and "id" in t}
    persisted = sum(1 for t in threads_n if isinstance(t, dict) and t.get("id") in ids_n1)
    return persisted / len(threads_n) * 100


def _compute_tfr(mem_n: Dict, mem_n1: Dict) -> Optional[float]:
    """M3: Thread Follow-up Rate — open threads in N that were pursued/resolved in N+1."""
    threads_n = mem_n.get("Active_Change_Talk_Threads")
    threads_n1 = mem_n1.get("Active_Change_Talk_Threads")
    if threads_n is None or threads_n1 is None:
        return None  # field missing — old schema
    open_threads_n = [t for t in threads_n if isinstance(t, dict) and t.get("status") == "open"]
    if not open_threads_n:
        return None  # no open threads to follow up
    status_map_n1 = {
        t["id"]: t.get("status")
        for t in threads_n1 if isinstance(t, dict) and "id" in t
    }
    followed_up = sum(
        1 for t in open_threads_n
        if status_map_n1.get(t["id"]) in {"pursued", "resolved"}
    )
    return followed_up / len(open_threads_n) * 100


def _compute_all_metrics(
    memories: List[Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Compute M1/M2/M3 for each consecutive session pair."""
    results: list[dict[str, Any]] = []
    for i in range(len(memories) - 1):
        mem_n = memories[i]
        mem_n1 = memories[i + 1]
        if mem_n is None or mem_n1 is None:
            continue
        results.append({
            "transition": f"S{i + 1}→S{i + 2}",
            "bfrr": _compute_bfrr(mem_n, mem_n1),
            "ctpr": _compute_ctpr(mem_n, mem_n1),
            "tfr": _compute_tfr(mem_n, mem_n1),
        })
    return results


def plot_memory_metrics(
    state: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot M1 (BFRR), M2 (CTPR), and M3 (TFR) as a grouped bar chart.

    Args:
        state: Full run state containing ``all_therapist_memories``.
        metadata: Run metadata dict.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_therapist_memories: list[str] = state.get("all_therapist_memories", [])
        if len(all_therapist_memories) < 2:
            logger.info("Fewer than 2 therapist memories — skipping memory metrics plot.")
            return

        memories = _parse_memories(all_therapist_memories)
        results = _compute_all_metrics(memories)
        if not results:
            logger.info("No valid consecutive memory pairs — skipping memory metrics plot.")
            return

        transitions = [r["transition"] for r in results]
        bfrr_vals = [r["bfrr"] for r in results]
        ctpr_vals = [r["ctpr"] for r in results]
        tfr_vals = [r["tfr"] for r in results]

        # Determine which metrics have data
        has_bfrr = any(v is not None for v in bfrr_vals)
        has_ctpr = any(v is not None for v in ctpr_vals)
        has_tfr = any(v is not None for v in tfr_vals)

        if not has_bfrr and not has_ctpr and not has_tfr:
            logger.info("No memory metric data available — skipping plot.")
            return

        if not has_ctpr and not has_tfr:
            logger.warning(
                "Active_Change_Talk_Threads not found in therapist memories. "
                "M2 (CTPR) and M3 (TFR) are skipped. Only M1 (BFRR) is plotted."
            )

        # Build bars for available metrics
        metric_info: list[tuple[str, list[Optional[float]], str]] = []
        if has_bfrr:
            metric_info.append(("M1: BFRR", bfrr_vals, "#2ecc71"))
        if has_ctpr:
            metric_info.append(("M2: CTPR", ctpr_vals, "#3498db"))
        if has_tfr:
            metric_info.append(("M3: TFR", tfr_vals, "#e67e22"))

        n_metrics = len(metric_info)
        n_transitions = len(transitions)
        bar_width = 0.8 / n_metrics
        x = np.arange(n_transitions)

        fig, ax = plt.subplots(figsize=(max(8, n_transitions * 2.5), 6))

        for idx, (label, values, color) in enumerate(metric_info):
            offset = (idx - (n_metrics - 1) / 2) * bar_width
            plot_vals = [v if v is not None else 0 for v in values]
            bars = ax.bar(x + offset, plot_vals, bar_width,
                          label=label, color=color, edgecolor="white", linewidth=0.5)
            # Annotate bars
            for i, (bar, raw_val) in enumerate(zip(bars, values)):
                if raw_val is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{raw_val:.0f}%", ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color=color)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, 2,
                            "N/A", ha="center", va="bottom",
                            fontsize=8, color="#999", fontstyle="italic")

        # Ideal retention line
        ax.axhline(100, color="#27ae60", linewidth=1.2, linestyle="--", alpha=0.5,
                    label="Ideal (100%)")

        ax.set_xlabel("Session Transition", fontsize=12)
        ax.set_ylabel("Rate (%)", fontsize=12)
        ax.set_title("Therapist Memory Fidelity Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(transitions, fontsize=11)
        ax.set_ylim(0, 115)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotation box with metric definitions
        legend_lines = []
        if has_bfrr:
            legend_lines.append("M1 BFRR: Biographical facts retained across sessions")
        if has_ctpr:
            legend_lines.append("M2 CTPR: Change talk threads persisted (by id)")
        if has_tfr:
            legend_lines.append("M3 TFR: Open threads followed up in next session")
        ax.text(0.02, 0.02, "\n".join(legend_lines),
                transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

        plt.savefig(output_dir / "23_memory_metrics.png", bbox_inches="tight")
        plt.close()
        logger.info(f"Memory metrics plot saved to {output_dir / '23_memory_metrics.png'}")
    except Exception as e:
        logger.error(f"Failed to plot memory metrics: {e}", exc_info=True)
        plt.close()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.plot_memory_metrics <path_to_json>")
        sys.exit(1)

    from src.analytics.analyzer import load_json, create_output_dir

    json_path = sys.argv[1]
    data = load_json(json_path)
    metadata = data.get("metadata", {})
    state = data.get("state", data)
    output_dir = create_output_dir(metadata, json_path)
    plot_memory_metrics(state, metadata, output_dir)
