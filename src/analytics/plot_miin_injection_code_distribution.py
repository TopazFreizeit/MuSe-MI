"""Visualization: MIIN injection technique_used distribution.

Tracks the distribution of MI-Inconsistent (IMI) codes that the therapist
was *instructed* to produce during MIIN injection turns only.  This is
prompt *requested*, not what the downstream annotator classified.

Charts produced:
    Left  — Per-session stacked bar of MIIN code counts.
    Right — Aggregate donut showing the overall distribution across all sessions.
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# Canonical MIIN codes and their colours (consistent with plot_miin_fatal_flaw)
MIIN_CODES = ["ADWP", "CON", "DIR", "RCWP", "WA"]
_MIIN_COLORS = {
    "CON":  "#c0392b",  # dark red — most severe
    "DIR":  "#e74c3c",  # red
    "ADWP": "#e67e22",  # orange
    "RCWP": "#f39c12",  # amber
    "WA":   "#d35400",  # burnt orange
}
_MIIN_LABELS = {
    "ADWP": "Advise w/o Permission",
    "CON":  "Confront",
    "DIR":  "Direct",
    "RCWP": "Raise Concern w/o Permission",
    "WA":   "Warn",
}


def _extract_miin_injection_codes(
    all_session_turns: List[List[Dict[str, Any]]],
) -> List[Counter]:
    """Return a list of Counters (one per session) of technique_used
    values from MIIN injection turns only."""
    session_counters: list[Counter] = []
    for turns in all_session_turns:
        counts: Counter = Counter()
        for turn in turns:
            if turn.get("speaker") == "therapist" and turn.get("miin_injection"):
                code = (turn.get("technique_used") or "").strip()
                if code:
                    counts[code] += 1
        session_counters.append(counts)
    return session_counters


def plot_miin_injection_code_distribution(
    state: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot the distribution of technique_used across MIIN injection turns.

    Produces a two-panel figure:
        Left:  Per-session stacked bar chart.
        Right: Aggregate donut chart across all sessions.

    Args:
        state: Full run state containing ``all_session_turns``.
        metadata: Run metadata dict.
        output_dir: Directory to save the plot PNG.
    """
    try:
        all_session_turns: list[list[dict]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping MIIN injection code distribution plot.")
            return

        session_counters = _extract_miin_injection_codes(all_session_turns)

        # Global totals
        global_counts: Counter = Counter()
        for c in session_counters:
            global_counts.update(c)

        if not global_counts:
            logger.info("No MIIN injection turns found — skipping MIIN injection code distribution plot.")
            return

        # Only keep codes that actually appeared (sorted by canonical order)
        active_codes = [c for c in MIIN_CODES if global_counts.get(c, 0) > 0]
        if not active_codes:
            logger.info("No recognised MIIN codes in injection turns — skipping plot.")
            return

        n_sessions = len(session_counters)
        session_labels = [f"S{i + 1}" for i in range(n_sessions)]

        # ── Figure layout ────────────────────────────────────────────
        fig, (ax_bar, ax_donut) = plt.subplots(
            1, 2,
            figsize=(max(10, n_sessions * 2.2 + 5), 6),
            gridspec_kw={"width_ratios": [3, 1]},
        )

        # ── Left: per-session stacked bar chart ─────────────────────
        x = np.arange(n_sessions)
        bar_width = 0.55
        bottom = np.zeros(n_sessions)

        for code in active_codes:
            values = np.array([sc.get(code, 0) for sc in session_counters], dtype=float)
            ax_bar.bar(
                x, values, bar_width,
                bottom=bottom,
                label=f"{code} — {_MIIN_LABELS.get(code, code)}",
                color=_MIIN_COLORS.get(code, "#999999"),
                edgecolor="white",
                linewidth=0.5,
            )
            # Annotate non-zero cells
            for i, v in enumerate(values):
                if v > 0:
                    ax_bar.text(
                        x[i], bottom[i] + v / 2, str(int(v)),
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white",
                    )
            bottom += values

        # Session totals on top of bars
        totals = np.array([sum(sc.values()) for sc in session_counters], dtype=float)
        for i, t in enumerate(totals):
            if t > 0:
                ax_bar.text(
                    x[i], t + 0.15, str(int(t)),
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                )

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(session_labels, fontsize=11)
        ax_bar.set_xlabel("Session", fontsize=12)
        ax_bar.set_ylabel("MIIN Injection Count", fontsize=12)
        ax_bar.set_title("MIIN Injection Code Distribution per Session", fontsize=13, fontweight="bold")
        ax_bar.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_bar.legend(fontsize=8, loc="upper left", framealpha=0.9)

        total_injections = int(sum(global_counts.values()))
        ax_bar.text(
            0.98, 0.98, f"Total injections: {total_injections}",
            transform=ax_bar.transAxes, fontsize=10,
            va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

        # ── Right: aggregate donut chart ─────────────────────────────
        donut_values = [global_counts[c] for c in active_codes]
        donut_colors = [_MIIN_COLORS.get(c, "#999999") for c in active_codes]
        donut_labels = [f"{c}\n({v})" for c, v in zip(active_codes, donut_values)]

        wedges, texts, autotexts = ax_donut.pie(
            donut_values,
            labels=donut_labels,
            colors=donut_colors,
            autopct="%1.0f%%",
            pctdistance=0.75,
            startangle=90,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")
        for t in texts:
            t.set_fontsize(9)

        ax_donut.set_title("Overall Distribution", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_dir / "26_miin_injection_code_distribution.png", bbox_inches="tight")
        plt.close()
        logger.info(f"MIIN injection code distribution plot saved to {output_dir / '26_miin_injection_code_distribution.png'}")
    except Exception as e:
        logger.error(f"Failed to plot MIIN injection code distribution: {e}", exc_info=True)
        plt.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.plot_miin_injection_code_distribution <path_to_json>")
        sys.exit(1)

    from src.analytics.analyzer import load_json, create_output_dir

    json_path = sys.argv[1]
    data = load_json(json_path)
    metadata = data.get("metadata", {})
    state = data.get("state", data)
    output_dir = create_output_dir(metadata, json_path)
    plot_miin_injection_code_distribution(state, metadata, output_dir)
    logger.info(f"Plot saved to {output_dir}")
