"""
Visualize SOCRATES + Amrhein + Stage of Change + Global MITI judge results across one or more runs.

Reads `judge_results.json` files (as produced by `src.judge.run_judges`) and
emits four PNGs:
  - socrates_trajectories.png  — 1x3 grid: cols = {Recognition, Ambivalence, Taking Steps}.
                                  One line per input file per panel.
  - amrhein_summary.png        — 1x3: mean strength (CT & ST); CT Ratio; code counts at final session.
  - stage_of_change_trajectories.png — 1x3 grid: PC, C, A scores over sessions.
  - global_miti_scores.png     — 1x4 grid: 4 MITI global scores over sessions.

Usage:
    python -m src.analytics.plot_judge_results <judge_results.json> [<more.json> ...] [--output-dir DIR]

If --output-dir is omitted, PNGs are written to the parent directory of the
first input file.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

SOCRATES_RANGES = {
    "recognition": (7, 35),
    "ambivalence": (4, 20),
    "taking_steps": (8, 40),
}
SUBSCALES = ("recognition", "ambivalence", "taking_steps")
SUBSCALE_LABELS = {
    "recognition": "Recognition (7–35)",
    "ambivalence": "Ambivalence (4–20)",
    "taking_steps": "Taking Steps (8–40)",
}
AMRHEIN_CODES = ("C", "A", "D", "R", "N", "TS")


def load_run(path: Path) -> Tuple[str, dict]:
    with open(path) as f:
        data = json.load(f)
    label = path.parent.name or path.stem
    return label, data


def _socrates_trajectory(run: dict, mode: str, subscale: str) -> Tuple[List[int], List[int]]:
    """Return (sessions, scores) for a single (mode, subscale)."""
    block = run.get(f"socrates_{mode}", {})
    if not block:
        return [], []
    sessions, scores = [], []
    for s in sorted(block.keys(), key=int):
        result = block[s]
        if result is None:
            continue
        sessions.append(int(s))
        scores.append(result["subscales"][subscale])
    return sessions, scores


def _amrhein_metrics(run: dict) -> Tuple[List[int], List[Optional[float]], List[Optional[float]], List[Optional[int]], List[Optional[int]], List[Optional[float]]]:
    """Returns sessions, ct_means, st_means, n_cts, n_sts, ct_ratios"""
    block = run.get("amrhein", {})
    if not block:
        return [], [], [], [], [], []
    
    sessions = []
    ct_means = []
    st_means = []
    n_cts = []
    n_sts = []
    ct_ratios = []
    
    for s in sorted(block.keys(), key=int):
        result = block[s]
        if result is None:
            continue
        sessions.append(int(s))
        
        # Handle new nested format or old flat format
        if "change_talk" in result and "sustain_talk" in result:
            ct = result["change_talk"]
            st = result["sustain_talk"]
            ct_means.append(ct.get("mean_strength"))
            st_means.append(st.get("mean_strength"))
            
            n_ct = ct.get("n_utterances", 0)
            n_st = st.get("n_utterances", 0)
            n_cts.append(n_ct)
            n_sts.append(n_st)
            
            total = n_ct + n_st
            ct_ratios.append(n_ct / total if total > 0 else None)
        else:
            # Legacy format
            ct_means.append(result.get("mean_strength"))
            st_means.append(None)
            n_ct = result.get("n_utterances", 0)
            n_cts.append(n_ct)
            n_sts.append(None)
            ct_ratios.append(None)
            
    return sessions, ct_means, st_means, n_cts, n_sts, ct_ratios


def _amrhein_final_counts(run: dict) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """Returns (ct_counts, st_counts) at the final session."""
    block = run.get("amrhein", {})
    if not block:
        return None, None
    last_key = max(block.keys(), key=int)
    result = block[last_key]
    if result is None:
        return None, None
        
    if "change_talk" in result and "sustain_talk" in result:
        return result["change_talk"]["code_counts"], result["sustain_talk"]["code_counts"]
    else:
        # Legacy
        return result.get("code_counts"), None


def _stage_of_change_trajectory(run: dict, subscale: str) -> Tuple[List[int], List[int]]:
    block = run.get("stage_of_change", {})
    if not block:
        return [], []
    sessions, scores = [], []
    for s in sorted(block.keys(), key=int):
        result = block[s]
        if result is None:
            continue
        sessions.append(int(s))
        scores.append(result[f"{subscale}_score"])
    return sessions, scores


def _global_miti_trajectory(run: dict, score_name: str) -> Tuple[List[int], List[Optional[int]]]:
    block = run.get("global_miti", {})
    if not block:
        return [], []
    sessions, scores = [], []
    for s in sorted(block.keys(), key=int):
        result = block[s]
        if result is None:
            continue
        sessions.append(int(s))
        if score_name in result:
            scores.append(result[score_name]["score"])
        else:
            scores.append(None)
    return sessions, scores


def plot_socrates(runs: List[Tuple[str, dict]], out_path: Path) -> Optional[Path]:
    have_any = any(run.get("socrates_self") for _, run in runs)
    if not have_any:
        print("[skip] no socrates_self block in any run")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    cmap = plt.get_cmap("tab10")

    for col, sub in enumerate(SUBSCALES):
        ax = axes[col]
        plotted = False
        for idx, (label, run) in enumerate(runs):
            xs, ys = _socrates_trajectory(run, "self", sub)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", color=cmap(idx % 10), label=label)
            plotted = True
        ax.set_title(f"SOCRATES Self — {SUBSCALE_LABELS[sub]}")
        lo, hi = SOCRATES_RANGES[sub]
        ax.set_ylim(lo - 1, hi + 1)
        ax.axhspan(lo, hi, color="0.95", zorder=0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Session")
        if col == 0:
            ax.set_ylabel("Self-Report")
        if not plotted:
            ax.text(
                0.5, 0.5, "no socrates_self data", ha="center", va="center",
                transform=ax.transAxes, color="gray",
            )

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.1))

    fig.suptitle("SOCRATES 8A Subscale Trajectories")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def plot_amrhein(runs: List[Tuple[str, dict]], out_path: Path) -> Optional[Path]:
    have_any = any(run.get("amrhein") for _, run in runs)
    if not have_any:
        print("[skip] no amrhein blocks in any run")
        return None

    fig, (ax_mean, ax_ratio, ax_counts) = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap("tab10")

    # 1. Mean Strength (CT & ST) Bubble Chart
    for idx, (label, run) in enumerate(runs):
        xs, ct_means, st_means, n_cts, n_sts, _ = _amrhein_metrics(run)
        if not xs:
            continue
        ct_plot = [np.nan if v is None else v for v in ct_means]
        st_plot = [np.nan if v is None else v for v in st_means]
        
        c = cmap(idx % 10)
        # Plot lines (no markers)
        ax_mean.plot(xs, ct_plot, color=c, label=f"{label} (CT)", linestyle="-", alpha=0.6)
        if any(v is not None for v in st_means):
            ax_mean.plot(xs, st_plot, color=c, label=f"{label} (ST)", linestyle="--", alpha=0.6)
            
        # Plot bubbles scaled by count
        base_size = 30
        for i, x in enumerate(xs):
            if n_cts[i] is not None and n_cts[i] > 0 and ct_means[i] is not None:
                ax_mean.scatter(x, ct_means[i], s=n_cts[i] * base_size, color=c, alpha=0.8, edgecolors='white', zorder=3)
            if n_sts[i] is not None and n_sts[i] > 0 and st_means[i] is not None:
                ax_mean.scatter(x, st_means[i], s=n_sts[i] * base_size, color=c, marker='s', alpha=0.8, edgecolors='white', zorder=3)
            
    ax_mean.set_title("Mean Strength (Bubble size = utterance count)")
    ax_mean.set_xlabel("Session")
    ax_mean.set_ylabel("Strength (1–5)")
    ax_mean.set_ylim(0.5, 5.5)
    ax_mean.axhspan(1, 5, color="0.95", zorder=0)
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend(loc="best", fontsize=8)

    # 2. CT Ratio
    for idx, (label, run) in enumerate(runs):
        xs, _, _, _, _, ratios = _amrhein_metrics(run)
        if not xs or all(r is None for r in ratios):
            continue
        ratio_plot = [np.nan if v is None else v for v in ratios]
        ax_ratio.plot(xs, ratio_plot, marker="o", color=cmap(idx % 10), label=label)

    ax_ratio.set_title("Proportion of Change Talk")
    ax_ratio.set_xlabel("Session")
    ax_ratio.set_ylabel("Ratio: CT / (CT + ST)")
    ax_ratio.set_ylim(-0.05, 1.05)
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend(loc="best", fontsize=8)

    # 3. Final Code Counts
    bar_runs = []
    for label, run in runs:
        ct_counts, st_counts = _amrhein_final_counts(run)
        if ct_counts is None:
            continue
        ct_vals = [ct_counts.get(c, 0) for c in AMRHEIN_CODES]
        st_vals = [st_counts.get(c, 0) for c in AMRHEIN_CODES] if st_counts else [0]*len(AMRHEIN_CODES)
        bar_runs.append((label, ct_vals, st_vals))

    if bar_runs:
        n_runs = len(bar_runs)
        x = np.arange(len(AMRHEIN_CODES))
        width = 0.8 / n_runs
        
        for i, (label, ct_vals, st_vals) in enumerate(bar_runs):
            pos = x + i * width - 0.4 + width / 2
            c = cmap(i % 10)
            
            # Stacked or dodged? Since we have multiple runs, let's plot net CT and ST differently 
            # Or just plot CT as solid, and ST as negative to make it a tornado plot
            # Let's plot ST underneath 0 to visually separate them
            ax_counts.bar(pos, ct_vals, width, label=f"{label} (CT)", color=c)
            if any(st_vals):
                # Negate ST vals to plot them downward
                neg_st_vals = [-v for v in st_vals]
                ax_counts.bar(pos, neg_st_vals, width, color=c, alpha=0.5, hatch='//', label=f"{label} (ST)")
            
        ax_counts.set_xticks(x)
        ax_counts.set_xticklabels(AMRHEIN_CODES)
        ax_counts.set_title("Final Session Counts (ST shown negative)")
        ax_counts.set_xlabel("Code (C, A, D, N, TS)")
        ax_counts.set_ylabel("Count")
        ax_counts.grid(True, axis="y", alpha=0.3)
        # Fix y-axis to be symmetric if ST exists
        y_min, y_max = ax_counts.get_ylim()
        bound = max(abs(y_min), abs(y_max))
        ax_counts.set_ylim(-bound, bound)
        ax_counts.legend(loc="best", fontsize=8)
    else:
        ax_counts.text(0.5, 0.5, "no count data", ha="center", va="center", color="gray")

    fig.suptitle("Amrhein Commitment-Strength Summary")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def plot_stage_of_change(runs: List[Tuple[str, dict]], out_path: Path) -> Optional[Path]:
    have_any = any(run.get("stage_of_change") for _, run in runs)
    if not have_any:
        print("[skip] no stage_of_change blocks in any run")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    scales = (("pc", "Precontemplation (PC)", (-8, 8)),
              ("c", "Contemplation (C)", (-8, 8)),
              ("a", "Action (A)", (-8, 8)))

    cmap = plt.get_cmap("tab10")

    for col, (sub, label, (lo, hi)) in enumerate(scales):
        ax = axes[col]
        plotted = False
        for idx, (run_label, run) in enumerate(runs):
            xs, ys = _stage_of_change_trajectory(run, sub)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", color=cmap(idx % 10), label=run_label)
            plotted = True
        ax.set_title(f"RCQ-TV — {label}")
        ax.set_ylim(lo - 1, hi + 1)
        ax.axhspan(lo, hi, color="0.95", zorder=0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Session")
        if col == 0:
            ax.set_ylabel("Score")
        if not plotted:
            ax.text(0.5, 0.5, f"no data", ha="center", va="center", transform=ax.transAxes, color="gray")

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.1))

    fig.suptitle("Stage of Change (RCQ-TV) trajectories")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def plot_global_miti(runs: List[Tuple[str, dict]], out_path: Path) -> Optional[Path]:
    have_any = any(run.get("global_miti") for _, run in runs)
    if not have_any:
        print("[skip] no global_miti blocks in any run")
        return None

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharex=True)
    scores = ("Cultivating Change Talk", "Softening Sustain Talk", "Partnership", "Empathy")
    cmap = plt.get_cmap("tab10")

    for col, score_name in enumerate(scores):
        ax = axes[col]
        plotted = False
        for idx, (run_label, run) in enumerate(runs):
            xs, ys = _global_miti_trajectory(run, score_name)
            if not xs:
                continue
            
            ys_plot = [np.nan if v is None else v for v in ys]
            ax.plot(xs, ys_plot, marker="o", color=cmap(idx % 10), label=run_label)
            plotted = True
        ax.set_title(score_name)
        ax.set_ylim(0.5, 5.5)
        ax.axhspan(1, 5, color="0.95", zorder=0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Session")
        if col == 0:
            ax.set_ylabel("Score (1-5)")
        if not plotted:
            ax.text(0.5, 0.5, f"no data", ha="center", va="center", transform=ax.transAxes, color="gray")

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.1))

    fig.suptitle("Global MITI Scores trajectories")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path,
                   help="One or more judge_results.json files")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to write PNGs (default: parent of first input)")
    args = p.parse_args()

    runs = []
    for path in args.inputs:
        if not path.exists():
            raise SystemExit(f"file not found: {path}")
        runs.append(load_run(path))

    out_dir = args.output_dir or args.inputs[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_socrates(runs, out_dir / "socrates_trajectories.png")
    plot_amrhein(runs, out_dir / "amrhein_summary.png")
    plot_stage_of_change(runs, out_dir / "stage_of_change_trajectories.png")
    plot_global_miti(runs, out_dir / "global_miti_scores.png")


if __name__ == "__main__":
    main()
