#!/usr/bin/env python3
"""
Publication-ready visualizer and statistical analyzer comparing Patient Ambivalence
across Patient Variations (Patient-Psi, SimPatient, MuSeMI) and Therapist MI Competence levels.

Experimental Matrix:
  - 3 Patient Architectures: MuSeMI (Ours), Patient-Psi, SimPatient
  - 3 Therapist Competence Levels:
      1. MI-Consistent (Good Therapist / 0 MI-IN)
      2. MI-Inconsistent (3 MI-IN)
      3. Highly MI-Inconsistent (6 MI-IN)
  - 30 Clinical Personas / Patients across 4 Longitudinal Sessions (1,080 session observations)

Visualizations:
  1. Primary Figure (1x3 Panel): Longitudinal Composite Ambivalence Trajectories (Sessions 1-4)
     side-by-side for MuSeMI, Patient-Psi, and SimPatient with within-subject Cousineau-Morey 95% CIs.
  2. Multi-Panel Figure (3x3 Grid): Decomposed Ambivalence Dimensions:
     Problem Uncertainty (U), Decisional Conflict (D), Conversational Vacillation (V) across all architectures.
  3. Sensitivity & Progress Figure (2x3 Panel):
     - Top: Longitudinal Ambivalence Resolution (Session 4 - Session 1 Delta)
     - Bottom: Therapist Discriminative Power at Session 4 (Highly Inconsistent minus MI-Consistent)

Statistical Exports:
  - Full CSV dataset (1,080 rows)
  - Summary tables (Mean +/- SD, deltas, paired Wilcoxon p-values, Cohen's dz) in CSV, Markdown, and LaTeX.
"""

import os
import sys
import glob
import yaml
import pathlib
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
sys.path.append(str(ROOT))

TRANSCRIPTS_DIR = ROOT / "transcripts"
OUT_DIR = ROOT / "src_ambivalence" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUTOMISC_OUT_DIR = ROOT / "src_automisc" / "results"
AUTOMISC_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication Plotting Theme (Matching ACL / IEEE / Nature style) ────────────
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 20,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.titlesize": 26,
    "lines.linewidth": 3.0,
    "lines.markersize": 10,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.45,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Therapist Competence Configuration (Okabe-Ito Colorblind-Safe)
COMPETENCE_CONFIG = {
    "Good": {
        "label": "MI-Consistent (Good Therapist)",
        "color": "#009E73",      # Green
        "marker": "o",
        "linestyle": "-",
    },
    "Inconsistent": {
        "label": "MI-Inconsistent (3 MI-IN)",
        "color": "#56B4E9",      # Sky Blue
        "marker": "s",
        "linestyle": "--",
    },
    "Highly_Inconsistent": {
        "label": "Highly MI-Inconsistent (6 MI-IN)",
        "color": "#D55E00",      # Vermillion
        "marker": "^",
        "linestyle": ":",
    },
}

MODEL_ORDER = ["MuSeMI", r"Patient-$\Psi$", "SimPatient"]

MODEL_DISPLAY_NAMES = {
    "MuSeMI": "MuSeMI (Ours)",
    r"Patient-$\Psi$": r"Patient-$\Psi$",
    "SimPatient": "SimPatient",
}

# ── Data Loading & Classification ──────────────────────────────────────────────

def classify_ambivalence_file(filename: str) -> Tuple[str, str]:
    """
    Classifies an ambivalence evaluation file into (model_name, competence_level).
    """
    fn = filename.lower()
    
    # 1. SimPatient
    if "simpatient" in fn:
        model = "SimPatient"
        if "miin_count_6" in fn or "more_adversarial" in fn:
            comp = "Highly_Inconsistent"
        elif "miin_count_3" in fn or "adversarial" in fn:
            comp = "Inconsistent"
        else:
            comp = "Good"
        return model, comp

    # 2. Patient-Psi
    if "patient_psi" in fn:
        model = r"Patient-$\Psi$"
        if "miin_count_6" in fn or "more_adversarial" in fn:
            comp = "Highly_Inconsistent"
        elif "miin_count_3" in fn or "adversarial" in fn:
            comp = "Inconsistent"
        else:
            comp = "Good"
        return model, comp

    # 3. MuSeMI (default / original)
    if "disable_cognitive_interpreter" in fn:
        model = "No Cog. Interp."
        if "miin_count_6" in fn or "more_adversarial" in fn:
            comp = "Highly_Inconsistent"
        elif "miin_count_3" in fn or "adversarial" in fn:
            comp = "Inconsistent"
        else:
            comp = "Good"
        return model, comp

    # MuSeMI conditions
    model = "MuSeMI"
    if "more_adversarial" in fn:
        comp = "Highly_Inconsistent"
    elif "adversarial" in fn:
        comp = "Inconsistent"
    else:
        comp = "Good"
    return model, comp


def load_baselines_ambivalence_data(transcripts_dir: pathlib.Path = TRANSCRIPTS_DIR) -> pd.DataFrame:
    """
    Scans transcripts directory for all *_ambivalence_gemma-4-31b-it.yaml files
    and compiles a tidy DataFrame across all 12 conditions (or 9 baseline conditions).
    """
    records = []
    yaml_files = sorted(transcripts_dir.glob("*/*ambivalence_gemma-4-31b-it.yaml"))
    
    print(f"Discovered {len(yaml_files)} ambivalence evaluation files.")

    for yf in yaml_files:
        model, comp = classify_ambivalence_file(yf.name)
        if model not in MODEL_ORDER:
            continue

        # Determine patient id
        if yf.parent.name.isdigit():
            pat_id = int(yf.parent.name)
        else:
            pat_id = int(yf.stem.split("_")[0])

        with open(yf, "r", encoding="utf-8") as fin:
            try:
                data = yaml.safe_load(fin)
            except Exception as e:
                print(f"Skipping {yf.name} due to parse error: {e}")
                continue

        if not data or "sessions" not in data:
            continue

        for session in data["sessions"]:
            s_num = session.get("session_num")
            eval_data = session.get("evaluation", {})
            comp_score = eval_data.get("composite_ambivalence_score")
            
            u_mean = eval_data.get("problem_uncertainty", {}).get("dimension_mean")
            d_mean = eval_data.get("decisional_conflict", {}).get("dimension_mean")
            v_mean = eval_data.get("conversational_vacillation", {}).get("dimension_mean")

            u_scores = eval_data.get("problem_uncertainty", {}).get("item_scores", {})
            d_scores = eval_data.get("decisional_conflict", {}).get("item_scores", {})
            v_scores = eval_data.get("conversational_vacillation", {}).get("item_scores", {})

            records.append({
                "patient_id": pat_id,
                "model": model,
                "competence": comp,
                "competence_label": COMPETENCE_CONFIG[comp]["label"],
                "session": s_num,
                "composite_ambivalence": float(comp_score) if comp_score is not None else np.nan,
                "problem_uncertainty": float(u_mean) if u_mean is not None else np.nan,
                "decisional_conflict": float(d_mean) if d_mean is not None else np.nan,
                "conversational_vacillation": float(v_mean) if v_mean is not None else np.nan,
                "U1": u_scores.get("U1"),
                "U2": u_scores.get("U2"),
                "U3": u_scores.get("U3"),
                "D1": d_scores.get("D1"),
                "D2": d_scores.get("D2"),
                "D3": d_scores.get("D3"),
                "V1": v_scores.get("V1"),
                "V2": v_scores.get("V2"),
                "V3": v_scores.get("V3"),
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} total session evaluations across {df['patient_id'].nunique()} patients.")
    print("Condition distribution:")
    print(df.groupby(["model", "competence"])["session"].count())
    return df


def apply_cousineau_morey(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Applies Cousineau-Morey (2008) correction for repeated measures.
    Removes between-subject variance per (model, competence) group so 95% CIs
    accurately reflect within-subject variance across longitudinal sessions.
    """
    df_corr = df.copy()
    for metric in metric_cols:
        corr_col = f"{metric}_corr"
        # Group by model and competence to normalize within each experimental condition
        def _cm_adjust(group):
            J = group.groupby("patient_id")["session"].transform("count").max()
            if J > 1:
                subj_mean = group.groupby("patient_id")[metric].transform("mean")
                grand_mean = group[metric].mean()
                multiplier = np.sqrt(J / (J - 1))
                return grand_mean + multiplier * (group[metric] - subj_mean)
            return group[metric]

        df_corr[corr_col] = df_corr.groupby(["model", "competence"], group_keys=False).apply(_cm_adjust)
    return df_corr


# ── Figure 1: 1x3 Primary Trajectories Panel ───────────────────────────────────

def plot_ambivalence_baselines_trajectories(df: pd.DataFrame):
    """
    Figure 1: 1x3 panel layout showing longitudinal Composite Ambivalence (Sessions 1-4)
    across the 3 Therapist Competence levels for MuSeMI, Patient-Psi, and SimPatient.
    """
    df_cm = apply_cousineau_morey(df, ["composite_ambivalence"])

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(19, 6.2),
        sharey=True,
        constrained_layout=False,
    )
    plt.subplots_adjust(top=0.86, bottom=0.22, left=0.07, right=0.98, wspace=0.15)

    y_min, y_max = 1.0, 5.0

    for idx, model_name in enumerate(MODEL_ORDER):
        ax = axes[idx]
        model_data = df_cm[df_cm["model"] == model_name]

        # Clinical reference line: Moderate ambivalence threshold = 3.0
        ax.axhline(
            3.0,
            color="#888888",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=1,
        )
        if idx == 2:
            ax.text(
                4.05, 3.0, "Moderate Threshold (3.0)",
                color="#555555",
                fontsize=13,
                va="center",
                style="italic",
            )

        for comp_key, cfg in COMPETENCE_CONFIG.items():
            comp_data = model_data[model_data["competence"] == comp_key]
            if comp_data.empty:
                continue

            # Plot Cousineau-Morey adjusted line with 95% within-subject CI
            sns.lineplot(
                data=comp_data,
                x="session",
                y="composite_ambivalence_corr",
                ax=ax,
                color=cfg["color"],
                marker=cfg["marker"],
                linestyle=cfg["linestyle"],
                linewidth=3.2,
                markersize=10,
                errorbar="ci",
                err_style="bars",
                err_kws={"capsize": 4.5, "elinewidth": 1.8},
                legend=False,
                zorder=3,
            )

        # Panel title and styling
        ax.set_title(MODEL_DISPLAY_NAMES.get(model_name, model_name), pad=12, fontsize=23)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["S1", "S2", "S3", "S4"], fontsize=20)
        ax.set_xlabel("Clinical Session", fontsize=21, labelpad=8)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0.8, 4.2)

        if idx == 0:
            ax.set_ylabel("Composite Ambivalence (1–5)", fontsize=21)
        else:
            ax.set_ylabel("")

    # Unified Legend at bottom
    handles = [
        Line2D(
            [0], [0],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            marker=cfg["marker"],
            lw=3.2,
            markersize=10,
            label=cfg["label"],
        )
        for cfg in COMPETENCE_CONFIG.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=20,
        columnspacing=2.2,
        handletextpad=0.5,
    )

    png_path = OUT_DIR / "fig_ambivalence_baselines_trajectories.png"
    pdf_path = OUT_DIR / "fig_ambivalence_baselines_trajectories.pdf"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")
    
    # Mirror to automisc
    fig.savefig(str(AUTOMISC_OUT_DIR / "fig_ambivalence_baselines_trajectories.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(AUTOMISC_OUT_DIR / "fig_ambivalence_baselines_trajectories.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Trajectories Figure: {png_path.name} | {pdf_path.name}")


# ── Figure 2: 3x3 Multi-Panel Dimensions Grid ──────────────────────────────────

def plot_ambivalence_baselines_multipanel(df: pd.DataFrame):
    """
    Figure 2: 3x3 facet grid:
      Rows: MuSeMI, Patient-Psi, SimPatient
      Cols: Problem Uncertainty (U), Decisional Conflict (D), Conversational Vacillation (V)
    """
    dim_metrics = [
        ("problem_uncertainty", "Problem Uncertainty (U)"),
        ("decisional_conflict", "Decisional Conflict (D)"),
        ("conversational_vacillation", "Conversational Vacillation (V)"),
    ]
    df_cm = apply_cousineau_morey(df, [m[0] for m in dim_metrics])

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(18, 14),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.93, hspace=0.25, wspace=0.15)

    for row_idx, model_name in enumerate(MODEL_ORDER):
        model_data = df_cm[df_cm["model"] == model_name]

        for col_idx, (metric_col, metric_title) in enumerate(dim_metrics):
            ax = axes[row_idx, col_idx]
            corr_col = f"{metric_col}_corr"

            # Horizontal reference at 3.0
            ax.axhline(3.0, color="#888888", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)

            for comp_key, cfg in COMPETENCE_CONFIG.items():
                comp_data = model_data[model_data["competence"] == comp_key]
                if comp_data.empty:
                    continue

                sns.lineplot(
                    data=comp_data,
                    x="session",
                    y=corr_col,
                    ax=ax,
                    color=cfg["color"],
                    marker=cfg["marker"],
                    linestyle=cfg["linestyle"],
                    linewidth=2.8,
                    markersize=8.5,
                    errorbar="ci",
                    err_style="bars",
                    err_kws={"capsize": 4, "elinewidth": 1.5},
                    legend=False,
                    zorder=3,
                )

            # Column Titles on top row
            if row_idx == 0:
                ax.set_title(metric_title, pad=12, fontsize=22)

            # Row Labels on right margin
            if col_idx == 2:
                ax.text(
                    1.05, 0.5, MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    transform=ax.transAxes,
                    fontsize=21,
                    fontweight="bold",
                    va="center",
                    rotation=270,
                )

            # Left Y-axis labels
            if col_idx == 0:
                ax.set_ylabel("Score (1–5)", fontsize=20)
            else:
                ax.set_ylabel("")

            # Bottom X-axis labels
            if row_idx == 2:
                ax.set_xlabel("Session", fontsize=20)
                ax.set_xticks([1, 2, 3, 4])
                ax.set_xticklabels(["S1", "S2", "S3", "S4"], fontsize=19)
            else:
                ax.set_xlabel("")

            ax.set_ylim(1.0, 5.0)
            ax.set_xlim(0.8, 4.2)

    # Unified Legend at bottom
    handles = [
        Line2D(
            [0], [0],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            marker=cfg["marker"],
            lw=3.0,
            markersize=9,
            label=cfg["label"],
        )
        for cfg in COMPETENCE_CONFIG.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=20,
        columnspacing=2.2,
        handletextpad=0.5,
    )

    png_path = OUT_DIR / "fig_ambivalence_baselines_multipanel.png"
    pdf_path = OUT_DIR / "fig_ambivalence_baselines_multipanel.pdf"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")

    fig.savefig(str(AUTOMISC_OUT_DIR / "fig_ambivalence_baselines_multipanel.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(AUTOMISC_OUT_DIR / "fig_ambivalence_baselines_multipanel.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Multipanel Grid: {png_path.name} | {pdf_path.name}")


# ── Figure 3: Progress & Discriminative Sensitivity ────────────────────────────

def plot_ambivalence_progress_and_differentiation(df: pd.DataFrame):
    """
    Figure 3: 2-row layout:
      Top Row (1x3): Net Ambivalence Resolution Delta (Session 4 - Session 1)
                     for Composite Ambivalence, Decisional Conflict, and Vacillation.
                     Negative delta = Successful Ambivalence Resolution.
      Bottom Row (1x3): Therapist Discriminative Separation at Session 4
                        (Highly Inconsistent minus MI-Consistent).
                        Positive gap = Active clinical sensitivity to therapist competence.
    """
    # 1. Pivot to compute S4 - S1 Delta per patient
    piv = df.pivot_table(
        index=["model", "competence", "patient_id"],
        columns="session",
        values=["composite_ambivalence", "problem_uncertainty", "decisional_conflict", "conversational_vacillation"],
    )

    delta_rows = []
    for (model, comp, pid), row in piv.iterrows():
        delta_rows.append({
            "model": model,
            "competence": comp,
            "patient_id": pid,
            "delta_composite": row[("composite_ambivalence", 4)] - row[("composite_ambivalence", 1)],
            "delta_conflict": row[("decisional_conflict", 4)] - row[("decisional_conflict", 1)],
            "delta_vacillation": row[("conversational_vacillation", 4)] - row[("conversational_vacillation", 1)],
            "s4_composite": row[("composite_ambivalence", 4)],
            "s4_conflict": row[("decisional_conflict", 4)],
            "s4_vacillation": row[("conversational_vacillation", 4)],
        })
    df_delta = pd.DataFrame(delta_rows)

    # 2. Compute discriminative gap at S4 per patient
    piv_s4 = df[df["session"] == 4].pivot_table(
        index=["model", "patient_id"],
        columns="competence",
        values=["composite_ambivalence", "decisional_conflict", "conversational_vacillation"],
    )

    gap_rows = []
    for (model, pid), row in piv_s4.iterrows():
        gap_rows.append({
            "model": model,
            "patient_id": pid,
            "gap_composite": row[("composite_ambivalence", "Highly_Inconsistent")] - row[("composite_ambivalence", "Good")],
            "gap_conflict": row[("decisional_conflict", "Highly_Inconsistent")] - row[("decisional_conflict", "Good")],
            "gap_vacillation": row[("conversational_vacillation", "Highly_Inconsistent")] - row[("conversational_vacillation", "Good")],
        })
    df_gap = pd.DataFrame(gap_rows)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(19, 12),
        constrained_layout=False,
    )
    plt.subplots_adjust(top=0.90, bottom=0.09, hspace=0.35, wspace=0.22, left=0.07, right=0.97)

    metrics_top = [
        ("delta_composite", r"$\Delta$ Composite Ambivalence (S4 - S1)", "Change in Score (Negative = Resolution)"),
        ("delta_conflict",  r"$\Delta$ Decisional Conflict (S4 - S1)",  "Change in Score (Negative = Resolution)"),
        ("delta_vacillation", r"$\Delta$ Conversational Vacillation (S4 - S1)", "Change in Score (Negative = Resolution)"),
    ]

    x_indices = np.arange(len(MODEL_ORDER))
    width = 0.25

    # ── Top Row: Progress Deltas ──
    for col_idx, (m_col, m_title, m_ylab) in enumerate(metrics_top):
        ax = axes[0, col_idx]
        ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6)

        for comp_idx, (comp_key, cfg) in enumerate(COMPETENCE_CONFIG.items()):
            sub_c = df_delta[df_delta["competence"] == comp_key]
            means = [sub_c[sub_c["model"] == m][m_col].mean() for m in MODEL_ORDER]
            sems = [sub_c[sub_c["model"] == m][m_col].sem() * 1.96 for m in MODEL_ORDER]

            pos = x_indices + (comp_idx - 1) * width
            bars = ax.bar(
                pos, means, width,
                yerr=sems, capsize=4,
                color=cfg["color"], alpha=0.88,
                edgecolor="black", linewidth=0.8,
                label=cfg["label"] if col_idx == 0 else None,
                zorder=3,
            )

        ax.set_title(m_title, fontsize=21, pad=10)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER], fontsize=19)
        ax.set_ylabel(m_ylab if col_idx == 0 else "", fontsize=19)

    # ── Bottom Row: Discriminative Separation Gap at Session 4 ──
    metrics_bottom = [
        ("gap_composite", "Session 4 Discriminative Gap\n(Highly Inconsistent - MI-Consistent)", "Score Gap (Positive = Sensitivity)"),
        ("gap_conflict", "Decisional Conflict Gap at S4\n(Highly Inconsistent - MI-Consistent)", "Score Gap (Positive = Sensitivity)"),
        ("gap_vacillation", "Vacillation Gap at S4\n(Highly Inconsistent - MI-Consistent)", "Score Gap (Positive = Sensitivity)"),
    ]

    model_colors = ["#2b5c8f", "#738595", "#b49265"]  # Distinct muted tones

    for col_idx, (m_col, m_title, m_ylab) in enumerate(metrics_bottom):
        ax = axes[1, col_idx]
        ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6)

        means = [df_gap[df_gap["model"] == m][m_col].mean() for m in MODEL_ORDER]
        sems = [df_gap[df_gap["model"] == m][m_col].sem() * 1.96 for m in MODEL_ORDER]

        bars = ax.bar(
            x_indices, means, width * 1.8,
            yerr=sems, capsize=5,
            color=model_colors, alpha=0.88,
            edgecolor="black", linewidth=0.8,
            zorder=3,
        )

        for bar, val in zip(bars, means):
            offset = 0.05 if val >= 0 else -0.15
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.2f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=16, fontweight="bold",
            )

        ax.set_title(m_title, fontsize=20, pad=10)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER], fontsize=19)
        ax.set_ylabel(m_ylab if col_idx == 0 else "", fontsize=19)

    # Legend for top row
    axes[0, 0].legend(loc="upper left", fontsize=15, framealpha=0.9)

    png_path = OUT_DIR / "fig_ambivalence_baselines_differentiation.png"
    pdf_path = OUT_DIR / "fig_ambivalence_baselines_differentiation.pdf"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")

    fig.savefig(str(AUTOMISC_OUT_DIR / "fig_ambivalence_baselines_differentiation.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(AUTOMISC_OUT_DIR / "fig_ambivalence_baselines_differentiation.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Differentiation Figure: {png_path.name} | {pdf_path.name}")


# ── Statistical Analysis & Tables ──────────────────────────────────────────────

def generate_statistical_tables(df: pd.DataFrame):
    """
    Computes rigorous summary tables (Mean +/- SD, deltas, Wilcoxon tests)
    and exports them in CSV, Markdown, and LaTeX booktabs format.
    """
    # 1. Longitudinal Session Means & SD
    agg_funcs = {
        "composite_ambivalence": ["mean", "std"],
        "problem_uncertainty": ["mean", "std"],
        "decisional_conflict": ["mean", "std"],
        "conversational_vacillation": ["mean", "std"],
    }
    grouped = df.groupby(["model", "competence", "session"]).agg(agg_funcs).reset_index()

    # Flatten column names
    grouped.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in grouped.columns
    ]

    csv_path = OUT_DIR / "ambivalence_baselines_summary_table.csv"
    grouped.to_csv(csv_path, index=False)
    grouped.to_csv(AUTOMISC_OUT_DIR / "ambivalence_baselines_summary_table.csv", index=False)
    print(f"Saved Statistical Summary: {csv_path.name}")

    # 2. Formatted Markdown & LaTeX Table
    rows = []
    for model in MODEL_ORDER:
        for comp in ["Good", "Inconsistent", "Highly_Inconsistent"]:
            sub = df[(df["model"] == model) & (df["competence"] == comp)]
            s1_comp = sub[sub["session"] == 1]["composite_ambivalence"].values
            s4_comp = sub[sub["session"] == 4]["composite_ambivalence"].values
            
            delta = s4_comp - s1_comp
            w_stat, p_val = stats.wilcoxon(s1_comp, s4_comp, alternative="two-sided") if len(s1_comp) == len(s4_comp) else (np.nan, np.nan)
            
            diff = s4_comp - s1_comp
            dz = diff.mean() / (diff.std() + 1e-9) if len(diff) > 1 else np.nan

            rows.append({
                "Model": model,
                "Competence": comp,
                "S1 Mean (SD)": f"{s1_comp.mean():.2f} ({s1_comp.std():.2f})",
                "S2 Mean (SD)": f"{sub[sub['session'] == 2]['composite_ambivalence'].mean():.2f} ({sub[sub['session'] == 2]['composite_ambivalence'].std():.2f})",
                "S3 Mean (SD)": f"{sub[sub['session'] == 3]['composite_ambivalence'].mean():.2f} ({sub[sub['session'] == 3]['composite_ambivalence'].std():.2f})",
                "S4 Mean (SD)": f"{s4_comp.mean():.2f} ({s4_comp.std():.2f})",
                "Delta (S4 - S1)": f"{delta.mean():+.2f} ({delta.std():.2f})",
                "Wilcoxon p": f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.3f}",
                "Cohen's dz": f"{dz:+.2f}",
            })

    df_table = pd.DataFrame(rows)
    md_path = OUT_DIR / "ambivalence_baselines_table.md"
    tex_path = OUT_DIR / "ambivalence_baselines_table.tex"

    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write("# Patient Ambivalence Across Architectures and MI Competence Levels\n\n")
        f_md.write(df_table.to_markdown(index=False))
        f_md.write("\n")

    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write("% LaTeX Booktabs Table: Patient Ambivalence Comparison Across Baselines\n")
        f_tex.write(df_table.to_latex(index=False, escape=False))
        f_tex.write("\n")

    print(f"Saved Markdown and LaTeX tables: {md_path.name} | {tex_path.name}")


# ── Main Runner ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Ambivalence baselines and competence levels")
    parser.add_argument(
        "--input-dir", "--transcripts-dir", "-i",
        dest="input_dir",
        type=str,
        default=str(TRANSCRIPTS_DIR),
        help=f"Directory containing transcripts evaluations (default: {TRANSCRIPTS_DIR})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        type=str,
        default=str(OUT_DIR),
        help=f"Directory to save figures and tables (default: {OUT_DIR})"
    )
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Visualizing Patient Ambivalence from {input_dir}")
    print("=" * 80)

    # 1. Ingest Data
    df = load_baselines_ambivalence_data(input_dir)
    
    # Save full clean dataset
    full_csv = out_dir / "ambivalence_baselines_comparison_data.csv"
    df.to_csv(full_csv, index=False)
    if out_dir != AUTOMISC_OUT_DIR:
        df.to_csv(AUTOMISC_OUT_DIR / "ambivalence_baselines_comparison_data.csv", index=False)
    print(f"Exported raw tidy dataset: {full_csv.name}")

    # 2. Figure 1: 1x3 Trajectory Panels
    print("\nGenerating Figure 1: 1x3 Trajectory Panels...")
    plot_ambivalence_baselines_trajectories(df)

    # 3. Figure 2: 3x3 Dimensions Multi-Panel
    print("\nGenerating Figure 2: 3x3 Dimensions Grid...")
    plot_ambivalence_baselines_multipanel(df)

    # 4. Figure 3: Progress & Differentiation
    print("\nGenerating Figure 3: Progress & Discriminative Sensitivity...")
    plot_ambivalence_progress_and_differentiation(df)

    # 5. Statistical Analysis & Tables
    print("\nGenerating Statistical Summary Tables...")
    generate_statistical_tables(df)

    print("\n" + "=" * 80)
    print("All ambivalence comparison figures and tables successfully generated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
