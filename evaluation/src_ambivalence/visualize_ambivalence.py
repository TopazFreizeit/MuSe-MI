#!/usr/bin/env python3
"""
Visualization and Table Generator for Patient Ambivalence Scores.

Generates publication-ready figures and appendix tables comparing therapist variations:
  1. MI-Consistent Therapist (Ideal)
  2. MI-Inconsistent Therapist (Occasional errors / resistance)
  3. Highly MI-Inconsistent Therapist (Confrontational / severe rupture)

Formatting:
  • seaborn whitegrid, Times New Roman, 300 DPI PDF + PNG
  • Okabe-Ito colorblind-safe palette
  • Cousineau-Morey correction for within-subject repeated measures 95% CIs
  • Single-panel main figure (Composite Ambivalence) + 1x3 dimensions panel
    (Problem Uncertainty, Decisional Conflict, Conversational Vacillation)
  • Appendix summary tables in CSV, Markdown, and LaTeX booktabs format
"""

import os
import sys
import glob
import yaml
import pathlib
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
sys.path.append(str(ROOT))

TRANSCRIPTS_DIR = ROOT / "transcripts"
OUT_DIR = ROOT / "src_ambivalence" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Also ensure automisc results has access if needed
AUTOMISC_OUT_DIR = ROOT / "src_automisc" / "results"
AUTOMISC_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication style (matching generate_acl_figures.py & venue-templates) ───
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 22,
    "axes.labelsize": 26,
    "axes.titlesize": 26,
    "axes.titleweight": "bold",
    "legend.fontsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "figure.titlesize": 26,
    "lines.linewidth": 3.0,
    "lines.markersize": 10,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.45,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Okabe-Ito colorblind-safe palette
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
]

VARIATIONS = ["baseline", "adversarial", "more_adversarial"]

DISPLAY_NAMES = {
    "baseline":         "MI-Consistent Therapist",
    "adversarial":      "MI-Inconsistent Therapist",
    "more_adversarial": "Highly MI-Inconsistent Therapist",
}

STYLE_MAP = {
    "baseline":         {"color": OKABE_ITO[0], "marker": "o"},  # MI-Consistent: orange circle
    "adversarial":      {"color": OKABE_ITO[1], "marker": "s"},  # MI-Inconsistent: sky-blue square
    "more_adversarial": {"color": OKABE_ITO[4], "marker": "^"},  # Highly MI-Inconsistent: blue triangle
}


# ── Data Loading & Preparation ─────────────────────────────────────────────────

def detect_variation(filename: str) -> str:
    """Classifies file into baseline, adversarial, or more_adversarial."""
    if "more_adversarial" in filename:
        return "more_adversarial"
    elif "adversarial" in filename:
        return "adversarial"
    else:
        return "baseline"

def load_ambivalence_dataframe(transcripts_dir: pathlib.Path = TRANSCRIPTS_DIR) -> pd.DataFrame:
    """
    Parses all Ambivalence YAML evaluation files and builds a structured dataframe.
    """
    records = []
    yaml_files = sorted(transcripts_dir.glob("*/*ambivalence*.yaml"))
    
    for yf in yaml_files:
        var = detect_variation(yf.name)
        # Infer patient id from directory or filename prefix
        if yf.parent.name.isdigit():
            pat_id = int(yf.parent.name)
        else:
            pat_id = int(yf.stem.split("_")[0])

        with open(yf, "r", encoding="utf-8") as fin:
            try:
                data = yaml.safe_load(fin)
            except yaml.YAMLError as e:
                print(f"Warning: skipping {yf.name} due to YAML parse error: {e}")
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

            # Extract individual item scores if available
            u_scores = eval_data.get("problem_uncertainty", {}).get("item_scores", {})
            d_scores = eval_data.get("decisional_conflict", {}).get("item_scores", {})
            v_scores = eval_data.get("conversational_vacillation", {}).get("item_scores", {})

            records.append({
                "patient_id": pat_id,
                "variation": var,
                "therapist_type": DISPLAY_NAMES.get(var, var),
                "session": s_num,
                "composite_ambivalence": comp_score,
                "problem_uncertainty": u_mean,
                "decisional_conflict": d_mean,
                "conversational_vacillation": v_mean,
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
    return df

def apply_cousineau_morey(df: pd.DataFrame, metric_cols: list) -> pd.DataFrame:
    """
    Applies Cousineau-Morey (2008) correction for repeated measures.
    Removes between-subject variance so 95% CIs reflect within-subject variance.
    """
    df_corr = df.copy()
    for metric in metric_cols:
        corr_col = f"{metric}_corr"
        if "patient_id" in df_corr.columns:
            J = df_corr.groupby("patient_id")[metric].transform('count').max()
            if J > 1:
                subj_mean = df_corr.groupby("patient_id")[metric].transform('mean')
                grand_mean = df_corr[metric].mean()
                multiplier = np.sqrt(J / (J - 1))
                df_corr[corr_col] = grand_mean + multiplier * (df_corr[metric] - subj_mean)
            else:
                df_corr[corr_col] = df_corr[metric]
        else:
            df_corr[corr_col] = df_corr[metric]
    return df_corr


# ── Figure 1: Main Composite Ambivalence Score (Single Panel) ──────────────────

def plot_ambivalence_adversarials_main(df: pd.DataFrame, out_dir: pathlib.Path = OUT_DIR):
    """
    Generates the central figure: Composite Ambivalence Score (1 to 5) across 4 sessions
    for the therapist conditions.
    """
    present_vars = [v for v in VARIATIONS if v in df["variation"].unique()]
    if not present_vars:
        present_vars = sorted(df["variation"].unique())

    sub = df[df["variation"].isin(present_vars)].copy()
    sub = apply_cousineau_morey(sub, ["composite_ambivalence"])

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    for var in present_vars:
        style = STYLE_MAP.get(var, {"color": OKABE_ITO[0], "marker": "o"})
        var_data = sub[sub["variation"] == var]
        sns.lineplot(
            data=var_data,
            x="session",
            y="composite_ambivalence_corr",
            ax=ax,
            color=style["color"],
            marker=style["marker"],
            linewidth=3.0,
            markersize=10,
            errorbar="ci",
            err_style="bars",
            err_kws={"capsize": 5, "elinewidth": 2.0},
            label=DISPLAY_NAMES.get(var, var),
        )

    ax.set_title("Patient Ambivalence Trajectory", pad=14)
    ax.set_xlabel("Session")
    ax.set_ylabel("Ambivalence Composite Score (1–5)")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(1.0, 5.0)
    ax.set_yticks([1.0, 2.0, 3.0, 4.0, 5.0])

    # Reference line at 3.0 (Moderate / Emergent Ambivalence)
    ax.axhline(3.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.5)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=1,
        frameon=False,
        fontsize=20,
        handletextpad=0.5,
    )

    png_path = out_dir / "fig_ambivalence_adversarials.png"
    pdf_path = out_dir / "fig_ambivalence_adversarials.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path.name} | {pdf_path.name}")


# ── Figure 2: Multi-Panel (Problem Uncertainty, Decisional Conflict, Vacillation)

def plot_ambivalence_adversarials_multipanel(df: pd.DataFrame, out_dir: pathlib.Path = OUT_DIR):
    """
    Generates a 1x3 publication panel showing Problem Uncertainty, Decisional Conflict,
    and Conversational Vacillation dimensions across sessions.
    """
    dimensions = ["problem_uncertainty", "decisional_conflict", "conversational_vacillation"]
    dim_titles = {
        "problem_uncertainty": "Problem Uncertainty",
        "decisional_conflict": "Decisional Conflict",
        "conversational_vacillation": "Conversational Vacillation"
    }

    present_vars = [v for v in VARIATIONS if v in df["variation"].unique()]
    if not present_vars:
        present_vars = sorted(df["variation"].unique())

    sub = df[df["variation"].isin(present_vars)].copy()
    sub = apply_cousineau_morey(sub, dimensions)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)

    for ax, metric in zip(axes, dimensions):
        corr_col = f"{metric}_corr"
        for var in present_vars:
            style = STYLE_MAP.get(var, {"color": OKABE_ITO[0], "marker": "o"})
            var_data = sub[sub["variation"] == var]
            sns.lineplot(
                data=var_data,
                x="session",
                y=corr_col,
                ax=ax,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.5,
                markersize=8,
                errorbar="ci",
                err_style="bars",
                err_kws={"capsize": 4, "elinewidth": 1.5},
                legend=False,
            )

        ax.set_title(dim_titles[metric])
        ax.set_xlabel("Session")
        ax.set_ylabel("Ambivalence Score (1–5)")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(1.0, 5.0)
        ax.set_yticks([1.0, 2.0, 3.0, 4.0, 5.0])
        ax.axhline(3.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.4)

    # Common unified legend at the bottom
    handles = [
        Line2D(
            [0], [0],
            color=STYLE_MAP.get(v, {"color": OKABE_ITO[0]})["color"],
            lw=3,
            marker=STYLE_MAP.get(v, {"marker": "o"})["marker"],
            markersize=10,
            label=DISPLAY_NAMES.get(v, v),
        )
        for v in present_vars
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(present_vars),
        frameon=False,
        fontsize=24,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    png_path = out_dir / "fig_ambivalence_adversarials_multipanel.png"
    pdf_path = out_dir / "fig_ambivalence_adversarials_multipanel.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path.name} | {pdf_path.name}")


# ── Appendix Tables (CSV, Markdown, LaTeX) ─────────────────────────────────────

def generate_ambivalence_appendix_table(df: pd.DataFrame, out_dir: pathlib.Path = OUT_DIR):
    """
    Computes Mean ± SD for Problem Uncertainty, Decisional Conflict, Conversational Vacillation,
    and Composite Ambivalence across sessions and overall. Exports to CSV, Markdown, and LaTeX.
    """
    present_vars = [v for v in VARIATIONS if v in df["variation"].unique()]
    if not present_vars:
        present_vars = sorted(df["variation"].unique())

    sub = df[df["variation"].isin(present_vars)].copy()

    rows = []
    for var in present_vars:
        var_name = DISPLAY_NAMES.get(var, var)
        var_data = sub[sub["variation"] == var]

        # By Session
        for s in [1, 2, 3, 4]:
            s_data = var_data[var_data["session"] == s]
            if len(s_data) == 0:
                continue
            row = {
                "Therapist Condition": var_name,
                "Session": f"Session {s}",
                "Problem Uncertainty (Mean ± SD)": f"{s_data['problem_uncertainty'].mean():.2f} ± {s_data['problem_uncertainty'].std():.2f}",
                "Decisional Conflict (Mean ± SD)": f"{s_data['decisional_conflict'].mean():.2f} ± {s_data['decisional_conflict'].std():.2f}",
                "Conversational Vacillation (Mean ± SD)": f"{s_data['conversational_vacillation'].mean():.2f} ± {s_data['conversational_vacillation'].std():.2f}",
                "Composite Ambivalence (Mean ± SD)": f"{s_data['composite_ambivalence'].mean():.2f} ± {s_data['composite_ambivalence'].std():.2f}",
            }
            rows.append(row)

        # Overall across all sessions
        overall_row = {
            "Therapist Condition": var_name,
            "Session": "Overall (1–4)",
            "Problem Uncertainty (Mean ± SD)": f"{var_data['problem_uncertainty'].mean():.2f} ± {var_data['problem_uncertainty'].std():.2f}",
            "Decisional Conflict (Mean ± SD)": f"{var_data['decisional_conflict'].mean():.2f} ± {var_data['decisional_conflict'].std():.2f}",
            "Conversational Vacillation (Mean ± SD)": f"{var_data['conversational_vacillation'].mean():.2f} ± {var_data['conversational_vacillation'].std():.2f}",
            "Composite Ambivalence (Mean ± SD)": f"{var_data['composite_ambivalence'].mean():.2f} ± {var_data['composite_ambivalence'].std():.2f}",
        }
        rows.append(overall_row)

    table_df = pd.DataFrame(rows)

    # 1. Save CSV
    csv_path = out_dir / "ambivalence_adversarials_table.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"  Saved Table CSV: {csv_path.name}")

    # 2. Save Markdown
    md_path = out_dir / "ambivalence_adversarials_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Patient Ambivalence Scores Across Therapist Conditions\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write("\n")
    print(f"  Saved Table Markdown: {md_path.name}")

    # 3. Save LaTeX
    latex_path = out_dir / "ambivalence_adversarials_table.tex"
    latex_code = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Therapist Condition} & \textbf{Session} & \textbf{Problem Uncertainty} & \textbf{Decisional Conflict} & \textbf{Conversational Vacillation} & \textbf{Composite Ambivalence} \\",
        r"\midrule"
    ]

    for i, r in enumerate(rows):
        is_overall = "Overall" in r["Session"]
        cond = f"\\textbf{{{r['Therapist Condition']}}}" if r["Session"] == "Session 1" else ""
        sess = f"\\textit{{{r['Session']}}}" if is_overall else r["Session"]

        # Format with proper LaTeX math \pm
        u_val = r['Problem Uncertainty (Mean ± SD)'].replace('±', r'\pm')
        d_val = r['Decisional Conflict (Mean ± SD)'].replace('±', r'\pm')
        v_val = r['Conversational Vacillation (Mean ± SD)'].replace('±', r'\pm')
        c_val = r['Composite Ambivalence (Mean ± SD)'].replace('±', r'\pm')

        uncert = f"\\textbf{{{u_val}}}" if is_overall else f"${u_val}$"
        deci = f"\\textbf{{{d_val}}}" if is_overall else f"${d_val}$"
        vac = f"\\textbf{{{v_val}}}" if is_overall else f"${v_val}$"
        comp = f"\\textbf{{{c_val}}}" if is_overall else f"${c_val}$"

        if is_overall:
            uncert = f"${uncert}$"
            deci = f"${deci}$"
            vac = f"${vac}$"
            comp = f"${comp}$"

        line = f"{cond} & {sess} & {uncert} & {deci} & {vac} & {comp} \\\\"
        latex_code.append(line)
        if is_overall and i < len(rows) - 1:
            latex_code.append(r"\midrule")

    latex_code.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Patient Ambivalence dimensions (Problem Uncertainty, Decisional Conflict, Conversational Vacillation) and composite scores (Mean $\pm$ SD; range 1--5) across four therapy sessions for MI-Consistent, MI-Inconsistent, and Highly MI-Inconsistent therapist conditions.}",
        r"\label{tab:ambivalence_adversarials}",
        r"\end{table*}"
    ])

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_code))
        f.write("\n")
    print(f"  Saved Table LaTeX: {latex_path.name}")

    # Also save copy to automisc results for paper compilation convenience
    table_df.to_csv(AUTOMISC_OUT_DIR / "ambivalence_adversarials_table.csv", index=False)
    with open(AUTOMISC_OUT_DIR / "ambivalence_adversarials_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_code))
        f.write("\n")

    return table_df


# ── Main Entrypoint ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Ambivalence evaluation data and generate appendix tables")
    parser.add_argument(
        "--input-dir", "--transcripts-dir", "-i",
        dest="input_dir",
        type=str,
        default=str(TRANSCRIPTS_DIR),
        help=f"Directory containing transcript evaluations (default: {TRANSCRIPTS_DIR})"
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

    print(f"Loading Ambivalence evaluation data from {input_dir}...")
    df = load_ambivalence_dataframe(input_dir)
    
    if df.empty:
        print(f"ERROR: No Ambivalence evaluation records found in {input_dir}.")
        sys.exit(1)

    print(f"  Loaded {len(df)} session evaluation records from {df['patient_id'].nunique()} patients across variations: {df['variation'].unique().tolist()}.")

    # Save tidy session dataset
    tidy_csv = out_dir / "ambivalence_session_data.csv"
    df.to_csv(tidy_csv, index=False)
    if out_dir != AUTOMISC_OUT_DIR:
        df.to_csv(AUTOMISC_OUT_DIR / "ambivalence_session_data.csv", index=False)
    print(f"  Saved tidy dataset: {tidy_csv.name}")

    print("\n→ Generating Figure 1: Main Composite Ambivalence...")
    plot_ambivalence_adversarials_main(df, out_dir)
    if out_dir != AUTOMISC_OUT_DIR:
        plot_ambivalence_adversarials_main(df, AUTOMISC_OUT_DIR)

    print("\n→ Generating Figure 2: Multi-Panel (Problem Uncertainty, Decisional Conflict, Vacillation)...")
    plot_ambivalence_adversarials_multipanel(df, out_dir)
    if out_dir != AUTOMISC_OUT_DIR:
        plot_ambivalence_adversarials_multipanel(df, AUTOMISC_OUT_DIR)

    print("\n→ Generating Appendix Tables...")
    table_df = generate_ambivalence_appendix_table(df, out_dir)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE (Mean ± SD):")
    print("=" * 80)
    print(table_df.to_string(index=False))
    print("=" * 80)
    print(f"\nAll Ambivalence visualizations and tables successfully generated in:\n  - {out_dir}/\n  - {AUTOMISC_OUT_DIR}/")

if __name__ == "__main__":
    main()
