"""Visualization: MITI global scores multi-line chart across sessions."""
import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set visualization style
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def plot_miti_global_scores(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot MITI global scores over sessions with 4 lines (one per score)."""
    try:
        df_miti = df[df["miti_avg_score"].notna()]
        
        if len(df_miti) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Define colors for each MITI score
        colors = {
            "Cultivating Change Talk": "#1f77b4",  # Blue
            "Softening Sustain Talk": "#ff7f0e",   # Orange
            "Partnership": "#2ca02c",               # Green
            "Empathy": "#d62728"                    # Red
        }
        
        markers = {
            "Cultivating Change Talk": "o",
            "Softening Sustain Talk": "s",
            "Partnership": "^",
            "Empathy": "D"
        }
        
        # Plot each MITI score
        if df_miti["miti_cultivating_change_talk"].notna().any():
            ax.plot(df_miti["session"], df_miti["miti_cultivating_change_talk"], 
                   marker=markers["Cultivating Change Talk"], linewidth=2, markersize=8,
                   label="Cultivating Change Talk", color=colors["Cultivating Change Talk"])
        
        if df_miti["miti_softening_sustain_talk"].notna().any():
            ax.plot(df_miti["session"], df_miti["miti_softening_sustain_talk"],
                   marker=markers["Softening Sustain Talk"], linewidth=2, markersize=8,
                   label="Softening Sustain Talk", color=colors["Softening Sustain Talk"])
        
        if df_miti["miti_partnership"].notna().any():
            ax.plot(df_miti["session"], df_miti["miti_partnership"],
                   marker=markers["Partnership"], linewidth=2, markersize=8,
                   label="Partnership", color=colors["Partnership"])
        
        if df_miti["miti_empathy"].notna().any():
            ax.plot(df_miti["session"], df_miti["miti_empathy"],
                   marker=markers["Empathy"], linewidth=2, markersize=8,
                   label="Empathy", color=colors["Empathy"])
        
        ax.set_xlabel("Session Number", fontsize=12)
        ax.set_ylabel("MITI Score (1-5)", fontsize=12)
        ax.set_title("MITI Global Scores Across Sessions", fontsize=14, fontweight="bold")
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks(df_miti["session"])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        
        # Add reference line at midpoint (3)
        ax.axhline(y=3, linestyle=":", alpha=0.5, color="gray", linewidth=1)
        
        # Add average MITI score annotation
        avg_miti = df_miti["miti_avg_score"].mean()
        ax.text(0.02, 0.98, f"Overall Avg: {avg_miti:.2f}/5.0",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
        
        plt.savefig(output_dir / "07_miti_global_scores.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot MITI global scores: {e}", exc_info=True)
        plt.close()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.plot_miti_global_scores <path_to_json>")
        sys.exit(1)

    from src.analytics.analyzer import load_json, calculate_statistics, create_output_dir

    json_path = sys.argv[1]
    data = load_json(json_path)
    metadata = data.get("metadata", {})
    state = data.get("state", data)
    df = calculate_statistics(state, metadata)
    output_dir = create_output_dir(metadata, json_path)
    plot_miti_global_scores(df, output_dir)
    logger.info(f"Plot saved to {output_dir}")
