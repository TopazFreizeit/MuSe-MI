"""Visualization: Speaker balance 2x2 grid (turn counts, word counts, text %, avg message length)."""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set visualization style
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def plot_speaker_balance(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot speaker balance with 4 graphs: turn counts, word counts, turn percentages, word percentages, and message lengths."""
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        width = 0.35
        x = df["session"]
        x_pos = np.arange(len(x))
        
        # Top-left: Turn counts
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(x_pos - width/2, df["therapist_turns"], width, label="Therapist", color="#1f77b4")
        ax1.bar(x_pos + width/2, df["patient_turns"], width, label="Patient", color="#2ca02c")
        ax1.set_xlabel("Session Number", fontsize=11)
        ax1.set_ylabel("Number of Turns", fontsize=11)
        ax1.set_title("Speaker Balance - Turn Counts", fontsize=12, fontweight="bold")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Top-right: Word counts
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x_pos - width/2, df["therapist_word_count"], width, label="Therapist", color="#1f77b4")
        ax2.bar(x_pos + width/2, df["patient_word_count"], width, label="Patient", color="#2ca02c")
        ax2.set_xlabel("Session Number", fontsize=11)
        ax2.set_ylabel("Number of Words", fontsize=11)
        ax2.set_title("Speaker Balance - Word Counts", fontsize=12, fontweight="bold")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Bottom-left: Percentage of text (words)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(x_pos - width/2, df["therapist_word_pct"], width, label="Therapist", color="#1f77b4")
        ax3.bar(x_pos + width/2, df["patient_word_pct"], width, label="Patient", color="#2ca02c")
        ax3.set_xlabel("Session Number", fontsize=11)
        ax3.set_ylabel("Percentage of Text (%)", fontsize=11)
        ax3.set_title("Speaker Balance - Text Percentage", fontsize=12, fontweight="bold")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(x)
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Bottom-right: Average message lengths
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df["session"], df["therapist_avg_words"], marker="o", linewidth=2, markersize=8, 
                label="Therapist", color="#1f77b4")
        ax4.plot(df["session"], df["patient_avg_words"], marker="s", linewidth=2, markersize=8, 
                label="Patient", color="#2ca02c")
        ax4.set_xlabel("Session Number", fontsize=11)
        ax4.set_ylabel("Average Words per Message", fontsize=11)
        ax4.set_title("Message Length Trends", fontsize=12, fontweight="bold")
        ax4.set_xticks(df["session"])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.savefig(output_dir / "02_speaker_balance.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot speaker balance: {e}", exc_info=True)
        plt.close()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.plot_speaker_balance <path_to_json>")
        sys.exit(1)

    from src.analytics.analyzer import load_json, calculate_statistics, create_output_dir

    json_path = sys.argv[1]
    data = load_json(json_path)
    metadata = data.get("metadata", {})
    state = data.get("state", data)
    df = calculate_statistics(state, metadata)
    output_dir = create_output_dir(metadata, json_path)
    plot_speaker_balance(df, output_dir)
    logger.info(f"Plot saved to {output_dir}")
