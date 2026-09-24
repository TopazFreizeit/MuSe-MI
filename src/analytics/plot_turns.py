"""Visualization: Turns per session line chart with trend line."""
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


def plot_turns(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot turns per session."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
    
        ax.plot(df["session"], df["turns"], marker="o", linewidth=2, markersize=8, color="#1f77b4")
        ax.set_xlabel("Session Number", fontsize=12)
        ax.set_ylabel("Number of Turns", fontsize=12)
        ax.set_title("Conversation Turns per Session", fontsize=14, fontweight="bold")
        ax.set_xticks(df["session"])
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df["session"], df["turns"], 1)
        p = np.poly1d(z)
        ax.plot(df["session"], p(df["session"]), "--", alpha=0.5, color="red", label=f"Trend: {'↑' if z[0] > 0 else '↓'}")
        ax.legend()
        
        plt.savefig(output_dir / "01_turns_per_session.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot turns: {e}", exc_info=True)
        plt.close()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.plot_turns <path_to_json>")
        sys.exit(1)

    from src.analytics.analyzer import load_json, calculate_statistics, create_output_dir

    json_path = sys.argv[1]
    data = load_json(json_path)
    metadata = data.get("metadata", {})
    state = data.get("state", data)
    df = calculate_statistics(state, metadata)
    output_dir = create_output_dir(metadata, json_path)
    plot_turns(df, output_dir)
    logger.info(f"Plot saved to {output_dir}")
