"""
This metric visualizes the patient's language type (DARN, CAT, Resistance) 
turn-by-turn within each session, based on the Clinical Analyzer's classification.
"""
import logging
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

def plot_turn_by_turn_patient_language(
    state: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    try:
        all_session_turns: list[list[dict]] = state.get("all_session_turns", [])
        if not all_session_turns:
            logger.info("No session turns — skipping turn_by_turn_patient_language plot.")
            return

        fig, axes = plt.subplots(
            nrows=len(all_session_turns), ncols=1,
            figsize=(10, max(3, 2 * len(all_session_turns))),
            sharex=False, sharey=True
        )
        if len(all_session_turns) == 1:
            axes = [axes]
        
        has_data = False
        for session_idx, turns in enumerate(all_session_turns):
            ax = axes[session_idx]
            
            x_turns = []
            y_darn = []
            y_cat = []
            y_res = []
            
            for turn in turns:
                if turn.get("speaker") == "therapist":
                    turn_num = turn.get("turn_number", 0)
                    
                    darn_val = str(turn.get("preparatory_change_talk", "none")).lower() != "none"
                    cat_val = str(turn.get("commitment_and_action_talk", "none")).lower() != "none"
                    res_val = str(turn.get("resistance_detected", "none")).lower() != "none"
                    
                    x_turns.append(turn_num)
                    y_darn.append(1 if darn_val else 0)
                    y_cat.append(1 if cat_val else 0)
                    y_res.append(1 if res_val else 0)
            
            if x_turns:
                has_data = True
                x = np.array(x_turns)
                
                # Plot Resistance
                ax.bar(x - 0.25, y_res, width=0.25, label="Resistance", color="#e74c3c")
                # Plot DARN
                ax.bar(x, y_darn, width=0.25, label="DARN (Prep CT)", color="#f39c12")
                # Plot CAT
                ax.bar(x + 0.25, y_cat, width=0.25, label="CAT (Mobilizing CT)", color="#27ae60")
                
            ax.set_title(f"Session {session_idx + 1}")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Absent", "Present"])
            ax.set_ylabel("")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            
            if session_idx == 0:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if not has_data:
            plt.close()
            return
            
        axes[-1].set_xlabel("Turn Number")
        fig.suptitle("Turn-by-Turn Patient Language (Analyzer Classification)", y=1.02, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        out_path = output_dir / "34_turn_by_turn_patient_language.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    except Exception as e:
        logger.exception(f"Failed to plot turn_by_turn_patient_language: {e}")
