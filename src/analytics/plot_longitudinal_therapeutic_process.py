"""Visualization: Longitudinal therapeutic process from therapist and patient perspectives.

Produces two separate plots using data from the session memory system:

Plot 27a — TTM Stage Progression (dual perspective)
    Compares the therapist's ``Current_Stage_of_Change`` assessment against the
    patient's ``Current_TTM_Stage`` self-report.  Both should trend upward across
    sessions (Precontemplation -> Contemplation -> Action).
    Divergence between perspectives may indicate misalignment in clinical perception.
    Per-session clinical reasoning is saved to a companion text file
    (``27a_ttm_stage_reasoning.txt``).

Plot 27b — Patient Resistance & Emotional State
    Tracks the patient's ``Resistance_Level`` (High / Medium / Low) over sessions.
    A declining trajectory is the expected therapeutic outcome.  Includes a formatted
    table showing the patient's emotional state narrative per session.
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

# Ordinal encodings
_TTM_STAGES = ["Precontemplation", "Contemplation", "Action"]
_TTM_ENCODE = {s: i + 1 for i, s in enumerate(_TTM_STAGES)}

_RESISTANCE_LEVELS = ["Low", "Medium", "High"]
_RESISTANCE_ENCODE = {"Low": 1, "Medium": 2, "High": 3}

# Colours
_THERAPIST_COLOR = "#2980b9"
_PATIENT_COLOR = "#e67e22"
_RESISTANCE_COLOR = "#c0392b"


def _parse_memories(raw_list: List[str]) -> List[Optional[Dict[str, Any]]]:
    """Parse a list of JSON strings into dicts."""
    parsed: list[Optional[dict]] = []
    for i, s in enumerate(raw_list):
        if not s:
            parsed.append(None)
            continue
        try:
            parsed.append(json.loads(s))
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Could not parse memory for session {i + 1}")
            parsed.append(None)
    return parsed


def _extract_therapist_data(memories: List[Optional[Dict]]) -> Dict[str, list]:
    """Extract stage-of-change, reasoning, and agenda items from therapist memories."""
    stages: list[Optional[str]] = []
    reasoning: list[Optional[str]] = []
    agendas: list[list[str]] = []
    for mem in memories:
        if mem is None:
            stages.append(None)
            reasoning.append(None)
            agendas.append([])
        else:
            stages.append(mem.get("Current_Stage_of_Change"))
            reasoning.append(mem.get("Clinical_Reasoning_Process"))
            agendas.append(mem.get("Therapist_Agenda_For_Next_Session", []))
    return {"stages": stages, "reasoning": reasoning, "agendas": agendas}


def _extract_patient_data(memories: List[Optional[Dict]]) -> Dict[str, list]:
    """Extract TTM stage, reasoning, resistance, and emotional state from patient memories."""
    stages: list[Optional[str]] = []
    reasoning: list[Optional[str]] = []
    resistance: list[Optional[str]] = []
    emotional: list[Optional[str]] = []
    for mem in memories:
        if mem is None:
            stages.append(None)
            reasoning.append(None)
            resistance.append(None)
            emotional.append(None)
        else:
            reasoning.append(mem.get("Reasoning_Process"))
            psych = (mem.get("Dynamic_Cognitive_State") or {}).get("Psychological_State", {})
            stages.append(psych.get("Current_TTM_Stage"))
            resistance.append(psych.get("Resistance_Level"))
            emotional.append(psych.get("Current_Emotional_State"))
    return {"stages": stages, "reasoning": reasoning, "resistance": resistance, "emotional": emotional}


def plot_longitudinal_therapeutic_process(
    state: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot longitudinal therapeutic process as two separate visualizations.

    Produces:
        27a_ttm_stage_progression.png
        27a_ttm_stage_reasoning.txt
        27b_patient_resistance_emotional_state.png
    """
    try:
        t_raw: list[str] = state.get("all_therapist_memories", [])
        p_raw: list[str] = state.get("all_patient_memories", [])

        if not t_raw and not p_raw:
            logger.info("No memory data — skipping longitudinal therapeutic process plots.")
            return

        t_memories = _parse_memories(t_raw)
        p_memories = _parse_memories(p_raw)
        n_sessions = max(len(t_memories), len(p_memories))

        if n_sessions == 0:
            logger.info("No parseable memories — skipping longitudinal therapeutic process plots.")
            return

        t_data = _extract_therapist_data(t_memories)
        p_data = _extract_patient_data(p_memories)

        sessions = np.arange(1, n_sessions + 1)
        session_labels = [f"S{i}" for i in sessions]

        # Encode ordinal values
        t_stage_vals = [_TTM_ENCODE.get(s) if s else None for s in t_data["stages"][:n_sessions]]
        p_stage_vals = [_TTM_ENCODE.get(s) if s else None for s in p_data["stages"][:n_sessions]]
        resistance_vals = [_RESISTANCE_ENCODE.get(r) if r else None for r in p_data["resistance"][:n_sessions]]

        # ── Plot 27a: TTM Stage Progression ──────────────────────────
        _plot_ttm_stage_progression(
            sessions, session_labels, n_sessions,
            t_stage_vals, p_stage_vals, t_data, p_data, output_dir,
        )

        # ── Plot 27b: Patient Resistance & Emotional State ───────────
        _plot_resistance_emotional_state(
            sessions, session_labels, n_sessions,
            resistance_vals, p_data, output_dir,
        )

    except Exception as e:
        logger.error(f"Failed to plot longitudinal therapeutic process: {e}", exc_info=True)
        plt.close("all")


def _plot_ttm_stage_progression(
    sessions, session_labels, n_sessions,
    t_stage_vals, p_stage_vals, t_data, p_data, output_dir,
):
    """Plot 27a — TTM Stage Progression.

    Saves a compact chart and a companion text file with per-session
    clinical reasoning detail.
    """
    fig, ax = plt.subplots(figsize=(max(8, n_sessions * 1.6), 5))

    # Plot therapist perspective
    t_x = [sessions[i] for i in range(n_sessions) if t_stage_vals[i] is not None]
    t_y = [v for v in t_stage_vals if v is not None]
    if t_x:
        ax.plot(t_x, t_y, "o-", color=_THERAPIST_COLOR, linewidth=2.5,
                markersize=10, label="Therapist Assessment", zorder=3)
        for x, y in zip(t_x, t_y):
            ax.annotate(
                _TTM_STAGES[y - 1], (x, y), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=8, color=_THERAPIST_COLOR,
                fontweight="bold",
            )

    # Plot patient perspective
    p_x = [sessions[i] for i in range(n_sessions) if p_stage_vals[i] is not None]
    p_y = [v for v in p_stage_vals if v is not None]
    if p_x:
        ax.plot(p_x, p_y, "s--", color=_PATIENT_COLOR, linewidth=2.5,
                markersize=10, label="Patient Self-Report", zorder=3)
        for x, y in zip(p_x, p_y):
            ax.annotate(
                _TTM_STAGES[y - 1], (x, y), textcoords="offset points",
                xytext=(0, -18), ha="center", fontsize=8, color=_PATIENT_COLOR,
                fontweight="bold",
            )

    ax.set_yticks(range(1, 4))
    ax.set_yticklabels(_TTM_STAGES, fontsize=9)
    ax.set_ylim(0.5, 3.5)
    ax.set_xticks(sessions)
    ax.set_xticklabels(session_labels, fontsize=11)
    ax.set_xlabel("Session", fontsize=11)
    ax.set_ylabel("TTM Stage", fontsize=11)
    ax.set_title("Stage of Change — Therapist vs Patient Perspective",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    # Highlight divergence sessions
    for i in range(n_sessions):
        if (t_stage_vals[i] is not None and p_stage_vals[i] is not None
                and t_stage_vals[i] != p_stage_vals[i]):
            ax.axvspan(sessions[i] - 0.3, sessions[i] + 0.3,
                       alpha=0.1, color="red", zorder=0)

    fig.tight_layout()
    path = output_dir / "27a_ttm_stage_progression.png"
    plt.savefig(path)
    plt.close()
    logger.info(f"TTM stage progression plot saved to {path}")

    # Write per-session reasoning to a companion text file
    detail_lines: list[str] = []
    for i in range(n_sessions):
        t_stage = t_data["stages"][i] if i < len(t_data["stages"]) and t_data["stages"][i] else "N/A"
        p_stage = p_data["stages"][i] if i < len(p_data["stages"]) and p_data["stages"][i] else "N/A"
        t_reason = t_data["reasoning"][i] if i < len(t_data["reasoning"]) and t_data["reasoning"][i] else "—"
        p_reason = p_data["reasoning"][i] if i < len(p_data["reasoning"]) and p_data["reasoning"][i] else "—"
        match = "✓" if t_stage == p_stage else "✗"
        detail_lines.append(
            f"S{i+1} [{match}]  Therapist ({t_stage}) → Patient ({p_stage})\n"
            f"    Therapist: {t_reason}\n"
            f"    Patient:   {p_reason}"
        )

    if detail_lines:
        txt_path = output_dir / "27a_ttm_stage_reasoning.txt"
        txt_path.write_text(
            "Per-Session Clinical Reasoning\n"
            + "─" * 80 + "\n\n"
            + "\n\n".join(detail_lines) + "\n",
            encoding="utf-8",
        )
        logger.info(f"TTM stage reasoning details saved to {txt_path}")


def _plot_resistance_emotional_state(
    sessions, session_labels, n_sessions,
    resistance_vals, p_data, output_dir,
):
    """Plot 27b — Patient Resistance Level with emotional state detail table."""
    fig, ax = plt.subplots(figsize=(max(10, n_sessions * 2.5), 7))

    r_x = [sessions[i] for i in range(n_sessions) if resistance_vals[i] is not None]
    r_y = [resistance_vals[i] for i in range(n_sessions) if resistance_vals[i] is not None]

    if r_x:
        ax.fill_between(r_x, r_y, alpha=0.2, color=_RESISTANCE_COLOR, step="mid")
        ax.step(r_x, r_y, where="mid", color=_RESISTANCE_COLOR, linewidth=2.5, zorder=3)
        ax.plot(r_x, r_y, "o", color=_RESISTANCE_COLOR, markersize=9, zorder=4)

        for xi, yi in zip(r_x, r_y):
            ax.annotate(
                _RESISTANCE_LEVELS[yi - 1], (xi, yi),
                textcoords="offset points", xytext=(12, 0),
                ha="left", va="center", fontsize=9, fontweight="bold",
                color=_RESISTANCE_COLOR,
            )

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(_RESISTANCE_LEVELS, fontsize=9)
    ax.set_ylim(0.5, 3.5)
    ax.set_xticks(sessions)
    ax.set_xticklabels(session_labels, fontsize=11)
    ax.set_xlabel("Session", fontsize=11)
    ax.set_ylabel("Resistance Level", fontsize=11)
    ax.set_title("Patient Resistance & Emotional State",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Emotional state detail table below chart
    emo_lines: list[str] = []
    for i in range(n_sessions):
        emo = p_data["emotional"][i] if i < len(p_data["emotional"]) else None
        res = p_data["resistance"][i] if i < len(p_data["resistance"]) else None
        res_str = res if res else "N/A"
        if emo:
            emo_lines.append(f"  S{i+1}  [{res_str:>6s}]  {emo}")
        else:
            emo_lines.append(f"  S{i+1}  [{res_str:>6s}]  —")

    if emo_lines:
        header = "Session  Resist.   Emotional State"
        table_text = header + "\n" + "─" * 90 + "\n" + "\n".join(emo_lines)
        fig.text(
            0.08, -0.02, table_text,
            fontsize=7, fontfamily="monospace", color="#333",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdf2e9", alpha=0.85, edgecolor="#e5c6a0"),
        )

    path = output_dir / "27b_patient_resistance_emotional_state.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Patient resistance & emotional state plot saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.plot_longitudinal_therapeutic_process <path_to_json>")
        sys.exit(1)

    from src.analytics.analyzer import load_json, create_output_dir

    json_path = sys.argv[1]
    data = load_json(json_path)
    metadata = data.get("metadata", {})
    state = data.get("state", data)
    output_dir = create_output_dir(metadata, json_path)
    plot_longitudinal_therapeutic_process(state, metadata, output_dir)
    logger.info(f"Plots saved to {output_dir}")
