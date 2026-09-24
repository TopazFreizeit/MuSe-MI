"""
Statistical analysis and visualization for therapy session data.
Minimal, efficient implementation following data science best practices.
"""
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter
import shutil

import pandas as pd
import numpy as np


from src.analytics.plot_turns import plot_turns
from src.analytics.plot_speaker_balance import plot_speaker_balance
from src.analytics.plot_memory_metrics import plot_memory_metrics
from src.analytics.plot_miin_injection_code_distribution import plot_miin_injection_code_distribution
from src.analytics.plot_longitudinal_therapeutic_process import plot_longitudinal_therapeutic_process
from src.analytics.plot_technique_distribution import plot_technique_distribution
from src.analytics.plot_threads_addressed import plot_threads_addressed
from src.analytics.plot_micro_skill_distribution import plot_micro_skill_distribution
from src.analytics.plot_patient_state_progression import plot_patient_state_progression
from src.analytics.plot_response_mode_technique import plot_response_mode_technique
from src.analytics.plot_latent_self_efficacy import plot_latent_self_efficacy
from src.analytics.plot_latent_anger import plot_latent_anger
from src.analytics.plot_latent_problem_recognition import plot_latent_problem_recognition
from src.analytics.plot_latent_motivational_readiness import plot_latent_motivational_readiness
from src.analytics.plot_patient_language_type_distribution import plot_patient_language_type_distribution
from src.analytics.plot_patient_darn_cat_code_distribution import plot_patient_darn_cat_code_distribution
from src.analytics.plot_therapist_stage_classification import plot_therapist_stage_classification
from src.analytics.plot_therapist_mi_phase import plot_therapist_mi_phase
from src.analytics.plot_target_category import plot_target_category
from src.analytics.plot_receptivity import plot_receptivity
from src.analytics.plot_turn_by_turn_patient_language import plot_turn_by_turn_patient_language

logger = logging.getLogger(__name__)

# MI-Inconsistent (MIIN) codes used in injection turns
MIIN_CODES = ["ADWP", "CON", "DIR", "RCWP", "WA"]


def analyze_run(json_path: str) -> None:
    """
    Complete analysis pipeline: load, calculate, export, visualize.
    
    Args:
        json_path: Path to saved state JSON file
    """
    try:
        logger.info(f"Starting analysis: {json_path}")
        
        # Load data
        data = load_json(json_path)
        metadata = data.get("metadata", {})
        state = data.get("state", data)  # Handle both formats
        
        # Calculate statistics
        df = calculate_statistics(state, metadata)
        
        # Create output directory
        output_dir = create_output_dir(state, json_path)
        
        # Export statistics
        _export_statistics(df, metadata, output_dir)
        
        # Generate visualizations
        _generate_visualizations(df, metadata, output_dir, state)
        
        # Print summary
        _log_summary(df, metadata, output_dir)
        
        logger.info(f"Analysis complete: {output_dir}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}", exc_info=True)
        raise


def create_output_dir(state: Dict[str, Any], json_path: str) -> Path:
    """Create output directory for this analysis run."""
    trace_id = state.get("therapy_program_trace_id")
    if trace_id:
        dirname = str(trace_id)
    else:
        # Fallback: use input filename
        dirname = Path(json_path).stem
    
    output_dir = Path("outputs") / dirname
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


    """

    
    Args:
        turns: List of TurnRecord dicts for one session
        speaker: "therapist" or "patient"
        
    Returns:
        Counter mapping T2 code strings to occurrence counts
    """
    codes: list[str] = []
    for turn in turns:
        if turn.get("speaker") != speaker:
            continue
        for utt in turn.get("utterances", []):
            t2 = utt.get("t2_label")
            if t2:
                codes.append(t2)
    return Counter(codes)


def calculate_statistics(state: Dict[str, Any], metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate per-session statistics from MultiSessionState.
    
    Returns:
        DataFrame with one row per session containing all metrics
    """
    try:
        session_turns = state.get("all_session_turns", [])
        
        sessions_data = []
    
        for session_num in range(1, len(session_turns) + 1):
            turns = session_turns[session_num - 1]
            
            # Extract full turn text from the 'volley' field
            def _get_turn_text(turn: Dict[str, Any]) -> str:
                return turn.get("volley", "")
            
            therapist_messages = [_get_turn_text(turn) for turn in turns if turn.get("speaker") == "therapist"]
            patient_messages = [_get_turn_text(turn) for turn in turns if turn.get("speaker") == "patient"]
            
            # Word counts (filter out empty messages)
            therapist_messages = [msg for msg in therapist_messages if msg.strip()]
            patient_messages = [msg for msg in patient_messages if msg.strip()]
            therapist_words = [len(msg.split()) for msg in therapist_messages]
            patient_words = [len(msg.split()) for msg in patient_messages]
            
            total_words = sum(therapist_words) + sum(patient_words)
            
            # Basic metrics
            session_metrics = {
                "session": session_num,
                "turns": len(therapist_messages) + len(patient_messages),
                "therapist_turns": len(therapist_messages),
                "patient_turns": len(patient_messages),
                "therapist_avg_words": np.mean(therapist_words) if therapist_words else 0,
                "patient_avg_words": np.mean(patient_words) if patient_words else 0,
                "therapist_turn_pct": len(therapist_messages) / (len(therapist_messages) + len(patient_messages)) * 100 if therapist_messages or patient_messages else 0,
                "patient_turn_pct": len(patient_messages) / (len(therapist_messages) + len(patient_messages)) * 100 if therapist_messages or patient_messages else 0,
                "therapist_word_count": sum(therapist_words),
                "patient_word_count": sum(patient_words),
                "therapist_word_pct": sum(therapist_words) / total_words * 100 if total_words > 0 else 0,
                "patient_word_pct": sum(patient_words) / total_words * 100 if total_words > 0 else 0,
            }
            
            # Longitudinal therapeutic process metrics from memory
            _memory_meta = metadata.get("memory", {})
            if _memory_meta.get("enabled", False):
                import json as _json
                # Therapist memory: stage of change and agenda
                t_mems = state.get("all_therapist_memories", [])
                if session_num <= len(t_mems) and t_mems[session_num - 1]:
                    try:
                        t_mem = _json.loads(t_mems[session_num - 1])
                        session_metrics["therapist_stage_of_change"] = t_mem.get("Current_Stage_of_Change", "")
                        session_metrics["therapist_agenda_count"] = len(t_mem.get("Therapist_Agenda_For_Next_Session", []))
                    except (ValueError, TypeError):
                        session_metrics["therapist_stage_of_change"] = ""
                        session_metrics["therapist_agenda_count"] = 0
                else:
                    session_metrics["therapist_stage_of_change"] = ""
                    session_metrics["therapist_agenda_count"] = 0

                # Patient memory: psychological state
                p_mems = state.get("all_patient_memories", [])
                if session_num <= len(p_mems) and p_mems[session_num - 1]:
                    try:
                        p_mem = _json.loads(p_mems[session_num - 1])
                        psych = (p_mem.get("Dynamic_Cognitive_State") or {}).get("Psychological_State", {})
                        session_metrics["patient_ttm_stage"] = psych.get("Current_TTM_Stage", "")
                        session_metrics["patient_resistance_level"] = psych.get("Resistance_Level", "")
                        session_metrics["patient_emotional_state"] = psych.get("Current_Emotional_State", "")
                    except (ValueError, TypeError):
                        session_metrics["patient_ttm_stage"] = ""
                        session_metrics["patient_resistance_level"] = ""
                        session_metrics["patient_emotional_state"] = ""
                else:
                    session_metrics["patient_ttm_stage"] = ""
                    session_metrics["patient_resistance_level"] = ""
                    session_metrics["patient_emotional_state"] = ""

            sessions_data.append(session_metrics)
        
        return pd.DataFrame(sessions_data)
    except Exception as e:
        logger.error(f"Failed to calculate statistics: {e}", exc_info=True)
        raise


def _export_statistics(df: pd.DataFrame, metadata: Dict[str, Any], output_dir: Path) -> None:
    """Export session metrics and aggregate statistics to CSV."""
    try:
        # Save session-level metrics
        df.to_csv(output_dir / "session_metrics.csv", index=False)
    
        # Calculate and save aggregate statistics
        stats = {
            "total_sessions": len(df),
            "avg_turns": df["turns"].mean(),
            "std_turns": df["turns"].std(),
            "min_turns": df["turns"].min(),
            "max_turns": df["turns"].max(),
            "avg_therapist_words": df["therapist_avg_words"].mean(),
            "avg_patient_words": df["patient_avg_words"].mean(),
        }
        
        # Add questionnaire stats if available (check column exists)
        if "avg_questionnaire_score" in df.columns and df["avg_questionnaire_score"].notna().any():
            stats["overall_avg_score"] = df["avg_questionnaire_score"].mean()
            stats["score_std"] = df["avg_questionnaire_score"].std()
            stats["score_min"] = df["avg_questionnaire_score"].min()
            stats["score_max"] = df["avg_questionnaire_score"].max()
        




        # Add MIIN injection code distribution stats if available
        if "miin_injected_total" in df.columns and df["miin_injected_total"].sum() > 0:
            stats["miin_injected_total"] = int(df["miin_injected_total"].sum())
            for code in MIIN_CODES:
                col = f"miin_injected_{code}"
                if col in df.columns:
                    stats[f"miin_injected_total_{code}"] = int(df[col].sum())
            stats["miin_injected_avg_per_session"] = df["miin_injected_total"].mean()

        # Save aggregate stats
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(output_dir / "statistics.csv", index=False)
        
        logger.info(f"Statistics exported to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to export statistics: {e}", exc_info=True)
        raise


def _generate_visualizations(df: pd.DataFrame, metadata: Dict[str, Any], output_dir: Path, state: Dict[str, Any] | None = None) -> None:
    """Generate all visualization plots as PNG files."""
    try:
        plot_turns(df, output_dir)
        plot_speaker_balance(df, output_dir)
        
        # Memory fidelity metrics (M1-BFRR, M2-CTPR, M3-TFR)
        memory_meta = metadata.get("memory", {})
        if memory_meta.get("enabled") and state:
            plot_memory_metrics(state, metadata, output_dir)





        # MIIN injection technique_used distribution
        miin_meta = metadata.get("miin_injection", {})
        if miin_meta.get("enabled", False) and state:
            plot_miin_injection_code_distribution(state, metadata, output_dir)

        # Longitudinal therapeutic process (therapist stage/agenda + patient psychological state)
        memory_meta = metadata.get("memory", {})
        if memory_meta.get("enabled") and state:
            plot_longitudinal_therapeutic_process(state, metadata, output_dir)

        # New fields: technique distribution, threads addressed (M5)
        if state:
            plot_technique_distribution(state, output_dir)
            plot_threads_addressed(state, output_dir)

        # Micro-skill anchor distribution (FA / FI / EC / ST per session)
        if state:
            plot_micro_skill_distribution(state, output_dir)

        # Patient state classification progression
        if state:
            plot_patient_state_progression(state, output_dir)

        # Response mode & technique timeline
        if state:
            plot_response_mode_technique(state, output_dir)

        # Latent variable intra-session trajectories (plots 29-32)
        if state and state.get("all_latent_variable_logs"):
            plot_latent_self_efficacy(state, output_dir)
            plot_latent_anger(state, output_dir)
            plot_latent_problem_recognition(state, output_dir)
            plot_latent_motivational_readiness(state, output_dir)

        # Patient language type distribution (plot 33)
        if state:
            plot_patient_language_type_distribution(state, metadata, output_dir)

        # Patient DARN-CAT planner code distribution (plot 34)
        if state:
            plot_patient_darn_cat_code_distribution(state, metadata, output_dir)

        # Therapist TTM stage-of-change classification per turn per session (plot 35)
        if state:
            plot_therapist_stage_classification(state, output_dir)

        # Therapist MI Phase classification per turn per session (plot 36)
        if state:
            plot_therapist_mi_phase(state, output_dir)

        # Plot dynamic target_category generation (plot 38)
        if state:
            plot_target_category(state, output_dir)

        # Plot dynamic receptivity (plot 39)
        if state:
            plot_receptivity(state, output_dir)

        # New 3-node architecture plots (plots 40-42)
        if state:
            plot_turn_by_turn_patient_language(state, metadata, output_dir)

        # Run Phoenix analytics visualizations if trace_id is present
        trace_id = state.get("therapy_program_trace_id")
        if trace_id:
            logger.info(f"Running Phoenix Analytics visualizations for {trace_id}...")
            
            project_root = Path(__file__).resolve().parent.parent.parent
            phoenix_analytics_dir = project_root / "src" / "phoenix_analytics"
            
            macro_script = phoenix_analytics_dir / "plot_macro_shift_distribution.py"
            if macro_script.exists():
                subprocess.run([
                    sys.executable, str(macro_script),
                    "--trace-id", str(trace_id),
                    "--output-dir", str(output_dir)
                ])
                
            semantic_script = phoenix_analytics_dir / "plot_semantic_impact_timeline.py"
            if semantic_script.exists():
                subprocess.run([
                    sys.executable, str(semantic_script),
                    "--trace-id", str(trace_id),
                    "--output-dir", str(output_dir)
                ])

        logger.info(f"Visualizations generated in {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}", exc_info=True)
        raise

def _log_summary(df: pd.DataFrame, metadata: Dict[str, Any], output_dir: Path) -> None:
    """Log analysis summary."""
    sessions = len(df)
    avg_turns = df['turns'].mean()
    
    logger.info(f"Sessions: {sessions} | Avg turns: {avg_turns:.1f}")
    
    if metadata:
        # Log model info for each module
        patient_meta = metadata.get("patient", {})
        therapist_meta = metadata.get("therapist", {})
        logger.info(f"Patient: {patient_meta.get('provider', 'unknown')}/{patient_meta.get('model', 'unknown')} (T={patient_meta.get('temperature', 'N/A')})")
        logger.info(f"Therapist: {therapist_meta.get('provider', 'unknown')}/{therapist_meta.get('model', 'unknown')} (T={therapist_meta.get('temperature', 'N/A')})")
    







