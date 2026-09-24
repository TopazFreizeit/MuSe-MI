"""
Graph configuration settings.
"""
import os
from typing import TypedDict, List, Optional, Literal, Dict, Any
from langchain_core.messages import BaseMessage


class MITIScoreResult(TypedDict):
    """Result for a single MITI global score evaluation.
    
    Attributes:
        score: Integer score from 1-5 on the Likert scale
        reasoning_steps: List of reasoning steps explaining the score assessment
    """
    score: int
    reasoning_steps: List[str]


class UtteranceRecord(TypedDict, total=False):
    """
    Record of a single parsed utterance within a volley.
    
    Populated in two stages:
    - Parser node sets `text`
    - Annotator node sets `t1_label`, `t1_explanation`, `t2_label`, `t2_explanation`
    
    Attributes:
        text: The utterance text segment parsed from the volley
        t1_explanation: Reasoning for the Tier-1 classification
        t2_explanation: Reasoning for the Tier-2 classification
    """
    text: str
    t1_explanation: str
    t2_explanation: str


class TurnRecord(TypedDict, total=False):
    """
    Record of a single conversational turn in a therapy session.

    Attributes:
        turn_number: Sequential turn number within the session (1-indexed)
        speaker: Who spoke - "therapist" or "patient"
        volley: The full text of what was said in this turn
        utterances: List of parsed and annotated utterance segments from the volley
        miin_injection: True if this turn had the MIIN persona override injected (therapist turns only; infrastructure field)
        patient_state_classification: TTM stage of change classified by the clinical analyzer (Precontemplation/Contemplation/Action); therapist turns only
        resistance_detected: Verbatim sustain talk from the patient, or 'none'; therapist turns only
        preparatory_change_talk: Verbatim DARN talk from the patient, or 'none'; therapist turns only
        commitment_and_action_talk: Verbatim commitment/action language, or 'none'; therapist turns only
        classification_reasoning: Analyzer reasoning trace (therapist turns only)
        strategic_reasoning: Strategist reasoning trace for technique selection (therapist turns only)
        macro_intent: Strategist's macro intent for the turn (therapist turns only)
        allowed_techniques: Strategist's constrained allowed techniques (therapist turns only)
        technique: MET technique code executed by the formulator (therapist turns only)
        current_mi_phase: Active MI phase determined by strategist (therapist turns only)
        formulator_reasoning: Formulator reasoning chain (therapist turns only)
        current_dominant_stage: Categorical stage based on motivational readiness (patient turns only). One of "Precontemplation"|"Contemplation"|"Action". Used to ground factual content.
        emerging_stage: Currently unused legacy field.
    """
    turn_number: int
    speaker: Literal["therapist", "patient"]
    volley: str
    utterances: List[UtteranceRecord]
    miin_injection: bool
    patient_state_classification: str
    resistance_detected: str
    preparatory_change_talk: str
    commitment_and_action_talk: str
    classification_reasoning: str
    strategic_reasoning: List[str]
    macro_intent: str
    allowed_techniques: List[str]
    technique: str
    current_mi_phase: str
    formulator_reasoning: List
    current_dominant_stage: str
    emerging_stage: str
    simpatient_miti_codes: List[str]
    simpatient_miti_reasoning: str
    simpatient_internal_state: dict
    simpatient_non_verbal_cue: int
    patient_latent_variables: Optional[dict]


class LatentTrajectoryEntry(TypedDict, total=False):
    """
    Chronological trajectory record for patient latent variables across turns and sessions.

    Attributes:
        step_index: Monotonically increasing index across the longitudinal run (0-based)
        component: Agent/module responsible for the state or transition
                   ("initial_state" | "cognitive_interpreter" | "patient" | "memory" | "bse")
        session_number: Session number for intra-session turns (1-indexed)
        turn_number: Turn number within session (0-indexed or 1-indexed)
        session_before: Preceding session number for inter-session transitions
        session_after: Succeeding session number for inter-session transitions
        anger: Anger score on 1-100 scale
        self_efficacy: Self-efficacy score on 1-100 scale
        problem_recognition: Problem recognition score on 1-100 scale
        motivational_readiness: Motivational readiness score on 1-100 scale
        details: Optional contextual metadata (e.g. semantic impacts, event titles, macro shifts)
    """
    step_index: int
    component: Literal["initial_state", "cognitive_interpreter", "patient", "memory", "bse"]
    session_number: Optional[int]
    turn_number: Optional[int]
    session_before: Optional[int]
    session_after: Optional[int]
    anger: float
    self_efficacy: float
    problem_recognition: float
    motivational_readiness: float
    details: Optional[Dict[str, Any]]


class ConversationState(TypedDict, total=False):
    """
    State object for a single therapy session.

    Attributes:
        messages: Full conversation history
        current_speaker: Current speaker - "therapist" or "patient"
        session_ended: Whether the session has ended
        session_end_reason: Reason code (1 or 2) if session ended
        turn_count: Number of conversation turns for safety limit
        patient_system_prompt: System prompt used for patient in this session
        session_number: Current session number (1-4) for logging purposes
        current_session_turns: List of TurnRecord objects containing turn metadata, parsed utterances
        miin_injection_turn_numbers: Sorted list of pre-determined sequential turn numbers for MIIN injection in this session
        miin_injection_active: Whether this session is designated for MIIN injection (infrastructure field)
        last_used_techniques: Ordered list of all MET technique codes used by the therapist
        patient_latent_variables: Current latent variable scores as a dict.
        latent_variable_logs: Ordered list of latent variable score snapshots within this session. Flat dict with 1-100 values.
        session_latent_trajectory: Full turn-by-turn chronological latent trajectory within this session.
    """
    messages: List[BaseMessage]
    current_speaker: str
    session_ended: bool
    session_end_reason: Optional[int]
    turn_count: int
    patient_system_prompt: str
    session_number: int
    current_session_turns: List[TurnRecord]
    miin_injection_turn_numbers: Optional[List[int]]
    miin_injection_active: bool
    last_used_techniques: List[str]
    patient_latent_variables: dict
    latent_variable_logs: List[dict]
    session_latent_trajectory: List[LatentTrajectoryEntry]
    patient_state_reasoning: Optional[dict]
    therapy_program_trace_id: Optional[str]
    patient_last_rolling_memory_update_turn: int
    therapist_last_rolling_memory_update_turn: int
    consistent_client_state: Optional[dict]
    simpatient_internal_state: Optional[dict]
    cami_therapist_state: Optional[dict]
    sustain_streak: Optional[int]
    recent_readiness_deltas: Optional[List[float]]


class MultiSessionState(TypedDict, total=False):
    """
    State object for multi-session therapy (parent graph).
    
    Attributes:
        current_session_number: Current session number (1-4)
        miin_injection_schedule: Pre-computed schedule mapping session_number -> sorted list of injection turn numbers for the entire run
        all_session_turns: List of session turn lists, each session containing a list of TurnRecord objects
        current_session: Nested ConversationState for the ongoing session
        patient_latent_variables: Dynamic latent variable scores (updated during therapy, preserved between sessions)
        patient_profile_data: Static patient profile data (loaded once at init from JSONL)
        all_latent_variable_logs: Per-session lists of latent variable snapshots, accumulated by increment_session_node. Each inner list matches the latent_variable_logs from the completed ConversationState.
        latent_variable_trajectory: Comprehensive longitudinal trajectory of latent variables across all turns and inter-session transitions.
        past_event_titles: Accumulated list of event_title strings from between-session event generation. Used to prevent the event generator from producing duplicate scenarios across sessions.
        bse_trigger_classes: Per-BSE audit record list, one entry per between-session event in the run.
        thread_id: The unique identifier for this therapy run.
        patient_id: The numeric identifier of the patient (e.g., 117).
    """
    thread_id: str
    current_session_number: int
    miin_injection_schedule: dict[int, list[int]]
    all_session_turns: List[List[TurnRecord]]
    current_session: ConversationState
    patient_latent_variables: dict
    patient_profile_data: dict
    all_latent_variable_logs: List[List[dict]]
    latent_variable_trajectory: List[LatentTrajectoryEntry]
    past_event_titles: List[str]
    bse_trigger_classes: List[dict]
    therapy_program_trace_id: Optional[str]
    patient_id: int


# LangGraph recursion limit
# Each conversation turn involves multiple node executions (speaker -> parser -> annotator -> check_end -> next_speaker)
# This should be set higher than MAX_TURNS * nodes_per_turn
RECURSION_LIMIT = 200

# Maximum conversation turns before forcing session end (safety limit)
MAX_TURNS = int(os.getenv("MAX_TURNS", "31"))

# Total number of therapy sessions in a complete run
NUM_SESSIONS = int(os.getenv("NUM_SESSIONS", "4"))