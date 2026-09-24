"""
Pydantic models for Therapist structured LLM output — 3-agent pipeline.
"""
from typing import Any, List, Literal, TypedDict, get_args
from pydantic import BaseModel, Field, field_validator

MItechniques = Literal[
    "ADP", # Advise with Permission
    "AF", # Affirm
    "EC", # Emphasize Control
    "FA", # Facilitate
    "GI", # Giving Information
    "OQ", # Open Question
    "CQ", # Closed Question
    "RCP", # Raise Concern with Permission
    "CR", # Complex Reflection
    "SR", # Simple Reflection
    "DSR", # Double-Sided Reflection
    "RF", # Reframe
    "ST", # Structure
    "SU", # Support
    "SM", # Summarize
    "SF" # Shifting Focus
]

StageOfChange = Literal["Precontemplation", "Contemplation", "Action"]
MIPhase = Literal["Engaging", "Focusing", "Evoking", "Planning"]

FormulatorTechniques = Literal[
    "ADP", "AF", "EC", "FA", "GI", "OQ", "CQ", "RCP", "CR", "SR", "DSR", "RF", "ST", "SU", "SM", "SF", "FILLER"
]


# ---------------------------------------------------------------------------
# Clinical Analyzer output
# ---------------------------------------------------------------------------

class ClinicalAnalyzerPatientDiagnosis(BaseModel):
    classification_reasoning: str = Field(
        description="A short trace string explaining *why* the utterance was classified as DARN/CAT/etc. (critical for debugging misclassifications)."
    )
    resistance_detected: str = Field(
        description="Verbatim sustain talk, defensiveness, or minimization from the patient. 'none' if absent."
    )
    preparatory_change_talk: str = Field(
        description="Verbatim DARN talk (desire, ability, reason, need to change). 'none' if absent."
    )
    commitment_and_action_talk: str = Field(
        description="Verbatim firm commitment language, concrete planning, or steps taken. 'none' if absent."
    )
    stage_of_change: StageOfChange = Field(
        description="TTM stage classification: Precontemplation, Contemplation, or Action."
    )
    planning_reasoning_trace: str = Field(
        description="Step-by-step reasoning evaluating whether the patient's current CAT talk warrants a transition to the planning phase."
    )
    ready_for_planning: bool = Field(
        description="True if the reasoning trace concludes the patient shows clear signs of commitment or action and is ready to formulate a plan."
    )
    stuckness_reasoning_trace: str = Field(
        description="Step-by-step reasoning analyzing the <conversation_history> to determine if the patient is repeating the same objection or sustain talk without progressing."
    )
    patient_is_stuck: bool = Field(
        description="True if the reasoning trace concludes the patient is stuck in a repetitive loop."
    )
    guidance_reasoning_trace: str = Field(
        description="Brief analysis of the patient's latest turn to see if they are explicitly asking for advice, information, or direction."
    )
    patient_requests_guidance_or_info: bool = Field(
        description="True if the reasoning trace concludes the patient explicitly seeks advice, information, or instructions."
    )


# ---------------------------------------------------------------------------
# State tracking structured records
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Clinical Analyzer output
# ---------------------------------------------------------------------------

class ClinicalAnalyzerOutput(BaseModel):
    patient_diagnosis: ClinicalAnalyzerPatientDiagnosis


# ---------------------------------------------------------------------------
# Therapist Rolling Memory
# ---------------------------------------------------------------------------

class ChangePlanItem(BaseModel):
    intention: str = Field(description="What the patient said they would do (verbatim or close paraphrase).")
    attempt_status: Literal["not_yet_attempted", "in_progress", "attempted_succeeded", "attempted_failed", "abandoned"] = Field(description="The current status of this plan item.")
    outcome_note: str = Field(description="One-line factual note on outcome, or empty string if not_yet_attempted.")

class TherapistRollingMemoryOutput(BaseModel):
    memory_update_reasoning: str = Field(description="Explicit rationale for every change made to this memory record.")
    biographical_facts: List[str] = Field(description="Hard facts about the patient's life (Accumulative).")
    core_values: List[str] = Field(description="Deeply held values the patient expresses (Accumulative).")
    target_behavior: str = Field(description="The specific behavior targeted for change (Consolidate).")
    darn_change_talk: List[str] = Field(description="Distilled instances of preparatory change talk. Limit to top 3-5 statements.")
    cat_change_talk: List[str] = Field(description="Distilled instances of mobilizing change talk. Limit to top 3-5 statements.")
    sustain_talk_themes: List[str] = Field(description="Current active resistance or reasons not to change. Limit to top 3-5 themes.")
    change_plan: List[ChangePlanItem] = Field(description="Agreed upon steps and execution status (Consolidate).")
    current_clinical_focus: str = Field(description="Overarching clinical priority for the current moment or next session (Consolidate).")


# ---------------------------------------------------------------------------
# Clinical Strategist output
# ---------------------------------------------------------------------------

class ClinicalStrategistOutput(BaseModel):
    strategist_reasoning_trace: List[str] = Field(
        description="Step-by-step reasoning: patient state, MI phase determination, explicitly referencing the Phase Escalation Heuristic if triggered."
    )
    current_mi_phase: MIPhase = Field(
        description="The current active Motivational Interviewing Phase: Engaging, Focusing, Evoking, or Planning."
    )
    macro_intent: str = Field(
        description="Specific clinical goal for the turn."
    )
    allowed_techniques: List[MItechniques] = Field(
        description="Hard constraint: Techniques the Formulator is allowed to use this turn."
    )


# ---------------------------------------------------------------------------
# Formulator output
# ---------------------------------------------------------------------------

class FormulatorOutput(BaseModel):
    reasoning_chain: List[Any] = Field(
        description="Exactly five reasoning steps: technique selection, draft, righting reflex check, structural verification, and refinement."
    )
    chosen_technique: FormulatorTechniques = Field(
        description="The MI technique code selected from the candidate list to execute in this response."
    )
    final_response: str = Field(
        description="The spoken therapist response (1st person, 3 sentences max)."
    )
