"""
Pydantic Data Transfer Objects for SimPatient Baseline.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SimPatientOutput(BaseModel):
    """Spoken response and non-verbal cue selection from SimPatient."""
    patient_response: str = Field(
        description="Your exact spoken words as the patient. Short, natural, colloquial."
    )
    non_verbal_cue_number: int = Field(
        default=0,
        ge=0,
        le=12,
        description="The integer corresponding to the chosen non-verbal cue (0 to 12)."
    )


class SimPatientInternalStateOutput(BaseModel):
    """Evaluator output updating the 4 cognitive state variables on a 1-10 scale."""
    patient_control: int = Field(
        ge=1,
        le=10,
        description="Your level of ability to regulate your own thoughts, emotions, and actions (1-10 scale)."
    )
    patient_efficacy: int = Field(
        ge=1,
        le=10,
        description="Your level of confidence in your ability to resist cravings, cope with triggers, and achieve recovery goals (1-10 scale)."
    )
    patient_awareness: int = Field(
        ge=1,
        le=10,
        description="Your level of ability to accurately perceive and evaluate your own thoughts, feelings, and behaviors (1-10 scale)."
    )
    patient_reward: int = Field(
        ge=1,
        le=10,
        description="The level in which substance and its cues trigger cravings and automatic behaviors in you (1-10 scale)."
    )
    reasoning: str = Field(
        description="Single overall reasoning for the updated values."
    )


class SimPatientBetweenSessionEventOutput(BaseModel):
    """Generated life event occurring between therapy sessions."""
    event: str = Field(
        description="A concise narrative description of an event experienced between sessions."
    )


class SimPatientMemoryState(BaseModel):
    """Cross-session memory persisted in runtime.store."""
    past_session_history: str = ""
    between_session_event: str = ""
    patient_control: int = 5
    patient_efficacy: int = 5
    patient_awareness: int = 5
    patient_reward: int = 5
    global_scores: Optional[dict] = None
    behavior_counts: Optional[dict] = None


class MITIEncodingOutput(BaseModel):
    """Output matching GenerateMITIEncodingResponse.cs for classifying counselor volleys."""
    behavior_codes: list[str] = Field(
        default_factory=list,
        description="List of behavior codes picked from: GI, Persuade, Persuade with, Q, SR, CR, AF, Seek, Emphasize, Confront."
    )
    reasoning: str = Field(
        description="Reasoning for choosing the behavior codes."
    )


class GlobalScoreDict(BaseModel):
    score: int = Field(
        ge=1,
        le=5,
        description="Score on a 1(low) to 5(high) Likert scale."
    )
    reasoning: str = Field(
        description="Reasoning for the score."
    )


class GlobalScoresOutput(BaseModel):
    """Output matching GenerateGlobalScoresResponse.cs for MITI global scores."""
    cultivating_change_talk: GlobalScoreDict = Field(description="Cultivating change talk global score and reasoning.")
    softening_sustain_talk: GlobalScoreDict = Field(description="Softening sustain talk global score and reasoning.")
    partnership: GlobalScoreDict = Field(description="Partnership global score and reasoning.")
    empathy: GlobalScoreDict = Field(description="Empathy global score and reasoning.")


class SessionSummaryOutput(BaseModel):
    """Output matching GenerateSessionSummaryResponse.cs for session summary."""
    summary: str = Field(
        description="A concise summary of the session highlighting key points, insights, and recommendations."
    )

    patient_awareness: int = 5
    patient_reward: int = 5
