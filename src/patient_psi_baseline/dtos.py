"""
Pydantic Data Transfer Objects for Patient-Ψ Baseline.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class CognitiveModelUnit(BaseModel):
    """A concrete cognitive behavioral situation-thought-emotion-behavior unit."""
    situation: str = Field(description="The triggering event or context.")
    automatic_thoughts: str = Field(description="Spontaneous inner monologue in response to situation.")
    emotion: str = Field(description="Emotions elicited by automatic thoughts.")
    behavior: str = Field(description="Actions or behaviors resulting from thoughts/emotions.")


class PatientPsiCCD(BaseModel):
    """Cognitive Conceptualization Diagram (CCD) based on Beck's CBT model."""
    life_history: str = Field(description="Relevant personal and developmental history.")
    core_beliefs: str = Field(description="Core belief category (e.g. Helpless, Unlovable, Worthless).")
    core_belief_description: str = Field(description="Specific core belief self-statement.")
    intermediate_beliefs: str = Field(description="Rules, conditional assumptions, and attitudes.")
    intermediate_beliefs_during_depression: str = Field(description="Assumptions active during distress/depression.")
    coping_strategies: str = Field(description="Maladaptive and adaptive coping strategies.")
    cognitive_models: List[CognitiveModelUnit] = Field(
        default_factory=list,
        description="Situational cognitive model instances."
    )


class PatientPsiOutput(BaseModel):
    """Structured response from Patient-Ψ simulated patient."""
    reasoning: str = Field(
        description="Brief internal reflection from the patient's perspective explaining their reaction."
    )
    response: str = Field(
        description="Exact spoken utterance as the patient. Colloquial, natural, authentic."
    )


class PatientPsiMemoryOutput(BaseModel):
    """Structured output for inter-session summary."""
    summary: str = Field(
        description="A concise, chronological, factual paragraph summarising what was discussed in the session."
    )
