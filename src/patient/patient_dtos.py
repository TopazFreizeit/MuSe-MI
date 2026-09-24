"""
Pydantic models for Patient structured LLM output and patient profile DTOs.
"""
from typing import Any, List
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Agent A: Cognitive Strategist structured output
# ---------------------------------------------------------------------------

class CognitiveLayer(BaseModel):
    decision_rule: str = Field(description="The heuristic the patient uses to decide whether to engage or deflect based on the therapist's approach (e.g., 'If they tell me what to do, I will push back').")
    optimizing_for: str = Field(description="What the patient is actively trying to protect or achieve in this moment (e.g., 'Protecting my autonomy', 'Avoiding shame', 'Maintaining the status quo').")

class PsychologicalLayer(BaseModel):
    defense_pattern: str = Field(description="The primary MI-specific resistance behavior active right now (e.g., 'Justifying', 'Minimizing', 'Interrupting', 'Deflecting', 'Agreeing to appease').")
    core_paradox: str = Field(description="The central ambivalence driving the patient: the tension between their sustain talk drivers and change talk drivers.")

class PatientCognitiveDirective(BaseModel):
    """Structured output for Agent A (Cognitive Strategist). Uses a Thinking Framework architecture."""
    cognitive_layer: CognitiveLayer
    conversational_stance: str = Field(description="Explicit stance based on your frustration and ambivalence (e.g., Deflective, Challenging, Cautious Consideration).")
    psychological_layer: PsychologicalLayer
    active_reasoning: str = Field(description="First-person, in-character reasoning about your current feelings and ambivalence.")
    pressing_disclosure_intention: str = Field(description="Your agenda to discuss a recent event or struggle, or what you want to admit/hide.")
    response: str = Field(description="Your exact spoken words as a natural, colloquial 1-2 sentence reply. Must be messy and authentic, NO therapy-speak.")


# ---------------------------------------------------------------------------
# Latent Variables (dynamic – updated every X turns during therapy)
# ---------------------------------------------------------------------------

class PatientLatentVariables(BaseModel):
    """All latent variable scores that change dynamically during therapy.
    Normalized to a 1-100 scale."""
    problem_recognition: float
    anger: float
    motivational_readiness: float
    self_efficacy: float


# ---------------------------------------------------------------------------
# Patient Profile Data (static – loaded once from profile JSONL)
# ---------------------------------------------------------------------------

class PatientProfileData(BaseModel):
    """Patient profile extracted from consistent clients JSONL."""
    idx: int
    topic: str
    personas: List[str]
    acceptable_plans: List[str]
    beliefs: List[str]
    motivation: List[str]
    behavior: str
    suggestibilities: List[float]
    initial_state: str
