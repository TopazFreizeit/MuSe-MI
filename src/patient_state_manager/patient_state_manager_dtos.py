"""
Pydantic DTOs for Patient State Manager structured LLM output and initialization context.

Each update model mirrors the JSON output schema defined in the corresponding prompt module.
Each init model carries the patient-specific psychological context loaded at startup.
"""
from typing import Any, List, Literal
from enum import Enum
from pydantic import BaseModel, Field


class SemanticImpact(str, Enum):
    SUCCESS = "Success"
    NEUTRAL = "Neutral"
    RESISTANCE = "Resistance"


class MacroShiftSemanticImpact(str, Enum):
    MAJOR_REGRESSION = "Major Regression"
    MINOR_REGRESSION = "Minor Regression"
    NO_CHANGE = "No Change"
    MINOR_PROGRESS = "Minor Progress"
    MAJOR_PROGRESS = "Major Progress"


class ReadinessShift(str, Enum):
    FORWARD = "Forward"
    NEUTRAL = "Neutral"
    BACKWARD = "Backward"


# ---------------------------------------------------------------------------
# Shared base for single-variable latent state updates
# ---------------------------------------------------------------------------

class LatentVariableSemanticUpdate(BaseModel):
    """Common schema for a single latent variable semantic evaluation."""
    reasoning: str = Field(
        description="Explicit reasoning explaining how the therapist's language impacted this specific clinical variable."
    )
    semantic_impact: SemanticImpact = Field(
        description="The categorical classification of the therapist's impact on this variable."
    )


class LatentVariableMacroShift(BaseModel):
    """Common schema for a single latent variable macro-shift evaluation."""
    reasoning: str = Field(
        description="Explicit reasoning explaining why this clinical variable shifted based on the recent event or session consolidation."
    )
    semantic_impact: MacroShiftSemanticImpact = Field(
        description="The categorical classification of the shift in this variable."
    )


# ---------------------------------------------------------------------------
# Unified State Update
# ---------------------------------------------------------------------------

class PatientStateUpdateDTO(BaseModel):
    """Unified structured output for evaluating the semantic impact on all latent variables simultaneously."""
    reasoning_general: str = Field(
        description="A broad single paragraph evaluating the therapist's latest interaction against the patient's decisional balance and acceptable change plans."
    )
    anger: LatentVariableSemanticUpdate = Field(description="Semantic impact on Anger (defensiveness/hostility).")
    motivational_readiness: LatentVariableSemanticUpdate = Field(description="Semantic impact on Motivational Readiness to Change.")
    self_efficacy: LatentVariableSemanticUpdate = Field(description="Semantic impact on Self-Efficacy (confidence to change).")
    problem_recognition: LatentVariableSemanticUpdate = Field(description="Semantic impact on Problem Recognition (acknowledgment of harm).")


# ---------------------------------------------------------------------------
# Macro Shifts (Between-Session and Session Consolidation)
# ---------------------------------------------------------------------------

class MacroShift(BaseModel):
    """Inter-session or end-of-session macro shifts applied to the patient's latent variables."""
    anger: LatentVariableMacroShift = Field(description="Semantic shift on Anger.")
    self_efficacy: LatentVariableMacroShift = Field(description="Semantic shift on Self-Efficacy (confidence to change).")
    problem_recognition: LatentVariableMacroShift = Field(description="Semantic shift on Problem Recognition (acknowledgment of harm).")
    motivational_readiness: LatentVariableMacroShift = Field(description="Semantic shift on Motivational Readiness to Change.")


# ---------------------------------------------------------------------------
# Rolling Memory Update (Every 12 turns & end of session)
# ---------------------------------------------------------------------------

class ChangePlanStatusItem(BaseModel):
    """A single change-plan item the patient is tracking."""
    intention: str = Field(description="What the patient said they would do (verbatim or close paraphrase). E.g., 'suggest a hike instead of drinks on Friday'.")
    attempt_status: Literal["not_yet_attempted", "in_progress", "attempted_succeeded", "attempted_failed", "abandoned"] = Field(
        description="Where this intention stands as of this session."
    )
    outcome_note: str = Field(description="One-line factual note about how the attempt went, or what blocked it. Empty string if not_yet_attempted.")


class _RollingMemoryBaseOutput(BaseModel):
    """
    Structured output for the Rolling Memory Agent (triggered every 12 turns & post-BSE).
    Maintains long-term context explicitly tracking MI fidelity and longitudinal consistency.
    """
    memory_update_reasoning: str = Field(
        description="Explicit rationale for why a fact or thread was added, modified, or dropped. Crucial for evaluating the LLM's memory component performance."
    )
    
    # Accumulative Fields
    facts: List[str] = Field(
        description="Specific, immutable declarative anchors established by the patient (e.g. names, dates, quantities). Do not drop existing facts unless explicitly contradicted."
    )
    agreed_change_plans: List[ChangePlanStatusItem] = Field(
        description="Action plans or concrete steps agreed upon. Update the attempt_status of existing plans or add new ones."
    )
    implicit_threads: List[str] = Field(
        description="Deferred topics or subjects the patient briefly hinted at but were not fully explored. Accumulate these to prevent memoryless sessions."
    )
    
    # Replaced Fields
    motivations: List[str] = Field(
        description="The patient's active Change Talk drivers. Limit to top 3-5 most potent drivers."
    )
    concerns: List[str] = Field(
        description="The patient's current Sustain Talk drivers (the cons of change). Limit to top 3-5 core concerns."
    )
    persona_and_stylistic_baseline: str = Field(
        description="Non-declarative memory tracking the patient's baseline emotional and communicative style (e.g. 'Guarded, uses sarcastic humor')."
    )


class RollingMemoryWithoutMacroShiftOutput(_RollingMemoryBaseOutput):
    """Rolling memory output used when hidden-variable impact is disabled."""


class RollingMemoryOutput(_RollingMemoryBaseOutput):
    """Rolling memory output including latent-variable macro shifts."""

    macro_shift: MacroShift = Field(
        description="Semantic categorizations to shift the patient's latent state based on the accumulated dialogue."
    )
