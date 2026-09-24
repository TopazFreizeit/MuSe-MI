"""
Pydantic DTOs for between-session patient events.
"""
from typing import List, Literal
from pydantic import BaseModel, Field

from ..patient_state_manager.patient_state_manager_dtos import MacroShift

# Step 5 (Edit B, 2026-04-29): structured trigger class for BSE auditability.
# The 2026-04-29 pilot showed every BSE in a four-session arc was a relapse-class
# event despite Self-Efficacy climbing 18→53. The BSE prompt conditioned the
# event OUTCOME on LV but left trigger SELECTION trigger-agnostic. This enum
# makes class selection structured and auditable; §4.6 of the proposal uses it
# for the admissibility check (≥75% relapse/partial-setback flags the patient
# as inadmissible to the Phase 1 sample).
TriggerClass = Literal["success", "near_miss", "mundane_stressor", "partial_setback", "relapse"]


class LatentStateSnapshot(BaseModel):
    """
    LV state at the time the BSE was generated. Logged alongside trigger_class
    for the §4.6 admissibility check, which evaluates trigger_class distribution
    *as a function of* SE/Readiness band — a relapse-monotonic BSE process when
    SE is mid-band is the failure mode the pilot exposed.
    """
    self_efficacy: float
    problem_recognition: float
    anger: float
    motivational_readiness: float


class BetweenSessionEventChunk(BaseModel):
    """Structured output for the Between-Session Event Generator."""
    event_title: str = Field(
        description="Short title summarising the life event (e.g., 'Relapse at a work party')."
    )
    event_summary: str = Field(
        description=(
            "A short narrative summary describing what happened "
            "between sessions, how the patient felt, and what impact it had on their drinking."
        )
    )
    reasoning_chain: List[str] = Field(
        description="Step-by-step reasoning used to generate a plausible, non-repetitive event."
    )
    trigger_class: TriggerClass = Field(
        description=(
            "Categorical class of the trigger event. Must be sampled from the band-conditioned "
            "set defined in Step 2 of the prompt's reasoning chain."
        )
    )
    macro_shift: MacroShift = Field(
        description="Semantic categorizations to shift the patient's latent state based on the impact of this event."
    )
