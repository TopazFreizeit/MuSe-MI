import json
from ..patient.patient_dtos import PatientProfileData

def build_between_session_event_prompt(
    profile_data: PatientProfileData,
    past_event_titles: list[str],  # formatted as bullet list inside the function
    between_session_occurrences: list[str],
    evolving_decisional_balance: dict,
    anger: float,
    self_efficacy: float,
    problem_recognition: float,
    motivational_readiness: float,
    allowed_trigger_classes: list[str]
) -> str:
    """
    Generates a simulated real-world event that occurs between therapy sessions.
    Outputs a narrative summary of the event.
    The full therapeutic conversation history MUST be passed as user context.
    """
    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    personas_str = _bullets(profile_data.personas)

    balance_str = json.dumps(evolving_decisional_balance, indent=2)
    past_events_str = "\n".join(f"- {t}" for t in past_event_titles) if past_event_titles else "(none yet)"
    occurrences_str = "\n".join(f"- {e}" for e in between_session_occurrences) if between_session_occurrences else "(none yet)"
    allowed_classes_str = ", ".join([f'"{c}"' for c in allowed_trigger_classes])

    behavior = profile_data.behavior

    trigger_definitions_map = {
        "success": f'"success": The patient navigated a high-risk situation successfully (e.g., refused {behavior}, chose a different activity).',
        "near_miss": f'"near_miss": Close call but maintained the goal (e.g., almost engaged in {behavior}, called a friend instead).',
        "mundane_stressor": f'"mundane_stressor": Ordinary life stress not directly related to {behavior} (e.g., car trouble, work deadline, family argument).',
        "partial_setback": '"partial_setback": Some slip but recovered or self-interrupted.',
        "relapse": f'"relapse": Full return to problematic {behavior}, no recovery within the event.'
    }
    allowed_definitions_str = "\n".join([f"- {trigger_definitions_map[c]}" for c in allowed_trigger_classes if c in trigger_definitions_map])

    prompt = f"""<role>
You are the 'Life Simulator' for a simulated patient undergoing Motivational Enhancement Therapy (MET).
Your task is to generate a realistic life event that happens to the patient in the time gap between their last therapy session and the upcoming one.
</role>

<input_data>
<patient_profile>
Topic of Conversation: {profile_data.topic}
Target Behavior: {behavior}

Core Personas:
{personas_str}
</patient_profile>

<past_events>
{past_events_str}
</past_events>

<past_between_session_occurrences>
{occurrences_str}
</past_between_session_occurrences>

<cognitive_state>
- Anger (Defensiveness/Hostility, 0-100): {anger:.1f}
- Self-Efficacy (Confidence to change, 0-100): {self_efficacy:.1f}
- Problem Recognition (Acknowledgment of harm, 0-100): {problem_recognition:.1f}
- Motivational Readiness (Stage of Change, 0-100): {motivational_readiness:.1f}
</cognitive_state>

<evolving_decisional_balance>
{balance_str}
</evolving_decisional_balance>

<context>
The Full Chat History of the last session is provided in the user message.
</context>
</input_data>

<mission>
Based on how the last session went AND the patient's current Cognitive State (including their Decisional Balance, Problem Recognition, and Anger), simulate what happens when reality hits.
Did they stick to their goals? Did a stressor trigger a relapse? Did they partially succeed?

**CRITICAL CONSTRAINT 1: LATENT-STATE-CONDITIONED TRIGGER-CLASS SAMPLING**
The CLASS of trigger you may choose has been pre-calculated based on the patient's current latent state to prevent unrealistic relapse trajectories.

You MUST sample `trigger_class` from the following allowed set: {allowed_classes_str}

Here are the definitions for the allowed trigger classes:
{allowed_definitions_str}

Pick the specific trigger from `static_triggers` (the patient's high-risk-events list) when sampling `relapse`, `partial_setback`, `near_miss`, or `success`. Invent a mundane stressor only when sampling `mundane_stressor`.

**CRITICAL CONSTRAINT 2: NARRATIVE OUTCOME MUST MATCH TRIGGER CLASS**
The event's narrative outcome must match the chosen `trigger_class`. Do not, e.g., choose `near_miss` and then write a relapse narrative.

Generate a short narrative summary capturing this event and how it affected the patient. Make it realistic, messy, and grounded in the likelihood predicted by their state.
You must also output a MACRO SHIFT to update the patient's latent variables, representing the lasting effect of this event on their psyche. The macro_shift's direction should be consistent with `trigger_class`: success/near_miss → progress on Self-Efficacy and forward-stage readiness, potentially reducing Anger; relapse/partial_setback → regression, potentially increasing Anger; mundane_stressor → small or no change.
</mission>

<output_format>
You must respond ONLY with a valid JSON object matching the exact structure below. No extra text.

{{
  "reasoning_chain": [
    "Step 1 (Trigger-Class Selection): Name the chosen trigger_class, confirm it is in the allowed set ({allowed_classes_str}). Explain how the specific invented trigger aligns with the patient's personas.",
    "Step 2 (Scenario Draft): Describe the scene — where, who, what happens, how the patient feels. Confirm the narrative outcome matches the chosen trigger_class."
  ],
  "trigger_class": "<one of: {allowed_classes_str}>",
  "chosen_trigger": "<a short description of the specific invented trigger event>",
  "event_title": "A short, distinct 4-5 word title for this event (e.g., 'Relapse at a work party').",
  "event_summary": "A short narrative summary of what happened between sessions, how the patient felt, and what impact it had on their drinking or motivation. Narrative outcome must match `trigger_class`.",
  "macro_shift": {{
    "anger": {{
      "reasoning": "Explain why anger shifted this way",
      "semantic_impact": "<Major Regression | Minor Regression | No Change | Minor Progress | Major Progress>"
    }},
    "self_efficacy": {{
      "reasoning": "Explain why self-efficacy shifted this way",
      "semantic_impact": "<Major Regression | Minor Regression | No Change | Minor Progress | Major Progress>"
    }},
    "problem_recognition": {{
      "reasoning": "Explain why problem recognition shifted this way",
      "semantic_impact": "<Major Regression | Minor Regression | No Change | Minor Progress | Major Progress>"
    }},
    "motivational_readiness": {{
      "reasoning": "Explain why motivational readiness shifted this way",
      "semantic_impact": "<Major Regression | Minor Regression | No Change | Minor Progress | Major Progress>"
    }}
  }}
}}
</output_format>
"""
    return prompt