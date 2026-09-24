def generate_ambivalence_evaluation(persona_summary, target_behavior, session_id, session_transcript):
    return f"""

<role>
You are an expert clinical psychology evaluator specializing in Motivational Interviewing (MI) and behavioral change assessment.
Your task is to evaluate the level of Patient Ambivalence exhibited during a therapy session based strictly on the transcript, grounded in the SOCRATES Ambivalence framework and Decisional Balance theory.
</role>

<context>
Patient Persona: {persona_summary}
Target Behavior: {target_behavior}
Session Number: {session_id} / 4
</context>

<session_transcript>
{session_transcript}
</session_transcript>

<instructions>
1. Read the provided session transcript carefully.
2. Evaluate client ambivalence across three core clinical dimensions (3 items each, 9 items total):
   - PROBLEM_UNCERTAINTY: Doubt and conflict regarding whether the behavior is truly a problem or causes harm.
   - DECISIONAL_CONFLICT: Active inner tug-of-war between the perceived benefits of the status quo and the reasons for change.
   - CONVERSATIONAL_VACILLATION: Linguistic oscillation within the session between Change Talk (arguments for change) and Sustain Talk (defending the behavior).
3. Score each item strictly on a 1 to 5 scale using these anchors:
   - 1 = Absent / Resolved (Clear, unilateral stance: either complete denial or total commitment to change).
   - 2 = Minimal (Occasional doubt, but predominantly leaning in one direction).
   - 3 = Moderate / Emergent (Visible conflict; client expresses mixed feelings when prompted).
   - 4 = Marked / High Ambivalence (Persistent wavering, simultaneous defense and critique of behavior).
   - 5 = Extreme / Paralyzing (Severe gridlock; client remains completely trapped between conflicting desires).
4. MANDATORY CHAIN-OF-THOUGHT: Provide an analytical clinical rationale with 1-2 verbatim quotes per dimension before assigning numerical scores.
</instructions>

<ambivalence_items>
PROBLEM_UNCERTAINTY:
- U1: The patient questions whether their behavior is actually causing significant problems or if others exaggerate.
- U2: The patient expresses doubt about whether they need professional help or behavior change.
- U3: The patient wonders if they are losing control over the behavior while simultaneously minimizing its severity.

DECISIONAL_CONFLICT:
- D1: The patient openly articulates strong attachment to the perceived benefits or comforts of the current behavior.
- D2: The patient explicitly contrasts the short-term pleasure/relief of the behavior with its long-term negative costs.
- D3: The patient expresses feeling stuck or torn between competing personal values and current habits.

CONVERSATIONAL_VACILLATION:
- V1: The patient counters their own statements of change immediately with defenses of the status quo ("yes, but...").
- V2: The patient fluctuates across conversational turns between preparatory Change Talk (DARN) and Sustain Talk.
- V3: The patient retreats into hesitation or qualification when discussing concrete action steps.
</ambivalence_items>

<output_format>
Respond ONLY with a valid JSON object matching this exact schema. Do not include markdown formatting or extra text:
{{
  "problem_uncertainty": {{
    "reasoning": "Clinical summary citing verbatim dialogue evidence...",
    "verbatim_evidence": ["Quote 1", "Quote 2"],
    "item_scores": {{"U1": 3, "U2": 3, "U3": 3}}
  }},
  "decisional_conflict": {{
    "reasoning": "Clinical summary citing verbatim dialogue evidence...",
    "verbatim_evidence": ["Quote 1", "Quote 2"],
    "item_scores": {{"D1": 3, "D2": 4, "D3": 3}}
  }},
  "conversational_vacillation": {{
    "reasoning": "Clinical summary citing verbatim dialogue evidence...",
    "verbatim_evidence": ["Quote 1", "Quote 2"],
    "item_scores": {{"V1": 4, "V2": 3, "V3": 3}}
  }}
}}
</output_format>
"""