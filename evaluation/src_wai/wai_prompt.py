
def generate_wai_evaluation(persona_summary, target_behavior, session_id, session_transcript):
    return f"""

<role>
You are an expert clinical psychology evaluator specializing in Motivational Interviewing (MI) and psychotherapy process research.
Your task is to evaluate the strength of the Therapeutic Alliance established in a specific therapy session based strictly on the dialogue transcript, using the standardized Working Alliance Inventory - Short Revised (WAI-SR) framework.
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
1. Read the provided session transcript thoroughly.
2. Evaluate the working alliance across the three core clinical dimensions of WAI-SR (4 items each, 12 items total):
   - GOAL: Consensus between patient and therapist on the target outcomes, future direction, and personal reasons for change.
   - TASK: Collaborative agreement on in-session activities, exploration methods, and concrete steps, with minimal friction or unsolicited advising.
   - BOND: Mutual trust, warmth, empathy, psychological safety, and unconditional positive regard (therapist rolls with resistance; patient feels respected).
3. Score each item strictly on a 1 to 5 Likert scale using these clinical anchors:
   - 1 = Absent / Severe Rupture (Active confrontation, hostility, total divergence, or disengagement).
   - 2 = Weak / Strained (Superficial compliance, persistent defensiveness, or patronizing tone).
   - 3 = Moderate / Mixed (Basic rapport present; transactional dialogue with lingering ambivalence).
   - 4 = Strong / Collaborative (Clear mutual agreement, active exploration, patient feels heard and accepted).
   - 5 = Exemplary / Deep Alliance (Profound mutual trust, strong collaborative partnership, joint commitment).
4. MANDATORY CHAIN-OF-THOUGHT: For each dimension, you MUST provide an analytical justification containing 1-2 verbatim quotes from the transcript illustrating your rating before providing the final numerical scores.
</instructions>

<wai_sr_items>
GOAL DIMENSION:
- G1: The therapist and patient share a clear, mutual understanding of what goals are important to the patient.
- G2: The therapist focuses on changes that the patient genuinely values, rather than imposing an agenda.
- G3: The patient perceives the session's direction as directly relevant to their personal life challenges.
- G4: The therapist helps the patient resolve ambivalence toward their own chosen goals.

TASK DIMENSION:
- T1: The patient finds the therapist's reflections, open questions, and exercises relevant and helpful.
- T2: The therapist and patient collaboratively determine the steps needed to facilitate change.
- T3: The patient actively engages with the conversational exercises rather than deflecting or minimizing.
- T4: The therapist avoids directive advice-giving without permission, fostering the patient's autonomy.

BOND DIMENSION:
- B1: The patient perceives the therapist as genuinely caring, accepting, and non-judgmental.
- B2: There is a strong sense of mutual trust, safety, and mutual respect throughout the exchange.
- B3: When resistance or frustration arises, the therapist successfully "rolls with resistance" to preserve connection.
- B4: The patient feels comfortable sharing vulnerable facts, doubts, and setbacks without fear of rebuke.
</wai_sr_items>

<output_format>
Respond ONLY with a valid JSON object matching this exact schema. Do not include markdown formatting or extra text:
{{
  "goal_dimension": {{
    "reasoning": "Clinical justification citing verbatim dialogue evidence...",
    "verbatim_evidence": ["Quote 1", "Quote 2"],
    "item_scores": {{"G1": 4, "G2": 4, "G3": 5, "G4": 4}}
  }},
  "task_dimension": {{
    "reasoning": "Clinical justification citing verbatim dialogue evidence...",
    "verbatim_evidence": ["Quote 1", "Quote 2"],
    "item_scores": {{"T1": 4, "T2": 3, "T3": 4, "T4": 4}}
  }},
  "bond_dimension": {{
    "reasoning": "Clinical justification citing verbatim dialogue evidence...",
    "verbatim_evidence": ["Quote 1", "Quote 2"],
    "item_scores": {{"B1": 5, "B2": 4, "B3": 4, "B4": 4}}
  }}
}}
</output_format>
"""