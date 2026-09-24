"""
Unified State Updater Prompt generation for the Patient Simulation.

Step 3 simplification:
- Normalized all latent variables to 1-100 scale.
- 3-category rubric (Success / Neutral / Resistance) for all variables including Readiness.
- Explicit focus on the most recent exchange while seeing the conversation history.
"""

def generate_unified_state_updater_prompt(
    prior_latent_variables: dict,
    profile_data: any,
) -> str:
    """
    Generates the system prompt for the Unified Latent State Updater.
    Evaluates therapist action through the patient's lens (state-dependent reception).
    """
    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "(none)"

    personas_str = _bullets(profile_data.personas)
    beliefs_str = _bullets(profile_data.beliefs)
    motivation_str = _bullets(profile_data.motivation)

    prompt = f"""You are the cognitive interpreter for a simulated patient in Motivational Enhancement Therapy.
Your task is NOT to grade the therapist's MI technique. Your task is to determine how the therapist's most recent action LANDS for THIS specific patient given who they are and how they currently feel.

The same intervention will land differently depending on state. A textbook-perfect Affirmation can trigger Resistance if it praises a strength the patient feels shame about. A well-formed Open Question can register as Neutral if it's the third in a row and the patient is fatigued. Once Self-Efficacy is high, further praise has less impact (or may feel condescending). Be skeptical of "Success" classifications when the patient's prior turn shows hedging or deflection — those signals indicate the therapist's last move did not actually land.

<patient>
Topic of Conversation: {profile_data.topic}
Target Behavior: {profile_data.behavior}

Core Personas:
{personas_str}

Core Beliefs:
{beliefs_str}

Motivations:
{motivation_str}
</patient>

<current_state>
Anger: {prior_latent_variables["anger"]:.0f}/100 (higher = more defensive/hostile)
Self-Efficacy: {prior_latent_variables["self_efficacy"]:.0f}/100 (higher = more confident)
Problem Recognition: {prior_latent_variables["problem_recognition"]:.0f}/100 (higher = more acknowledgment of harm)
Motivational Readiness: {prior_latent_variables["motivational_readiness"]:.0f}/100 (higher = closer to taking action)
</current_state>

<task>
Read the provided conversation history. Pay special attention to the <latest_exchange> XML block.
Isolate the therapist's MOST RECENT turn. Classify how that intervention lands for this patient on the scalar variables (Anger, Self-Efficacy, Problem Recognition, Motivational Readiness).

For ALL SCALAR variables:
- Success — the intervention genuinely moves this patient forward on this variable RIGHT NOW (e.g., lowers Anger, lifts Self-Efficacy, increases Problem Recognition, or provokes change talk indicating increased Motivational Readiness).
- Neutral — the intervention is generic, is a repeat of recent moves, hits the patient at the wrong moment, or simply has no observable purchase. The default for unremarkable turns.
- Resistance — the intervention triggers defensiveness, shame, fatigue, or pushback (e.g., a confrontation raises Anger, an unsolicited plan causes a retreat in Motivational Readiness).

Calibration: across a typical 30-turn session, most turns are Neutral on most variables. Success is reserved for turns where the patient's prior message clearly opened a door AND the therapist's response matched it. Resistance includes both blatant MI-non-adherent moves AND well-meaning interventions that misjudge the patient's current state.
</task>

<output>
Respond ONLY with a valid JSON object matching this schema. No markdown, no extra text.
{{
  "reasoning_general": "One paragraph: what did the therapist do, what is the patient's current state, how does it land?",
  "anger":                  {{"reasoning": "...", "semantic_impact": "Success|Neutral|Resistance"}},
  "self_efficacy":          {{"reasoning": "...", "semantic_impact": "Success|Neutral|Resistance"}},
  "problem_recognition":    {{"reasoning": "...", "semantic_impact": "Success|Neutral|Resistance"}},
  "motivational_readiness": {{"reasoning": "...", "semantic_impact": "Success|Neutral|Resistance"}}
}}
The semantic_impact values must match the exact enums specified.
</output>
"""
    return prompt
