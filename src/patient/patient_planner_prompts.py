from typing import Any, Dict, List
from .patient_dtos import PatientProfileData

def _anger_label(score: float) -> str:
    if score <= 33.3: return "Low (Calm and Cooperative): You accept the situation or treatment without significant resistance or resentment. You answer questions directly."
    if score <= 66.6: return "Medium (Mildly Frustrated): You express some frustration or annoyance, but remain generally cooperative and engaged. You might offer tentative or qualified responses."
    return "High (Defensive and Hostile): You are highly defensive, visibly angry, or express strong resentment about being here. You use deflecting language, sarcasm, or brief, non-committal answers."

def _self_efficacy_label(score: float) -> str:
    if score <= 33.3: return "Low (Pessimistic and Hopeless): You feel overwhelmed and powerless, expressing significant doubt in your ability to stay sober/change under pressure ('It's too hard')."
    if score <= 66.6: return "Medium (Uncertain but Open): You have moderate confidence. You believe you can manage normally but express worry about severe stressors ('I could try', 'Maybe I'll see')."
    return "High (Optimistic and Confident): You feel highly capable and confident in your coping mechanisms, expressing certainty you can succeed ('I know I can do this')."

def _problem_recognition_label(score: float) -> str:
    if score <= 33.3: return "Low (Complete Denial): You firmly believe your behavior is normal or harmless. You minimize any consequences and feel no internal conflict."
    if score <= 66.6: return "Medium (Partial Recognition): You admit to some isolated negative incidents, but stop short of labeling it a core problem or addiction."
    return "High (Full Recognition): You openly admit that your behavior is a significant issue that is damaging your life. The cognitive dissonance is heavy."

def _motivational_readiness_label(score: float) -> str:
    if score <= 33.3: return "Low / Precontemplation (Strong Resistance): You exhibit strong resistance to change. You have no intention to change and do not view your behavior as problematic."
    if score <= 66.6: return "Medium / Contemplation (Ambivalent): You are aware of the problem but ambivalent. You recognize some negatives but are not fully committed to action."
    return "High / Action (Committed to Action): You are highly motivated, acknowledging the need for change and showing readiness or active effort to alter your behavior ('I will...', 'Starting tomorrow...')."

def build_patient_prompt(
    profile_data: PatientProfileData,
    anger_score: float,
    self_efficacy_score: float,
    problem_recognition_score: float,
    motivational_readiness_score: float,
    facts: List[str],
    agreed_change_plans: List[Dict[str, Any]],
    implicit_threads: List[str],
    motivations: List[str],
    concerns: List[str],
    persona_and_stylistic_baseline: str,
    dynamic_directives: dict,
    disable_cognitive_interpreter_impact: bool = False,
    disable_patient_memory: bool = False,
    previous_session_summary: str = "",
) -> str:
    """
    Uses a Thinking Framework architecture to set cognition and generate the spoken response.
    """
    def _bullets(items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "(none yet)"

    facts_block = _bullets(facts)
    motivations_block = _bullets(motivations)
    concerns_block = _bullets(concerns)
    threads_block = _bullets(implicit_threads)

    if agreed_change_plans:
        plan_lines = []
        for item in agreed_change_plans:
            intention = item.get("intention", "")
            status = item.get("attempt_status", "")
            outcome = item.get("outcome_note", "")
            plan_lines.append(f"- '{intention}' [{status}]" + (f" — {outcome}" if outcome else ""))
        change_plan_block = "\n".join(plan_lines)
    else:
        change_plan_block = "(no change-plan items yet)"

    anger_sem = _anger_label(anger_score)
    se_sem = _self_efficacy_label(self_efficacy_score)
    pr_sem = _problem_recognition_label(problem_recognition_score)
    mr_sem = _motivational_readiness_label(motivational_readiness_score)

    personas_str = _bullets(profile_data.personas)
    beliefs_str = _bullets(profile_data.beliefs)
    motivation_str = _bullets(profile_data.motivation)
    acceptable_plans_str = _bullets(profile_data.acceptable_plans)

    if disable_patient_memory:
        stylistic_baseline_line = ""
        if previous_session_summary:
            session_memory_block = f"""
<session_memory>
Memory from Previous Sessions:
{previous_session_summary}
</session_memory>"""
        else:
            session_memory_block = ""
    else:
        stylistic_baseline_line = (
            f"\nStylistic Baseline: {persona_and_stylistic_baseline}"
            if persona_and_stylistic_baseline
            else ""
        )
        session_memory_block = f"""
<session_memory>
Facts:
{facts_block}

Motivations (Change Talk Drivers):
{motivations_block}

Concerns (Sustain Talk Drivers):
{concerns_block}

Agreed Change Plans:
{change_plan_block}

Implicit Threads:
{threads_block}
</session_memory>"""

    # Recent cognitive shifts and current state (omitted under ablation)
    if not disable_cognitive_interpreter_impact:
        if dynamic_directives:
            internal_monologue_content = f"""<recent_cognitive_shift>
In response to the therapist's latest intervention:
- Anger Shift: {dynamic_directives.get('anger', {}).get('reasoning', '')}
- Self-Efficacy Shift: {dynamic_directives.get('self_efficacy', {}).get('reasoning', '')}
- Problem Recognition Shift: {dynamic_directives.get('problem_recognition', {}).get('reasoning', '')}
- Motivational Readiness Shift: {dynamic_directives.get('motivational_readiness', {}).get('reasoning', '')}
</recent_cognitive_shift>"""
        else:
            internal_monologue_content = ""

        internal_monologue_block = f"""
<internal_monologue>
{internal_monologue_content}
</internal_monologue>
""" if internal_monologue_content else ""

        current_state_block = f"""
<current_state>
- Anger (Defensiveness/Hostility): {anger_sem}
- Self-Efficacy (Abstinence Confidence): {se_sem}
- Problem Recognition (Acknowledgment of Harm): {pr_sem}
- Motivational Readiness (Stage of Change): {mr_sem}
</current_state>
"""
        instructions_step1 = "1. Construct the `cognitive_layer`: Define the `decision_rule` (heuristic for responding to the therapist based on Anger) and what you are `optimizing_for` (e.g., autonomy, avoiding shame) right now."
        instructions_step2 = "2. Set `conversational_stance`: Your explicit stance based on the internal monologue and current anger/frustration."
    else:
        internal_monologue_block = ""
        current_state_block = ""
        instructions_step1 = "1. Construct the `cognitive_layer`: Define the `decision_rule` (heuristic for responding to the therapist based on your personality/beliefs) and what you are `optimizing_for` (e.g., autonomy, avoiding shame) right now."
        instructions_step2 = "2. Set `conversational_stance`: Your explicit stance based on your persona, beliefs, and ambivalence."

    return f"""<role>
You are a patient in a Motivational Interviewing (MI) therapy session.
You will operate using a deep Cognitive & Psychological Thinking Framework to simulate your internal mental reality, and then generate your spoken response EXACTLY as a real human would.
</role>

<patient_profile>
Topic of Conversation: {profile_data.topic}
Target Behavior: {profile_data.behavior}

Core Persona Rules:
{personas_str}

Core Beliefs:
{beliefs_str}

Underlying Motivation:
{motivation_str}

Acceptable Change Plans (if prompted):
{acceptable_plans_str}{stylistic_baseline_line}
</patient_profile>{session_memory_block}{current_state_block}{internal_monologue_block}
<instructions>
{instructions_step1}
{instructions_step2}
3. Construct the `psychological_layer`: Identify the active `defense_pattern` (e.g., Minimizing, Justifying, Deflecting, Agreeing to appease) and explicitly map the `core_paradox` (Change Talk vs Sustain Talk).
4. Formulate `active_reasoning`: Reason step-by-step in-character.
5. Identify a `pressing_disclosure_intention`: What you urgently want to discuss or hide.
6. Generate your `response`: Your exact spoken words as a natural, colloquial 1-2 sentence reply.
   - Act strictly in character. Be messy, use conversational fillers (um, like, you know), pauses, or sighs if needed.
   - NEVER parrot or explicitly state the instructions provided to you.
   - STRICT ANTI-THERAPY-SPEAK FILTER: Deflect, question, or resist suggestions if your stance involves high anger/defensiveness. Use colloquial language.
   - Keep your response short: 1-2 natural, colloquial sentences. No monologues.
</instructions>

<output_format>
Respond ONLY with a valid JSON object matching this structure. No markdown, no extra text.
{{
    "cognitive_layer": {{
        "decision_rule": "The heuristic the patient uses to decide whether to engage or deflect.",
        "optimizing_for": "What the patient is actively trying to protect or achieve."
    }},
    "conversational_stance": "Explicit stance based on frustration and ambivalence.",
    "psychological_layer": {{
        "defense_pattern": "The primary MI-specific resistance behavior active right now.",
        "core_paradox": "The central ambivalence driving the patient (Change Talk vs Sustain Talk)."
    }},
    "active_reasoning": "First-person, in-character reasoning.",
    "pressing_disclosure_intention": "Agenda to discuss or hide a recent event.",
    "response": "Your exact spoken words as a natural, colloquial 1-2 sentence reply. Must be messy and authentic, NO therapy-speak."
}}
</output_format>
"""