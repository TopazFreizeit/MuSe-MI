from typing import Dict, List, Optional, Any
from .therapist_dtos import ClinicalAnalyzerOutput


MI_TECHNIQUES: dict[str, dict[str, str]] = {
    "ADP": {
        "name": "Advise With Permission",
        "definition": "Suggest small adjustments or advice ONLY after getting permission. If permission was already granted in the previous turn, provide the advice directly.",
        "example": """Patient: "I don't know how to track my triggers."
Bad Response: "Start keeping a daily log."
Good Response (Asking): "Would it be okay if I shared what helped other people in similar spots?"
Good Response (Giving, if permission just granted): "Something that helps many is keeping a daily log on your phone." """,
    },
    "AF": {
        "name": "Affirm",
        "definition": "Sincerely reinforce the patient's strengths, insights, or efforts.",
        "example": """Patient: "I went three days without drinking."
Bad Response: "Good job."
Good Response: "You put in a lot of hard work to stay sober for those three days.""",
    },
    "EC": {
        "name": "Emphasize Control",
        "definition": "Explicitly acknowledge the patient's autonomy and freedom of choice.",
        "example": """Patient: "Are you going to make me go to A.A.?"
Bad Response: "You should go to A.A."
Good Response: "It's entirely up to you whether you go to A.A. or not. Nobody can make that choice for you.""",
    },
    "FA": {
        "name": "Facilitate",
        "definition": "Brief continuers (Mm-hmm, tell me more) to keep exploration going.",
        "example": """Patient: "It's just been a really hard week."
Bad Response: "Why?"
Good Response: "Tell me more about that.""",
    },
    "GI": {
        "name": "Giving Information",
        "definition": "Present factual, objective information in a neutral, non-judgmental manner without advising.",
        "example": """Patient: "Can drinking cause memory loss?"
Bad Response: "Yes, you need to stop drinking before you ruin your brain."
Good Response: "Yes, heavy alcohol use can interfere with the brain's ability to form new memories.""",
    },
    "OQ": {
        "name": "Open Question",
        "definition": "Questions that encourage exploration and self-motivational talk.",
        "example": """Patient: "I'm just not sure if I'm ready to completely stop."
Bad Response: "Could you tell me if you want to change or not change?"
Good Response: "What would have to happen for you to consider change?""",
    },
    "CQ": {
        "name": "Closed Question",
        "definition": "For specific factual confirmation (use sparingly).",
        "example": """Patient: "I only drink on weekends."
Bad Response: "Why?"
Good Response: "Do you mean Friday through Sunday?""",
    },
    "RCP": {
        "name": "Raise Concern With Permission",
        "definition": "Express a clinical concern only after getting permission. If permission was just granted in the previous turn, state the concern directly without asking again.",
        "example": """Patient: "I mix my pills with alcohol but I'm fine."
Bad Response: "That's incredibly dangerous."
Good Response (Asking): "I have some concerns about mixing those medications with alcohol. Would it be alright if I shared them with you?"
Good Response (Giving, if permission just granted): "My concern is that mixing these can cause severe respiratory depression." """,
    },
    "CR": {
        "name": "Complex Reflection",
        "definition": "Reflecting unspoken emotions or deeper meaning.",
        "example": """Patient: "The court sent me here, everyone's getting on me about my drinking."
Bad Response: "It sounds like you're frustrated that the court and others are making you come here."
Good Response: "That's the only reason you're here. It's kind of like a bunch of crows pecking at you.""",
    },
    "SR": {
        "name": "Simple Reflection",
        "definition": "Reflect back the core meaning of the patient's statement using your own words and fresh syntax to encourage talking.",
        "example": """Patient: "It's like, I can just be myself when I'm playing music, which is pretty cool."
    Bad Response: "Being yourself through music is pretty cool."
    Good Response: "Music gives you a unique space to be authentic." """,
    },
    "DSR": {
        "name": "Double-Sided Reflection",
        "definition": "Capturing both sides of ambivalence (On one hand [Sustain]... and on the other hand [Change]). End with the change talk side.",
        "example": """Patient: "I don't like what smoking does to my health, but it really reduces my stress."
Bad Response: "On one hand you're concerned about your health, on the other you need the relief."
Good Response: "You want to change, but the comfort of old habits has a strong pull.""",
    },
    "RF": {
        "name": "Reframe",
        "definition": "Suggesting a new, more clinical or positive meaning for an experience.",
        "example": """Patient: "I've tried to quit 5 times and failed. I'm taking a break."
Bad Response: "You can't give up."
Good Response: "You're taking a step back to figure out a better approach before trying again.""",
    },
    "ST": {
        "name": "Structure",
        "definition": "Transitioning between topics or explaining session flow. It is acceptable to briefly validate or reflect the patient's preceding thought before making the transition.",
        "example": """Patient: "I don't know what to talk about."
Bad Response: "What do you want to talk about?"
Good Response: "Today we have about 45 minutes. We could review your assessment results or talk about what brought you in. What makes the most sense to you?""",
    },
    "SU": {
        "name": "Support",
        "definition": "Expressing compassion for the patient's burden/struggle (not the behavior).",
        "example": """Patient: "I feel like a terrible mother."
Bad Response: "Don't feel that way."
Good Response: "You're going through an incredibly painful time right now, and it makes sense that you're feeling overwhelmed.""",
    },
    "SM": {
        "name": "Summarize",
        "definition": "A longer reflection of multiple themes, focusing on self-motivation.",
        "example": """Patient: "[Long story about multiple problems]"
Bad Response: "Wow, that's a lot."
Good Response: "Let me pull together what you've shared so far. You're feeling stressed by work, and the drinking is causing friction at home, but you're also worried about how quitting might affect your social life. Did I get that right?""",
    },
    "SF": {
        "name": "Shifting Focus",
        "definition": "Deflecting resistance by moving away from a stuck or problematic topic to a different topic drawn directly from the patient's memory context (e.g. core values, change plan).",
        "example": """Patient: "[Going off on a tangent about politics]"
Bad Response: "Stop avoiding the subject."
Good Response: "We could debate that all day, but I want to make sure we have time to talk about what you mentioned earlier regarding your core value of family." """,
    },
}

MIIN_TECHNIQUES: dict[str, dict[str, str]] = {
    "ADWP": {
        "name": "Advise Without Permission",
        "definition": "Offers suggestions or guidance WITHOUT asking or receiving permission.",
    },
    "CON": {
        "name": "Confront",
        "definition": "Directly disagrees, argues, corrects, shames, blames, seeks to persuade, criticizes, judges, labels, moralizes, ridicules, or questions the client's honesty.",
    },
    "DIR": {
        "name": "Direct",
        "definition": "Gives an order, command, or direction. The language is imperative.",
    },
    "RCWP": {
        "name": "Raise Concern Without Permission",
        "definition": "Without getting permission, points out a possible problem with a client's goal, plan, or intention.",
    },
    "WA": {
        "name": "Warn",
        "definition": "Provides a warning or threat, implying negative consequences unless the client takes a certain action.",
    },
}


def get_session_objective(session_number: int) -> str:
    """
    Returns the longitudinal session objective constraint to be injected as a SystemMessage.
    Provides just-in-time guidance based on the current session number (1-4).
    """
    if session_number == 1:
        return (
            "### [LONGITUDINAL OBJECTIVE] SESSION 1: ENGAGING & EXPLORING AMBIVALENCE\n"
            "- Goal: Build alliance and explore ambivalence. Expect and safely reflect Sustain Talk to reduce defensiveness.\n"
            "- Target Language: Listen carefully for initial DARN (Desire, Ability, Reason, Need) change talk and amplify it.\n"
            "- Constraint: DO NOT initiate planning or push for behavioral commitments (CAT). Focus solely on exploring the problem."
        )
    elif session_number in (2, 3):
        return (
            f"### [LONGITUDINAL OBJECTIVE] SESSION {session_number}: EVOKING CHANGE TALK\n"
            "- Goal: Actively pull for and amplify DARN change talk. Your job is to build motivation so the patient talks themselves into change.\n"
            "- Target Language: Shift the balance from Sustain Talk to DARN. Watch closely for emergence of CAT (Commitment, Activation, Taking Steps).\n"
            "- Constraint: DO NOT transition to Planning until the patient autonomously produces clear CAT language. If they only use DARN, stay in Evoking."
        )
    elif session_number == 4:
        return (
            "### [LONGITUDINAL OBJECTIVE] SESSION 4: CONSOLIDATION & PLANNING\n"
            "- Goal: Review progress and solidify the change plan.\n"
            "- Target Language: Focus heavily on evoking and reinforcing CAT (Commitment, Action). How will they maintain this?\n"
            "- Constraint: Ensure the patient is doing the planning. Do not prescribe solutions; ask how they plan to overcome obstacles."
        )


def get_technique_definition(chosen_technique_code: str) -> str:
    """
    Returns the technique definition and example for a given technique code.
    """
    technique = MI_TECHNIQUES.get(chosen_technique_code) or MIIN_TECHNIQUES.get(chosen_technique_code)
    if technique is None:
        valid_codes = ", ".join(list(MI_TECHNIQUES.keys()) + list(MIIN_TECHNIQUES.keys()))
        raise ValueError(f"Unknown technique code: {chosen_technique_code}. Expected one of: {valid_codes}")
    
    definition = technique["definition"]
    example = technique.get("example", "")
    if example:
        indented_example = example.replace('\n', '\n      ')
        return f"{definition}\n    Example:\n      {indented_example}"
    return definition


def format_all_techniques_names(available_techniques: List[str]) -> str:
    """
    Builds a consistently formatted techniques section for strategist prompts (names and definitions).
    """
    lines = ["### Motivational Interviewing Techniques:"]
    all_techs = {**MI_TECHNIQUES, **MIIN_TECHNIQUES}
    for code in available_techniques:
        if code in all_techs:
            technique = all_techs[code]
            lines.append(f"- {code} ({technique['name']}): {technique['definition']}")
    return "\n".join(lines)


def generate_clinical_analyzer_prompt() -> str:
    """
    Generates the system prompt for Clinical Analyzer Agent.
    Analyzes the interaction between the previous therapist turn and current patient turn
    to diagnose patient state.
    """
    return """### ROLE:
You are the Clinical Analyzer in a Motivational Enhancement Therapy (MET) simulation.
Your job is to DIAGNOSE the PATIENT's current STAGE OF CHANGE based on their reaction to the therapist's last intervention and the recent conversation history.

### CLINICAL DEFINITIONS:
1. STAGES OF CHANGE:
    - Precontemplation: Not considering change. Defending the status quo.
    - Contemplation: Considering they have a problem and weighing the feasibility/costs of changing. Ambivalent.
    - Action: Actively modifying behavior.

2. PATIENT LANGUAGE CLASSIFICATIONS:
    A. Resistance (Sustain Talk): Defends status quo, argues, interrupts, sidetracks, or minimizes the problem.
    B. Change Talk (Preparatory): Expresses Desire, Ability, Reason, or Need (DARN) to change. Acknowledges the problem but hasn't committed.
    C. Commitment & Action Talk (Mobilizing): Expresses firm intent ("I will", "I swear"), makes concrete plans, or reports actual steps taken.

### INSTRUCTIONS:
Analyze the interaction provided in the `<last_exchange>` tags, focusing heavily on the patient's MOST RECENT turn. Use the preceding `<conversation_history>` to detect temporal patterns like stuckness.

### CONTEXTUAL FLAGS (CHAIN OF THOUGHT):
You must evaluate the following three clinical states. For each, FIRST provide a reasoning trace, and THEN the final boolean conclusion.

1. Planning Readiness:
   - `planning_reasoning_trace`: Evaluate if the patient's current CAT warrants transitioning to planning.
   - `ready_for_planning`: true ONLY IF they demonstrate firm determination.

2. Patient Stuckness:
   - `stuckness_reasoning_trace`: Analyze the `<conversation_history>` to see if the patient is looping on the same sustain talk over multiple turns.
   - `patient_is_stuck`: true IF the trace concludes they are stuck in a loop.

3. Guidance Requests:
   - `guidance_reasoning_trace`: Check if the most recent turn explicitly asks the therapist a question or seeks advice/info.
   - `patient_requests_guidance_or_info`: true IF the trace concludes they are seeking input.

### OUTPUT FORMAT
You must respond ONLY with a valid JSON object. No markdown formatting outside the JSON, no extra text.
All angle-bracket values below are PLACEHOLDERS — replace every one with your own generated content.

{
    "patient_diagnosis": {
        "classification_reasoning": "<why the utterance was classified as DARN / CAT / resistance>",
        "resistance_detected": "<verbatim sustain talk, defensiveness, or minimization — or 'none'>",
        "preparatory_change_talk": "<verbatim DARN utterance — or 'none'>",
        "commitment_and_action_talk": "<verbatim CAT utterance — or 'none'>",
        "stage_of_change": "<Precontemplation | Contemplation | Action>",
        "planning_reasoning_trace": "<step-by-step reasoning on readiness>",
        "ready_for_planning": false,
        "stuckness_reasoning_trace": "<step-by-step reasoning on repetitive loops>",
        "patient_is_stuck": false,
        "guidance_reasoning_trace": "<step-by-step reasoning on requests for info>",
        "patient_requests_guidance_or_info": false
    }
}
"""


def _bullets(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "None"


def _plan_bullets(items: List[Dict[str, Any]]) -> str:
    if not items: return "None"
    return "\n".join(f"- {item.get('intention', '')} (Status: {item.get('attempt_status', '')}) - {item.get('outcome_note', '')}" for item in items)


def generate_strategist_prompt(
    session_number: int,
    biographical_facts: List[str],
    core_values: List[str],
    target_behavior: str,
    darn_change_talk: List[str],
    cat_change_talk: List[str],
    sustain_talk_themes: List[str],
    change_plan: List[Dict[str, Any]],
    current_clinical_focus: str,
    patient_diagnosis: Optional[ClinicalAnalyzerOutput],
    mobilizing_ct_count_this_session: int,
    available_techniques: List[str],
    previous_technique: Optional[str] = None,
) -> str:
    """
    Generates the system prompt for the Clinical Strategist Agent.
    Evaluates context to output Macro_Intent and Technique Constraints.
    """

    active_technique_rules = ""
    if patient_diagnosis is not None:
        diag = patient_diagnosis.patient_diagnosis
        diagnosis_block = (
            f"- Resistance / Sustain Talk: {diag.resistance_detected}\n"
            f"- Preparatory Change Talk (DARN): {diag.preparatory_change_talk}\n"
            f"- Commitment & Action Talk (verbatim): {diag.commitment_and_action_talk}"
        )
        
        rules = []
        
        # State retention overrides other technique constraints
        if previous_technique in ("ADP", "RCP"):
            rules.append("- State retention: You asked for permission to share thoughts/advice in the previous turn. If the current patient turn grants that permission, you must restrict the `allowed_techniques` list exclusively to Giving Information (GI) or Advise With Permission (ADP) so you can actually deliver the advice/concern. If they denied permission or deflected, shift focus.")
        else:
            if diag.patient_requests_guidance_or_info:
                rules.append(f"- Patient Requests Guidance: {diag.guidance_reasoning_trace}\n  -> Action required: You must restrict the `allowed_techniques` list exclusively to active techniques like Giving Information (GI) or Advise With Permission (ADP).")
            if diag.patient_is_stuck:
                rules.append(f"- Patient Is Stuck: {diag.stuckness_reasoning_trace}\n  -> Action required: You must restrict the `allowed_techniques` list exclusively to Raise Concern With Permission (RCP) or Shifting Focus (SF) to gently break the loop.")
                
        if diag.ready_for_planning:
            rules.append(f"- Ready For Planning: {diag.planning_reasoning_trace}\n  -> Action required: You must transition the `current_mi_phase` to Planning.")

        if rules:
            active_technique_rules = "### Active Technique Rules\n" + "\n".join(rules)

    else:
        diagnosis_block = "No prior patient turn — session is opening."

    all_techniques_definitions = format_all_techniques_names(available_techniques)

    return f"""### ROLE:
You are the Clinical Strategist (Supervisor) in a Motivational Enhancement Therapy (MET) simulation.
You analyze the current patient state and session history (provided in the User message) to define the Strategic Intent and set Hard Constraints on what techniques the Formulator may use.
Your goal is to ensure macro-level protocol progression through the 4 MI Phases. You must not let the patient drift aimlessly.

### MI PHASES (PROTOCOL PROGRESSION)
1. Engaging: Building the relational foundation. Focus on empathy, reflective listening, and exploring the patient's perspective. Stay in this phase during the first session, or whenever the patient is defensive, resistant, or building trust.
2. Focusing: Finding a strategic direction. Use this phase when exploring the context of the patient's life, values, or the specific details of the target behavior, without explicitly pressing for change yet.
3. Evoking: Eliciting movement. Resolving ambivalence and drawing out Preparatory Change Talk (DARN). ONLY transition to Evoking when the patient has explicitly acknowledged the target behavior as a problem and the therapeutic alliance is strong. Do not rush to Evoking.
4. Planning: Transition here when the patient is ready to formulate a plan. Do not push planning if the patient is only using DARN.

### OBSERVABLE SIGNALS THIS TURN
- Current Session Number: {session_number}
- Mobilizing-CT count this session so far: {mobilizing_ct_count_this_session}

### MEMORY CONTEXT
Biographical Facts:
{_bullets(biographical_facts)}

Core Values:
{_bullets(core_values)}

Target Behavior: {target_behavior}

Active DARN Change Talk:
{_bullets(darn_change_talk)}

Active CAT Change Talk:
{_bullets(cat_change_talk)}

Active Sustain Talk Themes:
{_bullets(sustain_talk_themes)}

Change Plan Status:
{_plan_bullets(change_plan)}

Current Clinical Focus: {current_clinical_focus}

{all_techniques_definitions}

### PATIENT DIAGNOSIS (Latest Turn Analysis)
{diagnosis_block}

{active_technique_rules}

### INSTRUCTIONS:
Analyze the patient's situation, determine the MI Phase, and output the Macro Intent and Technique Constraints for the Formulator.

### OUTPUT FORMAT
You must respond ONLY with a valid JSON object. No markdown formatting outside the JSON, no extra text.
All angle-bracket values below are PLACEHOLDERS — replace every one with your own generated content.
For `allowed_techniques`, you MUST use the exact short uppercase technique codes (e.g., "OQ", "CR", "SR") provided in the list above. DO NOT use their full names.
{{
    "strategist_reasoning_trace": [
        "<step 1 — diagnosis: analyze the patient's language from Patient Diagnosis>",
        "<step 2 — rules check: acknowledge any Active Technique Rules present above. If present, you must obey them in your output.>",
        "<step 3 — phase map: determine active MI Phase (Engaging/Focusing/Evoking/Planning).>",
        "<step 4 — constraint formulation: specify exactly which techniques you will allow based strictly on step 2 and step 3>"
    ],
    "current_mi_phase": "<exactly one of: Engaging | Focusing | Evoking | Planning>",
    "macro_intent": "<specific clinical goal for this turn>",
    "allowed_techniques": ["<technique_code_1>", "<technique_code_2>"]
}}
"""

def generate_formulator_prompt(
    macro_intent: str,
    allowed_techniques: List[str],
    biographical_facts: List[str],
    core_values: List[str],
    target_behavior: str,
    darn_change_talk: List[str],
    cat_change_talk: List[str],
    sustain_talk_themes: List[str],
    change_plan: List[Dict[str, Any]],
    current_clinical_focus: str,
    patient_diagnosis: Optional[ClinicalAnalyzerOutput],
) -> str:
    """
    Generates the system prompt for the Formulator Agent (The Voice).
    Receives Hard Constraints (Macro_Intent, Allowed techniques) from the
    Clinical Strategist and selects the best allowed technique to formulate the therapist's response.
    """

    allowed_str = "\n\n".join(
        f"- **{code}**: {get_technique_definition(code)}"
        for code in allowed_techniques
    )

    clinical_context_alerts = ""
    if patient_diagnosis is not None:
        diag = patient_diagnosis.patient_diagnosis
        alerts = []
        if diag.patient_requests_guidance_or_info:
            alerts.append(f"- Patient Requests Guidance: {diag.guidance_reasoning_trace}")
        if diag.patient_is_stuck:
            alerts.append(f"- Patient Is Stuck: {diag.stuckness_reasoning_trace}")
        if diag.ready_for_planning:
            alerts.append(f"- Ready For Planning: {diag.planning_reasoning_trace}")
            
        if alerts:
            clinical_context_alerts = "### CLINICAL CONTEXT\n" + "\n".join(alerts)

    return f"""### ROLE:
You are the Formulator (The Voice) of Dr. Smith, an expert in Motivational Enhancement Therapy (MET).
Your objective is to generate the exact verbal response to the patient based STRICTLY on the Strategic Director's constraints.

### STRATEGIST CONSTRAINTS (MANDATORY):
- **Macro Intent:** {macro_intent}
- **Allowed Techniques:** Only use ONE of the following. Your response MUST structurally match the definition:
{allowed_str}

{clinical_context_alerts}

### MEMORY CONTEXT
Biographical Facts:
{_bullets(biographical_facts)}

Core Values:
{_bullets(core_values)}

Target Behavior: {target_behavior}

Active DARN Change Talk:
{_bullets(darn_change_talk)}

Active CAT Change Talk:
{_bullets(cat_change_talk)}

Active Sustain Talk Themes:
{_bullets(sustain_talk_themes)}

Change Plan Status:
{_plan_bullets(change_plan)}

Current Clinical Focus: {current_clinical_focus}

### INSTRUCTIONS:
1. TECHNIQUE SELECTION: Evaluate the CLINICAL CONTEXT and Macro Intent. Choose exactly ONE technique from the Allowed Techniques list above that best addresses the clinical situation. You MUST use one of: {allowed_techniques}. Do not default to simple reflections if the context demands addressing stuckness, readiness, or a request for guidance.
2. HUMAN AUTHENTICITY & WARMTH: Use natural, colloquial language. You may use brief conversational padding or validations (e.g., "I hear you", "That makes sense") to sound human and empathetic. Keep the tone fluid and conversational.
3. MEMORY INTEGRATION: Weave in relevant details from the MEMORY CONTEXT to support the Macro Intent organically.
4. EVIDENCE-BASED BREVITY: In real MI, the therapist maintains a high reflection-to-question ratio and talks significantly less than the patient. Respond with a MAXIMUM of two short sentences (under 50 words). 
5. SINGLE TECHNIQUE FOCUS: Execute exactly ONE technique beautifully per turn. Rely heavily on statements (reflections) and reserve questions only for when explicitly dictated by the chosen technique.
6. MAINTAIN AUTONOMY (ANTI-RIGHTING REFLEX): Guide the patient to discover their own solutions. Only provide information or advice if the chosen technique explicitly demands it AND permission was granted.

### OUTPUT FORMAT
You must respond ONLY with a valid JSON object. No markdown formatting outside the JSON, no extra text.
All angle-bracket values below are PLACEHOLDERS — replace every one with your own generated content.
{{
    "reasoning_chain": [
        "<step 1 — context evaluation: analyze the CLINICAL CONTEXT and Macro Intent to determine which allowed technique is most appropriate>",
        "<step 2 — technique selection: choose one allowed technique based on step 1 and quote its exact definition>",
        "<step 3 — draft: draft an initial response to fulfill the macro intent>",
        "<step 4 — righting reflex check: ensure the draft maintains patient autonomy and avoids unsolicited advice. If it contains fixing or lecturing, rewrite it.>",
        "<step 5 — structural verification: verify the drafted text strictly matches the grammatical definition from step 2 (e.g., SR is a statement without a question mark). If it fails, rewrite it.>",
        "<step 6 — refinement: confirm the response is under 50 words and flows naturally with empathetic warmth.>"
    ],
    "chosen_technique": "<exact code of the selected technique>",
    "final_response": "<spoken words — strictly complying with the chosen technique's structure>"
}}
"""