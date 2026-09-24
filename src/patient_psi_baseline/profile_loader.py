"""
Patient-Ψ Profile Loader and Prompt Builder.

Loads pre-generated Cognitive Conceptualization Diagram (CCD) JSON files from
``patient_psi_profiles/{idx}.json`` and constructs the system prompt matching
the official Patient-Ψ paper specifications (Wang et al., ACL 2024).
"""
import json
import logging
from pathlib import Path
from typing import Optional, Union

from .dtos import PatientPsiCCD, CognitiveModelUnit
from .patient_types import get_patient_psi_style

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parent.parent.parent / "patient_psi_profiles"


def load_patient_psi_profile(patient_idx: Union[int, str]) -> PatientPsiCCD:
    """
    Load a pre-generated Patient-Ψ CCD profile from patient_psi_profiles/{idx}.json.

    Args:
        patient_idx: Integer or string index of the patient.

    Returns:
        PatientPsiCCD validated model.

    Raises:
        FileNotFoundError: If the profile file does not exist.
        ValueError: If the file is invalid JSON or fails schema validation.
    """
    profile_path = PROFILES_DIR / f"{patient_idx}.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Patient-Ψ profile not found at {profile_path}. Ensure pre-generated profiles exist."
        )

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load JSON from {profile_path}: {e}") from e

    return PatientPsiCCD.model_validate(data)


def build_patient_psi_system_prompt(
    ccd: PatientPsiCCD,
    style_name: str = "plain",
    rolling_summary: str = "",
    patient_name: Optional[str] = None,
) -> str:
    """
    Construct the Patient-Ψ system prompt based on Judith Beck's CBT Cognitive
    Conceptualization Diagram and the specified conversational style, matching
    the official Patient-Ψ paper specifications (Wang et al., ACL 2024).

    Args:
        ccd: The patient's Cognitive Conceptualization Diagram.
        style_name: One of 'plain', 'upset', 'verbose', 'reserved', 'tangent', 'pleasing'.
        rolling_summary: Summary from previous sessions if session > 1.
        patient_name: Optional explicit name for the patient.

    Returns:
        Formatted system prompt string.
    """
    style_cfg = get_patient_psi_style(style_name)
    style_content = style_cfg["content"]
    max_sentences = style_cfg["max_sentences"]

    if patient_name:
        name = patient_name
    else:
        # Check if life_history starts with a specific subject name
        words = ccd.life_history.split()
        if len(words) > 2 and words[0].lower() not in ("the", "this", "patient", "client"):
            first_part = ccd.life_history.split(" is ")[0].split(" who")[0].split(",")[0].strip()
            name = first_part if len(first_part.split()) <= 3 else "the patient"
        else:
            name = "the patient"

    # Select primary scenario from cognitive_models
    if ccd.cognitive_models:
        cm = ccd.cognitive_models[0]
        situation = cm.situation
        auto_thoughts = cm.automatic_thoughts
        emotion = cm.emotion
        behavior = cm.behavior
    else:
        situation = "None"
        auto_thoughts = "None"
        emotion = "None"
        behavior = "None"

    core_belief = f"{ccd.core_beliefs} ({ccd.core_belief_description})" if ccd.core_belief_description else ccd.core_beliefs

    # Inter-session memory block (separate section at bottom when present)
    memory_section = ""
    if rolling_summary:
        memory_section = f"\n\nMemory from Previous Sessions:\n{rolling_summary}\n"

    # Guideline 1: uses patientTypeContent directly
    guideline_1 = f"1. {style_content}\n" if style_content else "1. \n"

    prompt = (
        f"Imagine you are {name}, a patient who has been experiencing mental health challenges. "
        f"You have been attending therapy sessions for several weeks. Your task is to engage in a conversation "
        f"with the therapist as {name} would during a cognitive behavioral therapy (CBT) session. "
        f"Align your responses with {name}'s background information provided in the 'Relevant history' section. "
        f"Your thought process should be guided by the cognitive conceptualization diagram in the "
        f"'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real "
        f"patient would not explicitly think in those terms.\n\n"
        f"Patient History: {ccd.life_history}\n\n"
        f"Cognitive Conceptualization Diagram:\n"
        f"Core Beliefs: {core_belief}\n"
        f"Intermediate Beliefs: {ccd.intermediate_beliefs}\n"
        f"Intermediate Beliefs during Depression: {ccd.intermediate_beliefs_during_depression}\n"
        f"Coping Strategies: {ccd.coping_strategies}\n\n"
        f"You will be asked about your experiences over the past week. Engage in a conversation with the therapist "
        f"regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, "
        f"but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be "
        f"informed by the diagram, enabling the therapist to infer your thought processes.\n\n"
        f"Situation: {situation}\n"
        f"Automatic Thoughts: {auto_thoughts}\n"
        f"Emotions: {emotion}\n"
        f"Behavior: {behavior}\n"
        f"{memory_section}\n"
        f"In the upcoming conversation, you will simulate {name} during the therapy session, while the user will play "
        f"the role of the therapist. Adhere to the following guidelines:\n"
        f"{guideline_1}"
        f"2. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. "
        f"Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n"
        f"3. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before "
        f"delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n"
        f"4. Maintain consistency with {name}'s profile throughout the conversation. Ensure that your responses align with "
        f"the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n"
        f"5. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {name}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\n"
        f"You are now {name}. Respond to the therapist's prompts as {name} would, regardless of the specific questions asked. "
        f"Limit each of your responses to a maximum of {max_sentences} sentences. "
        f'If the therapist begins the conversation with a greeting like "Hi," initiate the conversation as the patient.\n\n'
        f"Output format:\n"
        f"You must output a JSON object with:\n"
        f'- "reasoning": A brief 1-2 sentence reflection of your internal feelings and why you are responding this way.\n'
        f'- "response": Your exact spoken words to the counselor as {name} (maximum {max_sentences} sentences).'
    )
    return prompt.strip()
