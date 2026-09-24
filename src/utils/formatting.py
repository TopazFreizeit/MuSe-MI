"""
Formatting utilities for therapy conversation system.

All functions that format, convert, or build strings for prompts, transcripts,
or message histories live here to ensure consistent output across all modules.
"""
import json
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from .constants import THERAPIST_PREFIX, PATIENT_PREFIX


# ---------------------------------------------------------------------------
# Utterance / transcript line formatting
# ---------------------------------------------------------------------------

def format_speaker_utterance(content: str, prefix: str) -> str:
    """
    Format a single utterance with a speaker prefix.

    Args:
        content: The utterance text.
        prefix: Speaker prefix constant (e.g. THERAPIST_PREFIX, PATIENT_PREFIX).

    Returns:
        ``"{prefix}\\n{content}\\n"``
    """
    return f"{prefix}\n{content}\n"


def format_conversation_history(messages: List[BaseMessage]) -> str:
    """
    Convert a live ``state["messages"]`` list into a plain-text transcript.

    Produces the same line format as :func:`reconstruct_transcript` so that
    the therapist node, summary node, and offline judges all see identical
    formatting.

    Args:
        messages: Named ``BaseMessage`` objects (``msg.name`` == "Therapist" or "Patient").

    Returns:
        Concatenated transcript string using ``THERAPIST_PREFIX`` / ``PATIENT_PREFIX``.
    """
    parts: List[str] = []
    for msg in messages:
        name = getattr(msg, "name", None)
        if name == "Therapist":
            parts.append(f"{THERAPIST_PREFIX}\n{msg.content}\n")
        elif name == "Patient":
            parts.append(f"{PATIENT_PREFIX}\n{msg.content}\n")
    return "".join(parts)


def reconstruct_transcript(turns: List[Dict[str, Any]]) -> str:
    """
    Reconstruct a session transcript from saved ``TurnRecord`` dicts.

    Produces identical output to :func:`format_conversation_history` — both use
    ``THERAPIST_PREFIX`` / ``PATIENT_PREFIX`` with the same newline structure —
    so offline judges and the live graph see the same format.

    Args:
        turns: List of TurnRecord dicts (keys: ``speaker``, ``volley``).

    Returns:
        Concatenated transcript string.
    """
    parts: List[str] = []
    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        prefix = THERAPIST_PREFIX if speaker == "therapist" else PATIENT_PREFIX
        parts.append(f"{prefix}\n{turn.get('volley', '')}\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Patient data formatting
# ---------------------------------------------------------------------------

def format_patient_data(patient_data: dict) -> str:
    """
    Format a patient assessment data dict into a readable multi-line string.

    Used to build the ``patient_profile`` parameter for :func:`generate_met_prompt`.

    Args:
        patient_data: Dict with keys matching the empirical patient dataset schema.

    Returns:
        Multi-line string with labelled fields.
    """
    patient_string_data = ""

    for key, value in patient_data.items():
        patient_string_data += f"{key.replace('_', ' ').title()}: {value}\n"

    return patient_string_data




def format_patient_profile_json(static_profile: dict) -> str:
    """
    Serialize a patient static profile dict to a JSON string for prompt embedding.

    Args:
        static_profile: Dict with keys matching ``static_profile`` in the MET patient dataset.

    Returns:
        Pretty-printed JSON string of the static profile.
    """
    return json.dumps(static_profile, indent=2, ensure_ascii=False)


def format_patient_state_json(initial_dynamic_state: dict, patient_rolling_state: str = "") -> str:
    """
    Resolve and serialize patient dynamic state for prompt embedding.

    Uses the rolling state from the memory node if available (session 2+),
    otherwise serializes the initial ``dynamic_state`` from the dataset (session 1).

    Args:
        initial_dynamic_state: Default dynamic state dict from the patient dataset.
        patient_rolling_memory: Rolling JSON string from the memory node
                               (empty string on session 1).

    Returns:
        JSON string of the resolved dynamic state.
    """
    if patient_rolling_state:
        return json.dumps(json.loads(patient_rolling_state), indent=2, ensure_ascii=False)
    return json.dumps(initial_dynamic_state, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Message perspective transforms (LangChain chat history)
# ---------------------------------------------------------------------------

def transform_messages_for_therapist(messages: List[BaseMessage]) -> str:
    """Format conversation messages into XML-tagged context blocks.

    Separates the conversation into three strict zones:

    - ``<conversation_history>``: All turns before the last therapist-patient exchange
    - ``<previous_therapist_turn>``: The therapist's most recent message
    - ``<current_patient_turn>``: The patient's most recent message

    Args:
        messages: Named ``BaseMessage`` objects from ``ConversationState``.

    Returns:
        XML-tagged conversation context string, or empty string if no messages.
    """
    if not messages:
        return ""

    parts: List[str] = []

    # The last message is always from the patient (therapist is about to respond).
    # The second-to-last is always the previous therapist turn.
    if len(messages) >= 2:
        history_msgs = messages[:-2]
        prev_therapist = messages[-2]
        current_patient = messages[-1]
    else:
        # Only one message: the patient's first response to therapist's opening
        history_msgs = []
        prev_therapist = None
        current_patient = messages[0]

    # Conversation history (older turns)
    if history_msgs:
        history_lines = []
        for msg in history_msgs:
            name = getattr(msg, "name", "Unknown")
            history_lines.append(f"{name}: {msg.content}")
        parts.append(
            "<conversation_history>\n"
            + "\n".join(history_lines)
            + "\n</conversation_history>"
        )

    # Previous therapist turn
    if prev_therapist:
        parts.append(
            "<previous_therapist_turn>\n"
            + prev_therapist.content
            + "\n</previous_therapist_turn>"
        )

    # Current patient turn
    parts.append(
        "<current_patient_turn>\n"
        + current_patient.content
        + "\n</current_patient_turn>"
    )

    return "\n\n".join(parts)


def transform_messages_for_patient(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Reframe conversation history from the patient's perspective.

    - Therapist messages → HumanMessage  (incoming stimulus)
    - Patient messages   → AIMessage     (own prior responses)

    Args:
        messages: Named ``BaseMessage`` list from ``ConversationState``.

    Returns:
        Transformed list for use as LangChain chat history.
    """
    transformed = []
    for msg in messages:
        if hasattr(msg, "name") and msg.name == "Therapist":
            transformed.append(HumanMessage(content=msg.content))
        elif hasattr(msg, "name") and msg.name == "Patient":
            transformed.append(AIMessage(content=msg.content))
    return transformed


# ---------------------------------------------------------------------------
# format therapist memory
# ---------------------------------------------------------------------------
def format_therapist_memory(memory) -> str:
    return memory.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# End of formatting module
# ---------------------------------------------------------------------------



