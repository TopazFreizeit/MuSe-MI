"""
Configuration settings for Therapist Agent.
"""
import random
import os


MIIN_INJECTION_ENABLED = os.getenv("MIIN_INJECTION_ENABLED", "false").lower() == "true"
MIIN_INJECTION_SESSIONS = os.getenv("MIIN_INJECTION_SESSIONS", "1,2,3,4").split(",")
MIIN_INJECTION_MIN_MAX_TURN = os.getenv("MIIN_INJECTION_MIN_MAX_TURN", "7,23").split(",")
MIIN_INJECTION_COUNT = int(os.getenv("MIIN_INJECTION_COUNT", "3"))


def generate_miin_injection_schedule() -> dict[int, list[int]]:
    """
    Pre-compute the MIIN injection schedule for the entire run.

    For each session in MIIN_INJECTION_SESSIONS, randomly selects
    MIIN_INJECTION_COUNT distinct therapist turns
    (odd-numbered in sequential counting: 1, 3, 5, ...) within the
    MIIN_INJECTION_MIN_MAX_TURN range.

    Returns:
        A dict mapping session_number -> sorted list of injection turn numbers.
        Empty dict if MIIN injection is disabled.
    """
    if not MIIN_INJECTION_ENABLED:
        return {}

    min_turn, max_turn = int(MIIN_INJECTION_MIN_MAX_TURN[0]), int(MIIN_INJECTION_MIN_MAX_TURN[1])
    eligible_therapist_turns = [
        turn for turn in range(min_turn, max_turn + 1) if turn % 2 == 1
    ]

    k = min(MIIN_INJECTION_COUNT, len(eligible_therapist_turns))

    miin_injection_schedule: dict[int, list[int]] = {}
    for session_number in MIIN_INJECTION_SESSIONS:
        chosen_turns = sorted(random.sample(eligible_therapist_turns, k=k))
        miin_injection_schedule[int(session_number)] = chosen_turns

    return miin_injection_schedule


# Maximum completion tokens for therapist structured output.
# Prevents runaway generation (LengthFinishReasonError) when
# the model over-generates reasoning in JSON mode.
MAX_COMPLETION_TOKENS = int(os.getenv("THERAPIST_MAX_COMPLETION_TOKENS", "2048"))

# Exponential backoff with jitter for API retries
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "4"))


class TherapistConfigClass:
    """Singleton configuration for therapist agent."""
    
    _instance = None
    _temperature: float | None = None
    
    if DEVELOPMENT_MODE := os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
        PROVIDER = "openrouter"
        MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
        FALLBACK_MODEL_NAME = "google/gemini-2.0-flash-001"
        OPENROUTER_PROVIDER_PREFS = {
            "order": ["akashml/fp8", "inceptron/fp8", "novita/bf16"],
            "allow_fallbacks": True,
        }
    else:
        PROVIDER = "openrouter"
        MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
        FALLBACK_MODEL_NAME = "google/gemini-2.0-flash-001"
        OPENROUTER_PROVIDER_PREFS = {
            "order": ["akashml/fp8", "inceptron/fp8", "novita/bf16"],
            "allow_fallbacks": True,
        }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_temperature(self) -> float:
        """
        Get therapist temperature (singleton pattern).
        """
        if self._temperature is None:
            self._temperature = float(os.getenv("THERAPIST_TEMPERATURE", "0.7"))
        return self._temperature
    
    def reset_temperature(self) -> None:
        """Reset temperature (useful for testing)."""
        self._temperature = None


# Module-level singleton instance
_config = TherapistConfigClass()

CONTEXT_WINDOW = 6