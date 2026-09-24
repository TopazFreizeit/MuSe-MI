"""
Configuration settings for Patient Agent.
"""
import os


# Default patient index (maps to idx in consistent_client_profiles/profiles.jsonl)
PATIENT_IDX = int(os.getenv("PATIENT_IDX", "0"))

# Context window size in turns (1 turn = 1 therapist msg + 1 patient msg)
PATIENT_CONTEXT_WINDOW_TURNS = int(os.getenv("PATIENT_CONTEXT_WINDOW_TURNS", "6"))

# Exponential backoff with jitter for API retries
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "4"))


class PatientConfigClass:
    """Singleton configuration for patient agent."""
    
    _instance = None
    _temperature: float | None = None
    
    
    if DEVELOPMENT_MODE := os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
        PROVIDER = "openrouter"
        MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
        FALLBACK_MODEL_NAME = "google/gemini-2.0-flash-001-lite"
        OPENROUTER_PROVIDER_PREFS = {
            "order": ["akashml", "inceptron", "novita"],
            "allow_fallbacks": True,
        }
    else:
        PROVIDER = "openrouter"
        MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
        FALLBACK_MODEL_NAME = "google/gemini-2.0-flash-001-lite"
        OPENROUTER_PROVIDER_PREFS = {
            "order": ["akashml", "inceptron", "novita"],
            "allow_fallbacks": True,
        }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_temperature(self) -> float:
        """
        Get patient temperature (singleton pattern).
        """
        if self._temperature is None:
            self._temperature = float(os.getenv("PATIENT_TEMPERATURE", "0.7"))
        return self._temperature
    
    def reset_temperature(self) -> None:
        """Reset temperature (useful for testing)."""
        self._temperature = None

    def get_provider_preferences(self) -> dict:
        """Get provider routing preferences, pinning logprob providers when in causal baseline recording."""
        from ..utils.config import CAUSAL_RECORD_BASELINE, OPENROUTER_LOGPROB_PROVIDERS
        if CAUSAL_RECORD_BASELINE:
            return {
                "order": OPENROUTER_LOGPROB_PROVIDERS,
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        return self.OPENROUTER_PROVIDER_PREFS


# Module-level singleton instance
_config = PatientConfigClass()
