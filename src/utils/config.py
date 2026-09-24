"""
Configuration for selecting LLM provider (Ollama or OpenRouter).
"""
import os
import logging
from dotenv import load_dotenv
from .model_config import get_ollama_model, get_openrouter_model

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Model configurations by provider
PROVIDER_MODELS = {
    "ollama": "qwen2.5:latest",
    "openrouter": "qwen/qwen3-next-80b-a3b-instruct:free"
}

# Ablation Study Flags
ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES = os.getenv("ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES", "false").lower() == "true"
DISABLE_COGNITIVE_INTERPRETER_IMPACT = os.getenv("DISABLE_COGNITIVE_INTERPRETER_IMPACT", "false").lower() == "true"
DISABLE_PATIENT_MEMORY = os.getenv("DISABLE_PATIENT_MEMORY", "false").lower() == "true"
ABLATE_LIFE_EVENT_SIMULATOR = os.getenv("ABLATE_LIFE_EVENT_SIMULATOR", "false").lower() == "true"
CONSISTENT_CLIENT_BASELINE = os.getenv("CONSISTENT_CLIENT_BASELINE", "false").lower() == "true"
CAMI_THERAPIST_BASELINE = os.getenv("CAMI_THERAPIST_BASELINE", "false").lower() == "true"
VANILLA_BASELINE = os.getenv("VANILLA_BASELINE", "false").lower() == "true"
SIMPATIENT_BASELINE = os.getenv("SIMPATIENT_BASELINE", "false").lower() == "true"
PATIENT_PSI_BASELINE = os.getenv("PATIENT_PSI_BASELINE", "false").lower() == "true"
PATIENT_PSI_STYLE = os.getenv("PATIENT_PSI_STYLE", "plain").lower()
SHIFT_DELTA_MULTIPLIER = float(os.getenv("SHIFT_DELTA_MULTIPLIER", "1.0"))

# Causal Desynchronization & Permutation Experiment Flags
CAUSAL_RECORD_BASELINE = os.getenv("CAUSAL_RECORD_BASELINE", "false").lower() == "true"
CAUSAL_FIXED_CONTEXT_PROBE = os.getenv("CAUSAL_FIXED_CONTEXT_PROBE", "false").lower() == "true"
CAUSAL_SHUFFLED_ROLLOUT = os.getenv("CAUSAL_SHUFFLED_ROLLOUT", "false").lower() == "true"
CAUSAL_ROLLOUT_SEED = int(os.getenv("CAUSAL_ROLLOUT_SEED", "42"))
CAUSAL_BASELINE_DIR = os.getenv("CAUSAL_BASELINE_DIR", "data/causal_study/baseline_traces")
CAUSAL_PERMUTED_DIR = os.getenv("CAUSAL_PERMUTED_DIR", "data/causal_study/permuted_results")
CAUSAL_PROBE_SEEDS = [int(s.strip()) for s in os.getenv("CAUSAL_PROBE_SEEDS", "42,123,999").split(",") if s.strip()]
OPENROUTER_LOGPROB_PROVIDERS = [s.strip() for s in os.getenv("OPENROUTER_LOGPROB_PROVIDERS", "Parasail").split(",") if s.strip()]


def get_llm_model(
    temperature: float = 0.7,
    provider: str = None,
    model_name: str = None,
    provider_preferences: dict = None,
    max_tokens: int = None,
    timeout: float = None,
    logprobs: bool = None,
    top_logprobs: int = None,
):
    """
    Get the configured LLM model based on provider setting.
    
    Args:
        temperature: Temperature for response generation (0.0 to 1.0)
        provider: LLM provider ("ollama" or "openrouter"). If None, uses LLM_PROVIDER env var
        model_name: Model name to use. If None, uses default for provider
        provider_preferences: Optional OpenRouter provider routing preferences,
            e.g. {"order": ["Cerebras", "Groq"], "allow_fallbacks": False}
        max_tokens: Maximum number of tokens in the response (optional)
        timeout: Request timeout in seconds (optional)
        logprobs: Whether to return log probabilities (optional)
        top_logprobs: Number of top log probabilities to return (optional)
    
    Returns:
        Configured LLM instance (either Ollama or OpenRouter)
    
    Raises:
        ValueError: If provider is not "ollama" or "openrouter"
    """
    selected_provider = provider.lower() if provider else os.getenv("LLM_PROVIDER", "ollama").lower()
    if selected_provider not in ["ollama", "openrouter"]:
        raise ValueError(
            f"Invalid provider: {selected_provider}. Must be 'ollama' or 'openrouter'."
        )
    
    logger.debug(f"Using LLM Provider: {selected_provider} | Model: {model_name} | Temperature: {temperature}")
    
    if selected_provider == "ollama":
        return get_ollama_model(model_name, temperature, max_tokens, timeout)
    elif selected_provider == "openrouter":
        return get_openrouter_model(
            model_name,
            temperature,
            provider_preferences,
            max_tokens,
            timeout,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )


def get_current_provider() -> str:
    """
    Get the currently configured LLM provider.
    
    Returns:
        The name of the current provider ("ollama" or "openrouter")
    """
    return os.getenv("LLM_PROVIDER", "ollama").lower()


def get_current_model() -> str:
    """
    Get the model name for the current provider.
    
    Returns:
        The model name string
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    return PROVIDER_MODELS.get(provider, "unknown")
