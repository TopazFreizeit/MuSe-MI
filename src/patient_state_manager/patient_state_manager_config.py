"""
Configuration settings for Patient State Manager (latent variable updates).
"""
import os

if DEVELOPMENT_MODE := os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
    PROVIDER = "openrouter"
    MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
    FALLBACK_MODEL_NAME = "google/gemini-2.0-flash-001"
    TEMPERATURE = 0.7
    OPENROUTER_PROVIDER_PREFS = {
        "order": ["akashml/fp8", "inceptron/fp8", "novita/bf16"],
        "allow_fallbacks": True,
    }
else:
    PROVIDER = "openrouter"
    MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
    FALLBACK_MODEL_NAME = "google/gemini-2.0-flash-001"
    TEMPERATURE = 0.7
    OPENROUTER_PROVIDER_PREFS = {
        "order": ["akashml/fp8", "inceptron/fp8", "novita/bf16"],
        "allow_fallbacks": True,
    }

# Exponential backoff with jitter for API retries
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))

# Enable or disable individual latent variable updates
ENABLE_READINESS = os.getenv("ENABLE_READINESS", "true").lower() == "true"
ENABLE_SELF_EFFICACY = os.getenv("ENABLE_SELF_EFFICACY", "true").lower() == "true"
ENABLE_REACTANCE = os.getenv("ENABLE_REACTANCE", "true").lower() == "true"
ENABLE_PERCEIVED_DISCREPANCY = os.getenv("ENABLE_PERCEIVED_DISCREPANCY", "true").lower() == "true"