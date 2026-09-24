"""
Configuration settings for the Between-Session Event Generator.
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

API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "4"))

ENABLE_BETWEEN_SESSION_EVENTS = os.getenv("ENABLE_BETWEEN_SESSION_EVENTS", "true").lower() == "true"
