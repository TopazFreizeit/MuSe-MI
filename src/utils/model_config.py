"""
Model configuration for Ollama and OpenRouter LLM agents.
"""
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os
import logging
from .cost_tracker import cost_tracker

logger = logging.getLogger(__name__)


# Default timeout for all LLM API calls (seconds)
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
# SDK-level retries are intentionally conservative because many call sites
# also use LangChain Runnable.with_retry(), which already applies backoff.
DEFAULT_MAX_RETRIES = int(os.getenv("OPENROUTER_SDK_MAX_RETRIES", "1"))


def get_ollama_model(model_name: str = "qwen2.5:latest", temperature: float = 0.7, max_tokens: int = None, timeout: float = None):
    """
    Create and return an Ollama LLM instance.
    
    Args:
        model_name: Name of the Ollama model to use
        temperature: Temperature for response generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens in the response (optional)
        timeout: Request timeout in seconds (optional, defaults to DEFAULT_REQUEST_TIMEOUT)
    
    Returns:
        ChatOllama instance configured with specified parameters
    """
    kwargs = {}
    if max_tokens is not None:
        kwargs["num_predict"] = max_tokens
    kwargs["timeout"] = timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        keep_alive="15m",
        **kwargs
    )


def get_openrouter_model(
    model_name: str = "tngtech/deepseek-r1t2-chimera:free",
    temperature: float = 0.7,
    provider_preferences: dict = None,
    max_tokens: int = None,
    timeout: float = None,
    logprobs: bool = None,
    top_logprobs: int = None,
):
    """
    Create and return an OpenRouter LLM instance.
    
    Args:
        model_name: Name of the OpenRouter model to use
        temperature: Temperature for response generation (0.0 to 1.0)
        provider_preferences: Optional OpenRouter provider routing preferences,
            e.g. {"order": ["Cerebras", "Groq"], "allow_fallbacks": False}
        max_tokens: Maximum number of tokens in the response (optional)
        timeout: Request timeout in seconds (optional)
        logprobs: Whether to return logprobs (optional)
        top_logprobs: Number of top logprobs to return (optional)
    
    Returns:
        ChatOpenAI instance configured for OpenRouter with specified parameters
    """
    kwargs = {"stream_options": {"include_usage": True}}
    _seed_env = os.getenv("LLM_SEED")
    if _seed_env is not None:
        logger.info(f"LLM seed enabled: {_seed_env}")
    extra_body = {}
    if provider_preferences:
        extra_body["provider"] = provider_preferences

    init_kwargs = {}
    if max_tokens is not None:
        init_kwargs["max_tokens"] = max_tokens
    if _seed_env is not None:
        init_kwargs["seed"] = int(_seed_env)
    if logprobs is not None:
        init_kwargs["logprobs"] = logprobs
    if top_logprobs is not None:
        init_kwargs["top_logprobs"] = top_logprobs
    init_kwargs["request_timeout"] = timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
    init_kwargs["max_retries"] = DEFAULT_MAX_RETRIES

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        include_response_headers=True,
        model_kwargs=kwargs,
        extra_body=extra_body if extra_body else None,
        callbacks=[cost_tracker],
        **init_kwargs,
    )
