"""
LangChain callback handler that logs per-call token usage and cost,
and accumulates a run-wide total for final reporting.
"""
import logging
import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# Per-million-token pricing (input_price, output_price) in USD.
# Source: OpenRouter model pages — update when pricing changes.
_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4.1-nano":          (0.10, 0.40),
    "openai/gpt-4o-mini":           (0.15, 0.60),
    "google/gemini-2.0-flash-001":      (0.15, 0.60),
    "google/gemini-2.0-flash-001-lite": (0.075, 0.30),
    "openai/gpt-oss-120b":          (1.10, 4.40),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.38),
    "qwen/qwen3-235b-a22b-07-25": (0.08, 0.1),
    "google/gemma-4-31b-it": (0.14, 0.4)
}


class CostTracker(BaseCallbackHandler):
    """Thread-safe singleton that tracks token usage and cost across all LLM calls."""

    _instance: "CostTracker | None" = None
    _init_lock = threading.Lock()

    _lock: threading.Lock
    _total_input_tokens: int
    _total_output_tokens: int
    _total_cost: float
    _call_count: int
    _failed_calls: int

    def __new__(cls) -> "CostTracker":
        with cls._init_lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._lock = threading.Lock()
                obj._total_input_tokens = 0
                obj._total_output_tokens = 0
                obj._total_cost = 0.0
                obj._call_count = 0
                obj._failed_calls = 0
                cls._instance = obj
        return cls._instance

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            llm_out = response.llm_output or {}
            usage = llm_out.get("token_usage") or {}
            model = llm_out.get("model_name", "unknown")

            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))

            if model in _PRICING:
                input_price, output_price = _PRICING[model]
                cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
                cost_str = f"${cost:.6f}"
            else:
                cost = 0.0
                cost_str = "N/A (unknown model)"

            with self._lock:
                self._total_input_tokens += prompt_tokens
                self._total_output_tokens += completion_tokens
                self._total_cost += cost
                self._call_count += 1

            logger.info(
                f"LLM call #{self._call_count} | model={model} | "
                f"in={prompt_tokens} out={completion_tokens} tokens | cost={cost_str}"
            )
        except Exception:
            logger.debug("CostTracker failed to process LLM response", exc_info=True)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:  # type: ignore[override]
        with self._lock:
            self._failed_calls += 1

        logger.debug(
            "LLM call failed | run_id=%s parent_run_id=%s tags=%s error=%s",
            run_id,
            parent_run_id,
            tags,
            repr(error),
        )

    def log_total(self) -> None:
        """Log the accumulated totals. Call once at end of run."""
        with self._lock:
            calls = self._call_count
            failed = self._failed_calls
            in_tok = self._total_input_tokens
            out_tok = self._total_output_tokens
            cost = self._total_cost

        logger.info(
            f"Run total | calls={calls} failed={failed} | "
            f"in={in_tok:,} out={out_tok:,} tokens | "
            f"total_cost=${cost:.4f}"
        )

    def reset(self) -> None:
        """Reset all counters (call before a fresh run if reusing the process)."""
        with self._lock:
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_cost = 0.0
            self._call_count = 0
            self._failed_calls = 0


cost_tracker = CostTracker()
