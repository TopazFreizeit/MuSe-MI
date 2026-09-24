"""
Span enrichment utilities for Phoenix / OpenTelemetry observability.

Provides two main tools:

1. ``@traced("node_name")`` – decorator for LangGraph node functions that
   **enriches the existing span** created by ``LangChainInstrumentor`` with
   ``therapy.*`` attributes extracted from the LangGraph state, and sets
   **OpenInference context attributes** (``session_id``, ``metadata``,
   ``tags``) via ``using_attributes`` so child LLM spans inherit
   human-readable metadata in Phoenix.

   Why enrich rather than create? ``LangChainInstrumentor`` already creates
   a CHAIN span for every node.  Creating an additional nested span with
   the same name produced **duplicate entries** in Phoenix — one showing the
   raw ``MultiSessionState`` (from the outer LangChain span) and one with
   the structured ``therapy.*`` attributes (from the inner OTel span).
   Enriching the existing span gives one clean entry per node.

2. ``enrich_span(...)`` – fire-and-forget helper that adds custom attributes
   (model provider, purpose, output summary, …) to the **current** span
   *and* updates the OpenInference context so that any subsequent child
   spans also carry the metadata.
   Works best when called inside a ``@traced`` function for attributes that
   cannot be auto-extracted from state.

Usage inside any LangGraph node::

    from src.utils.tracing import traced, enrich_span

    @traced("therapist")
    def my_node(state, config):
        enrich_span(
            node_name="therapist",
            description="Therapist generates response",
            phase="conversation",
        )
        ...
"""
import contextvars
import functools
import json as _json
import logging
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from openinference.instrumentation import using_attributes, suppress_tracing
from openinference.semconv.trace import SpanAttributes

from ..therapist import therapist_config as therapist_config_module
from .config import (
    ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES,
    DISABLE_COGNITIVE_INTERPRETER_IMPACT,
    DISABLE_PATIENT_MEMORY,
    VANILLA_BASELINE,
    CONSISTENT_CLIENT_BASELINE,
    SIMPATIENT_BASELINE,
    PATIENT_PSI_BASELINE,
    CAUSAL_RECORD_BASELINE,
    CAUSAL_FIXED_CONTEXT_PROBE,
    CAUSAL_SHUFFLED_ROLLOUT,
    CAUSAL_ROLLOUT_SEED,
)

logger = logging.getLogger(__name__)

# ── Context variables for logging context propagation ───────────────────
# These are set by the @traced decorator and read by SessionTurnFilter
# so that every log record automatically includes session/turn info.
_current_session: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "current_session", default=None
)
_current_turn: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "current_turn", default=None
)


class SessionTurnFilter(logging.Filter):
    """Logging filter that injects session and turn number into every log record.

    Add this filter to your logging handlers and use ``%(session)s`` /
    ``%(turn)s`` in the format string.  When session or turn is unknown
    the placeholder renders as ``-``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        session = _current_session.get()
        turn = _current_turn.get()
        record.session = str(session) if session is not None else "-"  # type: ignore[attr-defined]
        record.turn = str(turn) if turn is not None else "-"  # type: ignore[attr-defined]
        return True

# ── Project-level tracer (creates real spans visible in Phoenix) ────────
_tracer = trace.get_tracer("therapy.agent", "1.0.0")

# Public alias – use this in entry-point scripts to create a root span
# that parents all node spans under a single trace in Phoenix.
tracer = _tracer

_phoenix_tracer_provider: Optional[Any] = None


def setup_phoenix_tracing(
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Any:
    """
    Initialize Phoenix OTel registration and auto-instrument OpenAI calls.
    Safe to call multiple times (idempotent).
    """
    global _phoenix_tracer_provider
    if _phoenix_tracer_provider is not None:
        return _phoenix_tracer_provider

    current_tp = trace.get_tracer_provider()
    if (
        current_tp.__class__.__name__ == "TracerProvider"
        and "phoenix" in current_tp.__class__.__module__
    ):
        _phoenix_tracer_provider = current_tp
        return _phoenix_tracer_provider

    import os
    from phoenix.otel import register
    from openinference.instrumentation.openai import OpenAIInstrumentor

    proj = project_name or os.getenv("LANGCHAIN_PROJECT", "Baseline")
    port = os.getenv("PHOENIX_PORT", "6006")
    ep = endpoint or f"http://localhost:{port}/v1/traces"

    try:
        _phoenix_tracer_provider = register(project_name=proj, endpoint=ep)
        OpenAIInstrumentor().instrument(tracer_provider=_phoenix_tracer_provider)
        logger.info(f"Phoenix tracing registered for project '{proj}' at {ep}")
        return _phoenix_tracer_provider
    except Exception as exc:
        logger.warning(f"Could not initialize Phoenix tracing: {exc}")
        return None


def flush_phoenix_tracing() -> None:
    """Flush any pending spans to the Phoenix collector."""
    global _phoenix_tracer_provider
    if _phoenix_tracer_provider is not None and hasattr(_phoenix_tracer_provider, "force_flush"):
        try:
            _phoenix_tracer_provider.force_flush()
        except Exception as exc:
            logger.debug(f"Failed to flush Phoenix tracer provider: {exc}")

# ── Attribute key constants (namespaced to avoid collisions) ────────────
_PREFIX = "therapy"

ATTR_NODE_NAME = f"{_PREFIX}.node.name"
ATTR_NODE_DESCRIPTION = f"{_PREFIX}.node.description"
ATTR_NODE_PURPOSE = f"{_PREFIX}.node.purpose"
ATTR_SESSION_NUMBER = f"{_PREFIX}.session.number"
ATTR_SESSION_TURN = f"{_PREFIX}.session.turn"
ATTR_SPEAKER = f"{_PREFIX}.speaker"
ATTR_SESSION_PHASE = f"{_PREFIX}.session.phase"
ATTR_MODEL_PROVIDER = f"{_PREFIX}.model.provider"
ATTR_MODEL_NAME = f"{_PREFIX}.model.name"
ATTR_OUTPUT_SUMMARY = f"{_PREFIX}.output.summary"
ATTR_MIIN_INJECTION = f"{_PREFIX}.miin_injection"
ATTR_THREAD_ID = f"{_PREFIX}.thread_id"
ATTR_DOMINANT_STAGE = f"{_PREFIX}.patient.dominant_stage"
ATTR_EMERGING_STAGE = f"{_PREFIX}.patient.emerging_stage"
ATTR_READINESS_BUFFER_LEN = f"{_PREFIX}.patient.readiness_buffer_len"
ATTR_READINESS_PRE_DELTA = f"{_PREFIX}.patient.readiness_pre_delta"
ATTR_READINESS_CON_DELTA = f"{_PREFIX}.patient.readiness_con_delta"
ATTR_READINESS_ACT_DELTA = f"{_PREFIX}.patient.readiness_act_delta"
ATTR_READINESS_MNT_DELTA = f"{_PREFIX}.patient.readiness_mnt_delta"
ATTR_SHARED_FACTS_COUNT = f"{_PREFIX}.memory.shared_facts_count"
ATTR_CHANGE_PLAN_ITEMS_COUNT = f"{_PREFIX}.memory.change_plan_items_count"
ATTR_CORE_CONCERNS_COUNT = f"{_PREFIX}.memory.core_concerns_count"
ATTR_CORE_MOTIVATIONS_COUNT = f"{_PREFIX}.memory.core_motivations_count"
ATTR_CORE_CONCERNS_ADDED = f"{_PREFIX}.memory.core_concerns_added"
ATTR_CORE_CONCERNS_EDITED = f"{_PREFIX}.memory.core_concerns_edited"
ATTR_CORE_CONCERNS_RETIRED = f"{_PREFIX}.memory.core_concerns_retired"
ATTR_CORE_MOTIVATIONS_ADDED = f"{_PREFIX}.memory.core_motivations_added"
ATTR_CORE_MOTIVATIONS_EDITED = f"{_PREFIX}.memory.core_motivations_edited"
ATTR_CORE_MOTIVATIONS_RETIRED = f"{_PREFIX}.memory.core_motivations_retired"
ATTR_BSE_TRIGGER_CLASS = f"{_PREFIX}.bse.trigger_class"
ATTR_BSE_SELF_EFFICACY = f"{_PREFIX}.bse.self_efficacy"
ATTR_BSE_MOTIVATIONAL_READINESS = f"{_PREFIX}.bse.motivational_readiness"
ATTR_DIAGNOSIS_SUMMARY = f"{_PREFIX}.therapist.diagnosis_summary"
ATTR_NEXT_SESSION_PRIORITY = f"{_PREFIX}.therapist.next_session_priority"
ATTR_MR_IMPACT = f"{_PREFIX}.patient.mr_impact"
ATTR_ANGER_IMPACT = f"{_PREFIX}.patient.anger_impact"
ATTR_SE_IMPACT = f"{_PREFIX}.patient.se_impact"
ATTR_PR_IMPACT = f"{_PREFIX}.patient.pr_impact"
ATTR_IMPLICIT_THREADS_COUNT = f"{_PREFIX}.memory.implicit_threads_count"
ATTR_BIOGRAPHICAL_FACTS_COUNT = f"{_PREFIX}.memory.biographical_facts_count"
ATTR_CORE_VALUES_COUNT = f"{_PREFIX}.memory.core_values_count"
ATTR_DARN_CHANGE_TALK_COUNT = f"{_PREFIX}.memory.darn_change_talk_count"
ATTR_CAT_CHANGE_TALK_COUNT = f"{_PREFIX}.memory.cat_change_talk_count"
ATTR_SUSTAIN_TALK_THEMES_COUNT = f"{_PREFIX}.memory.sustain_talk_themes_count"
ATTR_CURRENT_CLINICAL_FOCUS = f"{_PREFIX}.therapist.current_clinical_focus"
ATTR_SIMPATIENT_CONTROL = f"{_PREFIX}.simpatient.control"
ATTR_SIMPATIENT_EFFICACY = f"{_PREFIX}.simpatient.efficacy"
ATTR_SIMPATIENT_AWARENESS = f"{_PREFIX}.simpatient.awareness"
ATTR_SIMPATIENT_REWARD = f"{_PREFIX}.simpatient.reward"
ATTR_PATIENT_PSI_STYLE = f"{_PREFIX}.patient_psi.style"
ATTR_TOKEN_PERPLEXITY = f"{_PREFIX}.patient.token_perplexity"
ATTR_AVG_LOGPROB = f"{_PREFIX}.patient.avg_logprob"
ATTR_TOTAL_LOGPROB = f"{_PREFIX}.patient.total_logprob"
ATTR_CAUSAL_SEED = f"{_PREFIX}.causal_probe.seed"
ATTR_CAUSAL_PATIENT_ID = f"{_PREFIX}.causal_probe.patient_id"
ATTR_CAUSAL_FACTUAL_PERPLEXITY = f"{_PREFIX}.causal_probe.factual_perplexity"
ATTR_CAUSAL_FACTUAL_AVG_LOGPROB = f"{_PREFIX}.causal_probe.factual_avg_logprob"
ATTR_CAUSAL_DELTA_PERPLEXITY = f"{_PREFIX}.causal_probe.delta_perplexity"
ATTR_CAUSAL_DELTA_AVG_LOGPROB = f"{_PREFIX}.causal_probe.delta_avg_logprob"

_NEXT_TURN_SPEAKERS = {
    "therapist": "therapist",
    "therapist_clinical_analyzer": "therapist",
    "therapist_strategist": "therapist",
    "therapist_formulator": "therapist",
    "cami_therapist": "therapist",
    "adversarial_therapist": "therapist",
    "patient": "patient",
    "consistent_client": "patient",
    "vanilla_client": "patient",
    "simpatient_client": "patient",
    "patient_psi_client": "patient",
    "causal_shuffled_bank": "patient",
    "fixed_context_probe_patient": "patient",
}


# ── Helpers for automatic state context extraction ──────────────────────

def _extract_state_context(state: Any, node_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract well-known keys from a LangGraph state dict.

    Handles both ``ConversationState`` (single-session, flat keys) and
    ``MultiSessionState`` (parent graph with nested ``current_session``).

    Returns a dict with extracted values (may be empty).
    """
    ctx: Dict[str, Any] = {}
    if not isinstance(state, dict):
        return ctx

    try:
        # ── Direct keys (ConversationState) ─────────────────────────
        session_number = state.get("session_number") or state.get("current_session_number")
        turn_count = state.get("turn_count")
        speaker = state.get("current_speaker")

        # ── Nested keys (MultiSessionState → current_session) ──────
        nested = state.get("current_session")
        if isinstance(nested, dict):
            if session_number is None:
                session_number = nested.get("session_number")
            if turn_count is None:
                turn_count = nested.get("turn_count")
            if speaker is None:
                speaker = nested.get("current_speaker")

        if session_number is not None:
            ctx["session_number"] = int(session_number)
        if turn_count is not None:
            ctx["turn_count"] = int(turn_count)
        if speaker:
            ctx["speaker"] = str(speaker)

        next_speaker = _NEXT_TURN_SPEAKERS.get(node_name or "")
        if next_speaker is not None:
            ctx["speaker"] = next_speaker
            ctx["turn_count"] = int(turn_count or 0) + 1
    except Exception:
        logger.debug("Failed to extract state context", exc_info=True)

    return ctx


def _set_span_attributes(span: trace.Span, ctx: Dict[str, Any]) -> None:
    """Set therapy.* OTel attributes on *span* from an extracted context dict."""
    if ctx.get("session_number") is not None:
        span.set_attribute(ATTR_SESSION_NUMBER, ctx["session_number"])
    if ctx.get("turn_count") is not None:
        span.set_attribute(ATTR_SESSION_TURN, ctx["turn_count"])
    if ctx.get("speaker") is not None:
        span.set_attribute(ATTR_SPEAKER, ctx["speaker"])


def _build_oi_metadata(node_name: str, ctx: Dict[str, Any]) -> Dict[str, str]:
    """
    Build a metadata dict for ``using_attributes(metadata=...)``.

    These key-value pairs appear in Phoenix's **metadata** column on every
    child span (including auto-instrumented ChatCompletion calls).
    """
    meta: Dict[str, str] = {
        "node": node_name,
        "ablation.static_hidden_variables": str(ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES),
        "ablation.disable_cognitive_interpreter_impact": str(DISABLE_COGNITIVE_INTERPRETER_IMPACT),
        "ablation.disable_patient_memory": str(DISABLE_PATIENT_MEMORY),
        "ablation.vanilla_baseline": str(VANILLA_BASELINE),
        "ablation.consistent_client_baseline": str(CONSISTENT_CLIENT_BASELINE),
        "ablation.simpatient_baseline": str(SIMPATIENT_BASELINE),
        "ablation.patient_psi_baseline": str(PATIENT_PSI_BASELINE),
        "ablation.causal_record_baseline": str(CAUSAL_RECORD_BASELINE),
        "ablation.causal_fixed_context_probe": str(CAUSAL_FIXED_CONTEXT_PROBE),
        "ablation.causal_shuffled_rollout": str(CAUSAL_SHUFFLED_ROLLOUT),
        "ablation.causal_rollout_seed": str(CAUSAL_ROLLOUT_SEED),
        "miin_injection.enabled": str(therapist_config_module.MIIN_INJECTION_ENABLED),
        "miin_injection.sessions": ",".join(therapist_config_module.MIIN_INJECTION_SESSIONS),
        "miin_injection.min_max_turn": ",".join(therapist_config_module.MIIN_INJECTION_MIN_MAX_TURN),
        "miin_injection.count": str(therapist_config_module.MIIN_INJECTION_COUNT),
    }
    if ctx.get("session_number") is not None:
        meta["session"] = str(ctx["session_number"])
    if ctx.get("turn_count") is not None:
        meta["turn"] = str(ctx["turn_count"])
    if ctx.get("speaker") is not None:
        meta["speaker"] = ctx["speaker"]
    return meta


def _build_oi_tags(node_name: str, ctx: Dict[str, Any]) -> list[str]:
    """Build a tag list for ``using_attributes(tags=...)``."""
    tags = [node_name]
    if DISABLE_COGNITIVE_INTERPRETER_IMPACT:
        tags.append("ablation-disable-cog-interpreter")
    if DISABLE_PATIENT_MEMORY:
        tags.append("ablation-disable-patient-memory")
    if CAUSAL_RECORD_BASELINE:
        tags.append("causal-record-baseline")
    if CAUSAL_FIXED_CONTEXT_PROBE:
        tags.append("causal-fixed-context-probe")
    if CAUSAL_SHUFFLED_ROLLOUT:
        tags.append("causal-shuffled-rollout")
    if therapist_config_module.MIIN_INJECTION_ENABLED:
        tags.append("miin-injection-enabled")
    if ctx.get("session_number") is not None:
        tags.append(f"session-{ctx['session_number']}")
    if ctx.get("speaker") is not None:
        tags.append(ctx["speaker"])
    return tags


def _build_session_id(ctx: Dict[str, Any]) -> Optional[str]:
    """Build a Phoenix session ID string (groups spans in the Sessions tab)."""
    sn = ctx.get("session_number")
    return f"session-{sn}" if sn is not None else None


# ── Decorator: wraps a node function in an explicit OTel span ───────────

def traced(span_name: str = None):
    """
    Decorator that enriches the current OTel span (created by
    ``LangChainInstrumentor``) with therapy-specific attributes and sets
    OpenInference context so child spans inherit session metadata.

    **Why no new span?** ``LangChainInstrumentor`` already creates a CHAIN
    span for every LangGraph node.  Creating an additional nested span with
    the same name produced duplicate entries in Phoenix — one showing the raw
    ``MultiSessionState`` (from the outer LangChain span) and one showing the
    structured ``therapy.*`` attributes (from the inner OTel span).  By
    enriching the existing span instead, each node appears exactly once in
    Phoenix with all attributes visible in the same entry.

    **OpenInference context** – Sets ``session_id``, ``metadata``, and
    ``tags`` via :func:`using_attributes` so that auto-instrumented child
    spans (ChatCompletion, LangChain chain runs) inherit human-readable
    metadata visible in Phoenix's Spans, Traces, and Sessions views.

    **Error handling** – If the wrapped function raises an exception the
    decorator records it on the span (``span.record_exception``) and sets
    the span status to ``ERROR`` before re-raising.

    **Context injection** – The first positional argument is assumed to be
    the LangGraph ``state`` dict.  The decorator auto-extracts
    ``session_number`` / ``current_session_number``, ``turn_count``, and
    ``current_speaker`` and sets them as ``therapy.*`` span attributes.

    Combine with :func:`enrich_span` inside the function body for
    additional attributes (model provider, output summary, purpose, …).

    Example::

        @traced("therapist")
        def therapist_node(state, config):
            enrich_span(node_name="therapist", ...)
            ...

        # span_name defaults to function name when omitted
        @traced()
        def my_helper(state, config):
            ...

    Args:
        span_name: Identifier used as the ``therapy.node.name`` attribute.
                   Defaults to the decorated function's ``__name__``.
    """
    def decorator(fn):
        _name = span_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Extract state context from the LangGraph state dict
            ctx = _extract_state_context(args[0], _name) if args else {}

            # Propagate session/turn to logging context vars so every
            # log record emitted inside this node includes them.
            _session_token = _current_session.set(ctx.get("session_number"))
            _turn_token = _current_turn.set(ctx.get("turn_count"))

            # Build OpenInference attributes that propagate to child spans
            oi_metadata = _build_oi_metadata(_name, ctx)
            oi_tags = _build_oi_tags(_name, ctx)
            oi_session_id = _build_session_id(ctx)

            # Enrich the *existing* span (created by LangChainInstrumentor)
            # rather than creating a nested duplicate.  This keeps one span
            # per node in Phoenix with both the LangChain input/output attrs
            # and the custom therapy.* attrs visible in the same entry.
            span = trace.get_current_span()
            if span is not None and span.is_recording():
                span.set_attribute(ATTR_NODE_NAME, _name)
                _set_span_attributes(span, ctx)

            # Wrap execution in using_attributes so that ALL child spans
            # (auto-instrumented LLM calls, chain runs, etc.) inherit
            # session_id, metadata, and tags in Phoenix.
            if "misc" in _name.lower():
                ctx_mgr = suppress_tracing()
            else:
                ctx_mgr = using_attributes(
                    session_id=oi_session_id,
                    metadata=oi_metadata,
                    tags=oi_tags,
                )

            with ctx_mgr:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if span is not None and span.is_recording():
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                finally:
                    _current_session.reset(_session_token)
                    _current_turn.reset(_turn_token)

        return wrapper
    return decorator


def traced_span(span_name: str = None):
    """
    Decorator that **creates** a new OTel CHAIN span around the wrapped function.

    Use this for worker-thread helpers that are not LangGraph nodes (so
    ``LangChainInstrumentor`` does not auto-create a CHAIN span for them).
    Each invocation gets its own named parent span in Phoenix, under which the
    helper's chain invocations nest cleanly.

    Mirrors :func:`traced` for state-context extraction, OpenInference
    ``using_attributes`` propagation, contextvar logging plumbing, and error
    handling. The difference: this one *creates* the span instead of enriching
    an existing one.

    Args:
        span_name: Identifier used as the span name and ``therapy.node.name``
                   attribute. Defaults to the decorated function's ``__name__``.
    """
    def decorator(fn):
        _name = span_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = _extract_state_context(args[0], _name) if args else {}

            _session_token = _current_session.set(ctx.get("session_number"))
            _turn_token = _current_turn.set(ctx.get("turn_count"))

            oi_metadata = _build_oi_metadata(_name, ctx)
            oi_tags = _build_oi_tags(_name, ctx)
            oi_session_id = _build_session_id(ctx)

            with _tracer.start_as_current_span(_name) as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
                span.set_attribute(ATTR_NODE_NAME, _name)
                _set_span_attributes(span, ctx)

                if "misc" in _name.lower():
                    ctx_mgr = suppress_tracing()
                else:
                    ctx_mgr = using_attributes(
                        session_id=oi_session_id,
                        metadata=oi_metadata,
                        tags=oi_tags,
                    )

                with ctx_mgr:
                    try:
                        result = fn(*args, **kwargs)
                        # Serialize the return value as formatted JSON so Phoenix
                        # renders it with syntax highlighting instead of a raw string.
                        if result is not None and span.is_recording():
                            try:
                                if hasattr(result, 'model_dump_json'):
                                    out_str = result.model_dump_json(indent=2)
                                elif isinstance(result, dict):
                                    out_str = _json.dumps(result, indent=2, default=str)
                                else:
                                    out_str = None
                                if out_str:
                                    span.set_attribute("output.value", out_str)
                                    span.set_attribute("output.mime_type", "application/json")
                            except Exception:
                                logger.debug("Failed to capture span output", exc_info=True)
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise
                    finally:
                        _current_session.reset(_session_token)
                        _current_turn.reset(_turn_token)

        return wrapper
    return decorator


def enrich_span(
    *,
    node_name: Optional[str] = None,
    description: Optional[str] = None,
    session_number: Optional[int] = None,
    turn_number: Optional[int] = None,
    speaker: Optional[str] = None,
    phase: str = "conversation",
    purpose: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    miin_injection: Optional[bool] = None,
    output_summary: Optional[str] = None,
    thread_id: Optional[str] = None,
    current_dominant_stage: Optional[str] = None,
    shared_facts_count: Optional[int] = None,
    change_plan_items_count: Optional[int] = None,
    core_concerns_count: Optional[int] = None,
    core_motivations_count: Optional[int] = None,
    core_concerns_added: Optional[int] = None,
    core_concerns_edited: Optional[int] = None,
    core_concerns_retired: Optional[int] = None,
    core_motivations_added: Optional[int] = None,
    core_motivations_edited: Optional[int] = None,
    core_motivations_retired: Optional[int] = None,
    trigger_class: Optional[str] = None,
    bse_self_efficacy: Optional[float] = None,
    bse_motivational_readiness: Optional[float] = None,
    diagnosis_summary: Optional[str] = None,
    next_session_priority: Optional[str] = None,
    mr_impact: Optional[str] = None,
    anger_impact: Optional[str] = None,
    se_impact: Optional[str] = None,
    pr_impact: Optional[str] = None,
    implicit_threads_count: Optional[int] = None,
    biographical_facts_count: Optional[int] = None,
    core_values_count: Optional[int] = None,
    darn_change_talk_count: Optional[int] = None,
    cat_change_talk_count: Optional[int] = None,
    sustain_talk_themes_count: Optional[int] = None,
    current_clinical_focus: Optional[str] = None,
    simpatient_control: Optional[int] = None,
    simpatient_efficacy: Optional[int] = None,
    simpatient_awareness: Optional[int] = None,
    simpatient_reward: Optional[int] = None,
    patient_psi_style: Optional[str] = None,
    token_perplexity: Optional[float] = None,
    avg_logprob: Optional[float] = None,
    total_logprob: Optional[float] = None,
    causal_seed: Optional[int] = None,
    causal_patient_id: Optional[int] = None,
    causal_factual_perplexity: Optional[float] = None,
    causal_factual_avg_logprob: Optional[float] = None,
    causal_delta_perplexity: Optional[float] = None,
    causal_delta_avg_logprob: Optional[float] = None,
) -> None:
    """
    Enrich the **current** OpenTelemetry span with human-readable therapy
    session attributes.

    Sets attributes on both:
    - The current OTel span (``therapy.*`` namespace) for direct visibility.
    - The OpenInference context (``using_attributes``) so that any child
      spans created *after* this call also carry the metadata in Phoenix's
      Spans view.

    Args:
        node_name:       Short node identifier (e.g. "therapist", "global_miti_judge").
        description:     One-line human-readable explanation of what this span is doing.
        session_number:  Therapy session number (1-4).
        turn_number:     Conversation turn within the session.
        speaker:         "therapist" or "patient" when applicable.
        phase:           High-level phase: "conversation", "annotation", "evaluation",
                         "summary", "routing", "orchestration".
        purpose:         Optional finer-grained purpose label.
        model_provider:  LLM provider (e.g. "ollama", "openrouter").
        model_name:      LLM model name (e.g. "qwen2.5:latest").
        miin_injection:  Whether this turn is a MIIN injection turn.
        output_summary:  Short summary of the output (first ~200 chars).
    """
    span = trace.get_current_span()
    if span is None or not span.is_recording():
        return

    try:
        if node_name is not None:
            span.set_attribute(ATTR_NODE_NAME, node_name)
        if description is not None:
            span.set_attribute(ATTR_NODE_DESCRIPTION, description)
        span.set_attribute(ATTR_SESSION_PHASE, phase)

        if purpose is not None:
            span.set_attribute(ATTR_NODE_PURPOSE, purpose)
        if session_number is not None:
            span.set_attribute(ATTR_SESSION_NUMBER, session_number)
        if turn_number is not None:
            span.set_attribute(ATTR_SESSION_TURN, turn_number)
        if speaker is not None:
            span.set_attribute(ATTR_SPEAKER, speaker)
        if model_provider is not None:
            span.set_attribute(ATTR_MODEL_PROVIDER, model_provider)
        if model_name is not None:
            span.set_attribute(ATTR_MODEL_NAME, model_name)
        if miin_injection is not None:
            span.set_attribute(ATTR_MIIN_INJECTION, miin_injection)
        if output_summary is not None:
            span.set_attribute(ATTR_OUTPUT_SUMMARY, output_summary[:500])
        if thread_id is not None:
            span.set_attribute(ATTR_THREAD_ID, thread_id)
        if current_dominant_stage is not None:
            span.set_attribute(ATTR_DOMINANT_STAGE, current_dominant_stage)
        if shared_facts_count is not None:
            span.set_attribute(ATTR_SHARED_FACTS_COUNT, shared_facts_count)
        if change_plan_items_count is not None:
            span.set_attribute(ATTR_CHANGE_PLAN_ITEMS_COUNT, change_plan_items_count)
        if core_concerns_count is not None:
            span.set_attribute(ATTR_CORE_CONCERNS_COUNT, core_concerns_count)
        if core_motivations_count is not None:
            span.set_attribute(ATTR_CORE_MOTIVATIONS_COUNT, core_motivations_count)
        if core_concerns_added is not None:
            span.set_attribute(ATTR_CORE_CONCERNS_ADDED, core_concerns_added)
        if core_concerns_edited is not None:
            span.set_attribute(ATTR_CORE_CONCERNS_EDITED, core_concerns_edited)
        if core_concerns_retired is not None:
            span.set_attribute(ATTR_CORE_CONCERNS_RETIRED, core_concerns_retired)
        if core_motivations_added is not None:
            span.set_attribute(ATTR_CORE_MOTIVATIONS_ADDED, core_motivations_added)
        if core_motivations_edited is not None:
            span.set_attribute(ATTR_CORE_MOTIVATIONS_EDITED, core_motivations_edited)
        if core_motivations_retired is not None:
            span.set_attribute(ATTR_CORE_MOTIVATIONS_RETIRED, core_motivations_retired)
        if trigger_class is not None:
            span.set_attribute(ATTR_BSE_TRIGGER_CLASS, trigger_class)
        if bse_self_efficacy is not None:
            span.set_attribute(ATTR_BSE_SELF_EFFICACY, bse_self_efficacy)
        if bse_motivational_readiness is not None:
            span.set_attribute(ATTR_BSE_MOTIVATIONAL_READINESS, bse_motivational_readiness)
        if diagnosis_summary is not None:
            span.set_attribute(ATTR_DIAGNOSIS_SUMMARY, diagnosis_summary)
        if next_session_priority is not None:
            span.set_attribute(ATTR_NEXT_SESSION_PRIORITY, next_session_priority)
        if mr_impact is not None:
            span.set_attribute(ATTR_MR_IMPACT, mr_impact)
        if anger_impact is not None:
            span.set_attribute(ATTR_ANGER_IMPACT, anger_impact)
        if se_impact is not None:
            span.set_attribute(ATTR_SE_IMPACT, se_impact)
        if pr_impact is not None:
            span.set_attribute(ATTR_PR_IMPACT, pr_impact)
        if implicit_threads_count is not None:
            span.set_attribute(ATTR_IMPLICIT_THREADS_COUNT, implicit_threads_count)
        if biographical_facts_count is not None:
            span.set_attribute(ATTR_BIOGRAPHICAL_FACTS_COUNT, biographical_facts_count)
        if core_values_count is not None:
            span.set_attribute(ATTR_CORE_VALUES_COUNT, core_values_count)
        if darn_change_talk_count is not None:
            span.set_attribute(ATTR_DARN_CHANGE_TALK_COUNT, darn_change_talk_count)
        if cat_change_talk_count is not None:
            span.set_attribute(ATTR_CAT_CHANGE_TALK_COUNT, cat_change_talk_count)
        if sustain_talk_themes_count is not None:
            span.set_attribute(ATTR_SUSTAIN_TALK_THEMES_COUNT, sustain_talk_themes_count)
        if current_clinical_focus is not None:
            span.set_attribute(ATTR_CURRENT_CLINICAL_FOCUS, current_clinical_focus)
        if simpatient_control is not None:
            span.set_attribute(ATTR_SIMPATIENT_CONTROL, simpatient_control)
        if simpatient_efficacy is not None:
            span.set_attribute(ATTR_SIMPATIENT_EFFICACY, simpatient_efficacy)
        if simpatient_awareness is not None:
            span.set_attribute(ATTR_SIMPATIENT_AWARENESS, simpatient_awareness)
        if simpatient_reward is not None:
            span.set_attribute(ATTR_SIMPATIENT_REWARD, simpatient_reward)
        if patient_psi_style is not None:
            span.set_attribute(ATTR_PATIENT_PSI_STYLE, patient_psi_style)
        if token_perplexity is not None:
            span.set_attribute(ATTR_TOKEN_PERPLEXITY, token_perplexity)
        if avg_logprob is not None:
            span.set_attribute(ATTR_AVG_LOGPROB, avg_logprob)
        if total_logprob is not None:
            span.set_attribute(ATTR_TOTAL_LOGPROB, total_logprob)
        if causal_seed is not None:
            span.set_attribute(ATTR_CAUSAL_SEED, causal_seed)
        if causal_patient_id is not None:
            span.set_attribute(ATTR_CAUSAL_PATIENT_ID, causal_patient_id)
        if causal_factual_perplexity is not None:
            span.set_attribute(ATTR_CAUSAL_FACTUAL_PERPLEXITY, causal_factual_perplexity)
        if causal_factual_avg_logprob is not None:
            span.set_attribute(ATTR_CAUSAL_FACTUAL_AVG_LOGPROB, causal_factual_avg_logprob)
        if causal_delta_perplexity is not None:
            span.set_attribute(ATTR_CAUSAL_DELTA_PERPLEXITY, causal_delta_perplexity)
        if causal_delta_avg_logprob is not None:
            span.set_attribute(ATTR_CAUSAL_DELTA_AVG_LOGPROB, causal_delta_avg_logprob)

    except Exception:
        # Tracing must never break the application
        logger.debug("Failed to enrich span", exc_info=True)
