import os
import re
import yaml
import json
import time
import logging
import traceback
from typing import List, Dict, Any, Annotated, Optional
import operator
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# OpenAI and OpenRouter
from openai import OpenAI

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# Phoenix / OTEL
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace, propagate
from opentelemetry.trace import Status, StatusCode

# Import Ambivalence Prompt
try:
    from src_ambivalence.ambivalence_prompt import generate_ambivalence_evaluation
except ImportError:
    from ambivalence_prompt import generate_ambivalence_evaluation

# Load environment variables
load_dotenv()

# Load config if available
CONFIG_PATH = Path("conf_automisc/config.yaml")
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

PHOENIX_PROJECT = "Ambivalence-Judge"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PHOENIX_HOST = os.getenv("PHOENIX_HOST", "http://localhost:6006")

# Disable default LangChain tracing to avoid duplicates
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Setup Phoenix
tracer_provider = register(
    project_name=PHOENIX_PROJECT,
    endpoint=f"{PHOENIX_HOST}/v1/traces",
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = trace.get_tracer(__name__)

# Global configurations
DEFAULT_MODEL_NAME = "google/gemma-4-31b-it"
MODEL_NAME = DEFAULT_MODEL_NAME

PROVIDER_KWARGS = {
    "order": ["deepinfra"],
    "ignore": ["siliconflow", "parasail"],
    "sort": "price"
}
MAX_TOKENS = 2500
COST_PER_1M_INPUT = 0.1
COST_PER_1M_OUTPUT = 0.4

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ---------------------------------------------------------------------------
# Pydantic Schemas for Ambivalence Structured Output
# ---------------------------------------------------------------------------

class ProblemUncertaintyScores(BaseModel):
    U1: int = Field(description="Questions whether behavior causes problems / others exaggerate (1-5)")
    U2: int = Field(description="Doubt about needing professional help or change (1-5)")
    U3: int = Field(description="Wonders about loss of control while minimizing severity (1-5)")

class DecisionalConflictScores(BaseModel):
    D1: int = Field(description="Articulates strong attachment to perceived benefits/comforts (1-5)")
    D2: int = Field(description="Contrasts short-term relief with long-term costs (1-5)")
    D3: int = Field(description="Feels stuck/torn between competing values and habits (1-5)")

class ConversationalVacillationScores(BaseModel):
    V1: int = Field(description="Counters change statements immediately with status quo defense ('yes, but...') (1-5)")
    V2: int = Field(description="Fluctuates between Change Talk (DARN) and Sustain Talk (1-5)")
    V3: int = Field(description="Retreats into hesitation/qualification on concrete action steps (1-5)")

# LLM Output Schemas (LLM only provides item scores, verbatim evidence, and clinical reasoning)
class ProblemUncertaintyDimensionLLM(BaseModel):
    reasoning: str = Field(description="Clinical justification citing verbatim dialogue evidence")
    verbatim_evidence: List[str] = Field(description="1-2 verbatim quotes illustrating rating")
    item_scores: ProblemUncertaintyScores

class DecisionalConflictDimensionLLM(BaseModel):
    reasoning: str = Field(description="Clinical justification citing verbatim dialogue evidence")
    verbatim_evidence: List[str] = Field(description="1-2 verbatim quotes illustrating rating")
    item_scores: DecisionalConflictScores

class ConversationalVacillationDimensionLLM(BaseModel):
    reasoning: str = Field(description="Clinical justification citing verbatim dialogue evidence")
    verbatim_evidence: List[str] = Field(description="1-2 verbatim quotes illustrating rating")
    item_scores: ConversationalVacillationScores

class AmbivalenceLLMResponse(BaseModel):
    problem_uncertainty: ProblemUncertaintyDimensionLLM
    decisional_conflict: DecisionalConflictDimensionLLM
    conversational_vacillation: ConversationalVacillationDimensionLLM

# Complete Evaluation Schemas with Programmatically Computed Means
class ProblemUncertaintyDimension(ProblemUncertaintyDimensionLLM):
    dimension_mean: float = Field(description="Mean score across U1-U3 (calculated by code)")

class DecisionalConflictDimension(DecisionalConflictDimensionLLM):
    dimension_mean: float = Field(description="Mean score across D1-D3 (calculated by code)")

class ConversationalVacillationDimension(ConversationalVacillationDimensionLLM):
    dimension_mean: float = Field(description="Mean score across V1-V3 (calculated by code)")

class AmbivalenceEvaluationResult(BaseModel):
    problem_uncertainty: ProblemUncertaintyDimension
    decisional_conflict: DecisionalConflictDimension
    conversational_vacillation: ConversationalVacillationDimension
    composite_ambivalence_score: float = Field(description="Composite mean score across all 9 items (calculated by code)")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def extract_and_parse_json(text: str, model_cls: Any) -> Any:
    """Fallback parser to extract and validate JSON from markdown fences or text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        json_str = cleaned[start:end + 1]
        data = json.loads(json_str)
        return model_cls.model_validate(data)
    raise ValueError(f"Could not locate valid JSON in model response: {text}")

def call_with_retry(messages: List[Dict], response_format: Any, span_name: str):
    """
    Executes an LLM call with retry backoff and token/cost telemetry instrumentation.
    Tries structured output first, with fallback to JSON extraction if necessary.
    """
    retries = 3
    delay = 5
    for i in range(retries):
        try:
            with tracer.start_as_current_span(span_name) as span:
                kwargs = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "temperature": 0.0,
                    "extra_body": {
                        "provider": PROVIDER_KWARGS
                    }
                }
                if MAX_TOKENS is not None:
                    kwargs["max_tokens"] = MAX_TOKENS

                try:
                    # Attempt native OpenAI/OpenRouter structured parse
                    kwargs["response_format"] = response_format
                    response = client.beta.chat.completions.parse(**kwargs)
                    parsed = response.choices[0].message.parsed
                    raw_content = getattr(response.choices[0].message, "content", None)
                except Exception as parse_err:
                    logger.warning(f"Native structured parsing exception ({parse_err}). Falling back to manual JSON parse.")
                    kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**kwargs)
                    raw_content = response.choices[0].message.content
                    parsed = extract_and_parse_json(raw_content, response_format)

                if parsed is None and raw_content:
                    parsed = extract_and_parse_json(raw_content, response_format)

                if parsed is None:
                    refusal = getattr(response.choices[0].message, "refusal", "Unknown parsing failure")
                    raise ValueError(f"LLM returned None for parsed object. Refusal: {refusal}, Content: {raw_content}")

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                cost = (input_tokens * COST_PER_1M_INPUT / 1e6) + (output_tokens * COST_PER_1M_OUTPUT / 1e6)

                span.set_attribute("llm.input_tokens", input_tokens)
                span.set_attribute("llm.output_tokens", output_tokens)
                span.set_attribute("llm.cost", cost)

                return parsed, cost
        except Exception as e:
            if i == retries - 1:
                raise
            logger.warning(f"Error in {span_name} (retry {i+1}/{retries}): {e}")
            time.sleep(delay)

def load_profiles(profiles_path: Path) -> Dict[int, Dict[str, Any]]:
    """Loads patient profiles from JSONL file keyed by integer index."""
    profiles = {}
    if not profiles_path.exists():
        logger.warning(f"Profiles file not found at {profiles_path}")
        return profiles
    with open(profiles_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                idx = data.get("idx")
                if idx is not None:
                    profiles[int(idx)] = data
    return profiles

def extract_patient_index(file_path: Path) -> Optional[int]:
    """Infers patient index from parent folder name or filename prefix."""
    parent_name = file_path.parent.name
    if parent_name.isdigit():
        return int(parent_name)
    stem_prefix = file_path.stem.split("_")[0]
    if stem_prefix.isdigit():
        return int(stem_prefix)
    return None

# ---------------------------------------------------------------------------
# LangGraph State & Tasks
# ---------------------------------------------------------------------------

class SessionTask(BaseModel):
    session_id: str
    session_num: int
    session_transcript: str
    persona_summary: str
    target_behavior: str
    parent_span_context: Dict[str, str]

class GraphState(BaseModel):
    file_path: str
    output_path: str = ""
    profiles_path: str = "src_wai/musemi_profiles.jsonl"
    patient_idx: Optional[int] = None
    persona_summary: str = ""
    target_behavior: str = ""
    sessions_data: List[Dict[str, Any]] = []
    evaluated_sessions: Annotated[List[Dict[str, Any]], operator.add] = []
    total_cost: float = 0.0

# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------

def parse_transcript(state: GraphState):
    """
    Parses the simulation transcript file into sessions and loads matching patient profile.
    """
    file_path = Path(state.file_path)
    dir_name = file_path.parent.name
    
    with tracer.start_as_current_span(f"Therapy Ambivalence: {dir_name}/{file_path.stem}") as span:
        carrier = {}
        propagate.inject(carrier)

        # 1. Identify Patient Profile
        patient_idx = state.patient_idx
        if patient_idx is None:
            patient_idx = extract_patient_index(file_path)
            
        profiles = load_profiles(Path(state.profiles_path))
        profile = profiles.get(patient_idx, {}) if patient_idx is not None else {}

        personas = profile.get("Personas", [])
        if isinstance(personas, list):
            persona_summary = "; ".join(personas) if personas else "Patient attending therapy sessions."
        else:
            persona_summary = str(personas)

        target_behavior = profile.get("Behavior", profile.get("topic", "behavior change"))

        # 2. Parse Transcript File into Sessions
        content = file_path.read_text(encoding="utf-8")
        session_blocks = re.split(r'---\s*(Session \d+)\s*---', content)

        sessions = []
        for i in range(1, len(session_blocks), 2):
            session_id = session_blocks[i].strip()
            session_transcript = session_blocks[i + 1].strip()
            
            # Extract session integer
            num_match = re.search(r'\d+', session_id)
            session_num = int(num_match.group(0)) if num_match else (i // 2 + 1)

            sessions.append({
                "session_id": session_id,
                "session_num": session_num,
                "session_transcript": session_transcript,
                "persona_summary": persona_summary,
                "target_behavior": target_behavior,
                "parent_span_context": carrier
            })

        logger.info(f"Loaded transcript {file_path.name}: Patient IDX={patient_idx}, {len(sessions)} sessions found.")
        return {
            "patient_idx": patient_idx,
            "persona_summary": persona_summary,
            "target_behavior": target_behavior,
            "sessions_data": sessions
        }

def session_orchestrator(state: GraphState):
    """Conditional edge router: fan out each session to evaluate_session."""
    sends = []
    for s in state.sessions_data:
        sends.append(Send("evaluate_session", SessionTask(**s)))
    return sends

def evaluate_session(task: SessionTask):
    """
    Evaluates one session using the standardized Ambivalence evaluation prompt.
    """
    ctx = propagate.extract(task.parent_span_context)
    with tracer.start_as_current_span(f"{task.session_id}", context=ctx) as s_span:
        prompt = generate_ambivalence_evaluation(
            persona_summary=task.persona_summary,
            target_behavior=task.target_behavior,
            session_id=str(task.session_num),
            session_transcript=task.session_transcript
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        parsed_ambivalence, cost = call_with_retry(
            messages=messages,
            response_format=AmbivalenceLLMResponse,
            span_name=f"Ambivalence Judge ({task.session_id})"
        )

        # Compute dimension means and composite score programmatically in code
        u_scores = [
            parsed_ambivalence.problem_uncertainty.item_scores.U1,
            parsed_ambivalence.problem_uncertainty.item_scores.U2,
            parsed_ambivalence.problem_uncertainty.item_scores.U3,
        ]
        d_scores = [
            parsed_ambivalence.decisional_conflict.item_scores.D1,
            parsed_ambivalence.decisional_conflict.item_scores.D2,
            parsed_ambivalence.decisional_conflict.item_scores.D3,
        ]
        v_scores = [
            parsed_ambivalence.conversational_vacillation.item_scores.V1,
            parsed_ambivalence.conversational_vacillation.item_scores.V2,
            parsed_ambivalence.conversational_vacillation.item_scores.V3,
        ]

        u_mean = round(sum(u_scores) / len(u_scores), 2)
        d_mean = round(sum(d_scores) / len(d_scores), 2)
        v_mean = round(sum(v_scores) / len(v_scores), 2)
        composite = round((sum(u_scores) + sum(d_scores) + sum(v_scores)) / 9.0, 2)

        full_evaluation = AmbivalenceEvaluationResult(
            problem_uncertainty=ProblemUncertaintyDimension(
                reasoning=parsed_ambivalence.problem_uncertainty.reasoning,
                verbatim_evidence=parsed_ambivalence.problem_uncertainty.verbatim_evidence,
                item_scores=parsed_ambivalence.problem_uncertainty.item_scores,
                dimension_mean=u_mean
            ),
            decisional_conflict=DecisionalConflictDimension(
                reasoning=parsed_ambivalence.decisional_conflict.reasoning,
                verbatim_evidence=parsed_ambivalence.decisional_conflict.verbatim_evidence,
                item_scores=parsed_ambivalence.decisional_conflict.item_scores,
                dimension_mean=d_mean
            ),
            conversational_vacillation=ConversationalVacillationDimension(
                reasoning=parsed_ambivalence.conversational_vacillation.reasoning,
                verbatim_evidence=parsed_ambivalence.conversational_vacillation.verbatim_evidence,
                item_scores=parsed_ambivalence.conversational_vacillation.item_scores,
                dimension_mean=v_mean
            ),
            composite_ambivalence_score=composite
        )

        return {
            "evaluated_sessions": [{
                "session_id": task.session_id,
                "session_num": task.session_num,
                "cost": cost,
                "evaluation": full_evaluation.model_dump()
            }]
        }

def aggregate_and_save(state: GraphState):
    """
    Aggregates session evaluations, computes multi-session summary statistics, and saves output.
    """
    sorted_sessions = sorted(state.evaluated_sessions, key=lambda x: x["session_num"])
    total_cost = sum(s["cost"] for s in sorted_sessions)

    composite_scores = [s["evaluation"]["composite_ambivalence_score"] for s in sorted_sessions]
    u_means = [s["evaluation"]["problem_uncertainty"]["dimension_mean"] for s in sorted_sessions]
    d_means = [s["evaluation"]["decisional_conflict"]["dimension_mean"] for s in sorted_sessions]
    v_means = [s["evaluation"]["conversational_vacillation"]["dimension_mean"] for s in sorted_sessions]

    mean_composite = round(sum(composite_scores) / len(composite_scores), 2) if composite_scores else 0.0
    mean_uncertainty = round(sum(u_means) / len(u_means), 2) if u_means else 0.0
    mean_decisional = round(sum(d_means) / len(d_means), 2) if d_means else 0.0
    mean_vacillation = round(sum(v_means) / len(v_means), 2) if v_means else 0.0

    trajectory = [
        {"session_id": s["session_id"], "session_num": s["session_num"], "composite_ambivalence_score": s["evaluation"]["composite_ambivalence_score"]}
        for s in sorted_sessions
    ]

    output_data = {
        "metadata": {
            "model_name": MODEL_NAME,
            "patient_idx": state.patient_idx,
            "target_behavior": state.target_behavior,
            "persona_summary": state.persona_summary,
            "transcript_file": state.file_path,
            "total_cost": round(total_cost, 6),
            "provider_kwargs": PROVIDER_KWARGS,
            "max_tokens": MAX_TOKENS
        },
        "summary": {
            "mean_composite_ambivalence": mean_composite,
            "mean_problem_uncertainty": mean_uncertainty,
            "mean_decisional_conflict": mean_decisional,
            "mean_conversational_vacillation": mean_vacillation,
            "session_trajectory": trajectory
        },
        "sessions": sorted_sessions
    }

    output_path = Path(state.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, sort_keys=False, allow_unicode=True)

    logger.info(f"Successfully saved Ambivalence evaluation to {output_path}. Composite: {mean_composite}, Total Cost: ${total_cost:.5f}")
    return {"total_cost": total_cost}

# ---------------------------------------------------------------------------
# Build LangGraph Pipeline
# ---------------------------------------------------------------------------

builder = StateGraph(GraphState)
builder.add_node("parse_transcript", parse_transcript)
builder.add_node("evaluate_session", evaluate_session)
builder.add_node("aggregate_and_save", aggregate_and_save)

builder.add_edge(START, "parse_transcript")
builder.add_conditional_edges("parse_transcript", session_orchestrator, ["evaluate_session"])
builder.add_edge("evaluate_session", "aggregate_and_save")
builder.add_edge("aggregate_and_save", END)

ambivalence_judge_app = builder.compile()

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ambivalence LLM as a Judge for multi-session therapy simulation transcripts")
    parser.add_argument("--input-file", required=True, help="Path to transcript .txt file")
    parser.add_argument("--output-file", help="Path to output .yaml file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name (default: google/gemma-4-31b-it)")
    parser.add_argument("--profiles-file", type=str, default="src_wai/musemi_profiles.jsonl", help="Path to patient profiles JSONL")
    parser.add_argument("--provider-kwargs", type=str, help="JSON-encoded OpenRouter provider preferences")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Max tokens for LLM generation")

    args = parser.parse_args()

    if args.model:
        MODEL_NAME = args.model
    if args.provider_kwargs:
        PROVIDER_KWARGS = json.loads(args.provider_kwargs)
    if args.max_tokens is not None:
        MAX_TOKENS = args.max_tokens

    input_path = Path(args.input_file)
    model_short_name = MODEL_NAME.split("/")[-1].replace(":", "_").replace(".", "_")

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.parent / f"{input_path.stem}_ambivalence_{model_short_name}.yaml"

    logger.info(f"Starting Ambivalence Judge for {input_path} with model={MODEL_NAME}")
    try:
        ambivalence_judge_app.invoke({
            "file_path": str(input_path),
            "output_path": str(output_path),
            "profiles_path": args.profiles_file
        })
        logger.info(f"Done. Output saved to {output_path}")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        traceback.print_exc()
        exit(1)
