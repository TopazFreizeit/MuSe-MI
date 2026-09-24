import os
import re
import yaml
import time
import logging
import traceback
from typing import List, Dict, Any, Annotated, Optional
import operator
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# OpenAI and OpenRouter
import openai
from openai import OpenAI

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# Phoenix / OTEL
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace, propagate
from opentelemetry.trace import Status, StatusCode

# AutoMISC components
from components.parser import PARSER_SYSTEM_PROMPT, few_shots, MISCParser
from components.prompts.loader import render_prompt, render_user_prompt
from components.prompts.response_formats import (
    CounsellorUtterance_t1,
    CounsellorUtterance_t2,
    ClientUtterance_t1,
    ClientUtterance_t2
)

# Load environment variables
load_dotenv()

# Load config
with open("conf_automisc/config.yaml", "r") as f:
    config = yaml.safe_load(f)

PHOENIX_PROJECT = os.getenv("PHOENIX_PROJECT", "Judges")
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

# Global configurations, will be overridden by argparse if provided
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
NUM_CONTEXT_TURNS = config.get("annotator", {}).get("num_context_turns", 5)
PROVIDER_KWARGS = {
    "order": ["deepinfra", "inceptron", "akashml", "novita"],
    "ignore": ["parasail"],
    "sort": "price"
}
MAX_TOKENS = None
COST_PER_1M_INPUT = 0.1
COST_PER_1M_OUTPUT = 0.4

# Global progress tracking
from threading import Lock
class ProgressTracker:
    def __init__(self):
        self.total_volleys = 0
        self.completed_volleys = 0
        self.lock = Lock()
        self.last_logged_percent = -1

    def set_total(self, total):
        self.total_volleys = total
        logging.info(f"Total volleys to process: {self.total_volleys}")

    def increment(self):
        with self.lock:
            self.completed_volleys += 1
            percent = (self.completed_volleys * 100) // self.total_volleys
            if percent >= self.last_logged_percent + 10:
                self.last_logged_percent = (percent // 10) * 10
                logging.info(f"Progress: {self.last_logged_percent}% ({self.completed_volleys}/{self.total_volleys})")

progress_tracker = ProgressTracker()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def call_with_retry(messages: List[Dict], response_format: Any, span_name: str):
    retries = 3
    delay = 5
    for i in range(retries):
        try:
            with tracer.start_as_current_span(span_name) as span:
                kwargs = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "response_format": response_format,
                    "temperature": 0.0,
                    "extra_body": {
                        "provider": PROVIDER_KWARGS
                    }
                }
                if MAX_TOKENS is not None:
                    kwargs["max_tokens"] = MAX_TOKENS

                response = client.beta.chat.completions.parse(**kwargs)
                
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                cost = (input_tokens * COST_PER_1M_INPUT / 1e6) + (output_tokens * COST_PER_1M_OUTPUT / 1e6)
                
                span.set_attribute("llm.input_tokens", input_tokens)
                span.set_attribute("llm.output_tokens", output_tokens)
                span.set_attribute("llm.cost", cost)
                
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raw_content = getattr(response.choices[0].message, "content", "No Content")
                    refusal = getattr(response.choices[0].message, "refusal", "Unknown parsing failure")
                    logging.warning(f"Failed to parse. Raw LLM content: {raw_content}")
                    raise ValueError(f"LLM returned None for parsed object. Refusal: {refusal}, Content: {raw_content}")
                
                return parsed, cost
        except Exception as e:
            if i == retries - 1:
                raise
            logging.warning(f"Error in {span_name} (retry {i+1}/{retries}): {e}")
            time.sleep(delay)

# Graph State
class VolleyTask(BaseModel):
    session_id: str
    session_idx: int
    volley_idx: int
    speaker: str
    text: str
    context_text: str
    parent_span_context: Dict[str, str]

class GraphState(BaseModel):
    file_path: str
    sessions_data: List[Dict] = []
    processed_volleys: Annotated[List[Dict], operator.add] = []
    output_path: str = "output_annotated.yaml"
    total_cost: float = 0.0

# Nodes
def parse_transcript(state: GraphState):
    dir_name = Path(state.file_path).parent.name
    with tracer.start_as_current_span(f"Therapy: {dir_name}") as span:
        # Inject context for sub-spans
        carrier = {}
        propagate.inject(carrier)
        
        content = Path(state.file_path).read_text()
        session_blocks = re.split(r'--- (Session \d+) ---', content)
        
        sessions = []
        total_volleys = 0
        
        # re.split returns [prefix, session_id, session_content, session_id, session_content, ...]
        for i in range(1, len(session_blocks), 2):
            session_id = session_blocks[i]
            session_content = session_blocks[i+1].strip()
            
            volleys_raw = []
            # Match "Therapist: ..." or "Patient: ..." or "Counsellor: ..." or "Client: ..."
            volley_matches = list(re.finditer(r'^(Therapist|Patient|Counsellor|Client):\s*(.*?)(?=\n^(?:Therapist|Patient|Counsellor|Client):|\Z)', session_content, re.MULTILINE | re.DOTALL))
            
            for match in volley_matches:
                speaker_raw = match.group(1).lower()
                speaker = "counsellor" if speaker_raw in ["therapist", "counsellor"] else "client"
                text = match.group(2).strip()
                volleys_raw.append({"speaker": speaker, "text": text})

            volleys = []
            for v_idx, v_data in enumerate(volleys_raw):
                # Build context_text: formatted string of NUM_CONTEXT_TURNS previous volleys
                start_idx = max(0, v_idx - NUM_CONTEXT_TURNS)
                context_segments = []
                for j in range(start_idx, v_idx):
                    context_segments.append(f"{volleys_raw[j]['speaker'].capitalize()}: {volleys_raw[j]['text']}")
                
                context_text = "\n".join(context_segments)

                if v_data["speaker"] == "client":
                    volleys.append({
                        "session_id": session_id,
                        "session_idx": i // 2,
                        "volley_idx": v_idx,
                        "speaker": v_data["speaker"],
                        "text": v_data["text"],
                        "context_text": context_text,
                        "parent_span_context": carrier
                    })
                    total_volleys += 1
            
            sessions.append({"session_id": session_id, "volleys": volleys})
            
        progress_tracker.set_total(total_volleys)
        return {"sessions_data": sessions}

def session_orchestrator(state: GraphState):
    # This function is used in conditional edges to fan out to process_volley
    sends = []
    for session in state.sessions_data:
        for v in session["volleys"]:
            sends.append(Send("process_volley", VolleyTask(**v)))
    return sends

def process_volley(task: VolleyTask):
    # Re-attach to parent context
    ctx = propagate.extract(task.parent_span_context)
    # We want Therapy > Session > Volley
    # Therapy is task.parent_span_context
    with tracer.start_as_current_span(f"{task.session_id}", context=ctx) as s_span:
        with tracer.start_as_current_span(f"Volley: {task.volley_idx} - {task.speaker}") as v_span:
            try:
                # 1. Parse into utterances
                messages = [
                    {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                    *few_shots,
                    {"role": "user", "content": task.text}
                ]
                parsed_res, cost = call_with_retry(messages, MISCParser, "Parser")
                total_volley_cost = cost
                
                utterances_data = []
                
                for u_text in parsed_res.utterances:
                    # T1 Annotation
                    t1_sys = render_prompt(speaker=task.speaker, structure="t1")
                    # Use the provided context_text for the prompt
                    user_p = render_user_prompt(transcript=task.context_text, speaker=task.speaker, utterance=u_text)
                    
                    t1_msgs = [
                        {"role": "system", "content": t1_sys},
                        {"role": "user", "content": user_p}
                    ]
                    t1_res, cost1 = call_with_retry(t1_msgs, CounsellorUtterance_t1 if task.speaker == "counsellor" else ClientUtterance_t1, "Annotator T1")
                    total_volley_cost += cost1
                    
                    # T2 Annotation
                    t2_sys = render_prompt(speaker=task.speaker, structure="t2", label=t1_res.label)
                    t2_msgs = [
                        {"role": "system", "content": t2_sys},
                        {"role": "user", "content": user_p}
                    ]
                    t2_res, cost2 = call_with_retry(t2_msgs, CounsellorUtterance_t2 if task.speaker == "counsellor" else ClientUtterance_t2, "Annotator T2")
                    total_volley_cost += cost2
                    
                    utterances_data.append({
                        "text": u_text,
                        "t1": {"label": t1_res.label, "explanation": t1_res.explanation},
                        "t2": {"label": t2_res.label, "explanation": t2_res.explanation},
                        "cost": cost1 + cost2
                    })
                
                progress_tracker.increment()
                return {
                    "processed_volleys": [{
                        "session_id": task.session_id,
                        "session_idx": task.session_idx,
                        "volley_idx": task.volley_idx,
                        "speaker": task.speaker,
                        "text": task.text,
                        "utterances": utterances_data,
                        "cost": total_volley_cost
                    }]
                }
            except Exception as e:
                v_span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

def aggregate_and_save(state: GraphState):
    # Sort by session_idx then volley_idx
    sorted_volleys = sorted(state.processed_volleys, key=lambda x: (x["session_idx"], x["volley_idx"]))
    
    sessions_map = {}
    total_therapy_cost = 0.0
    
    for v in sorted_volleys:
        s_id = v["session_id"]
        if s_id not in sessions_map:
            sessions_map[s_id] = {"session_id": s_id, "session_cost": 0.0, "volleys": []}
        
        sessions_map[s_id]["volleys"].append({
            "speaker": v["speaker"],
            "text": v["text"],
            "utterances": v["utterances"]
        })
        sessions_map[s_id]["session_cost"] += v["cost"]
        total_therapy_cost += v["cost"]
        
    output = {
        "metadata": {
            "model_name": MODEL_NAME,
            "num_context_turns": NUM_CONTEXT_TURNS,
            "provider_kwargs": PROVIDER_KWARGS,
            "max_tokens": MAX_TOKENS
        },
        "total_cost": round(total_therapy_cost, 6),
        "sessions": list(sessions_map.values())
    }
    
    with open(state.output_path, "w") as f:
        yaml.dump(output, f, sort_keys=False)
        
    logging.info(f"Successfully saved results to {state.output_path}. Total cost: ${output['total_cost']}")
    return {"total_cost": total_therapy_cost}

# Build Graph
builder = StateGraph(GraphState)
builder.add_node("parse_transcript", parse_transcript)
builder.add_node("process_volley", process_volley)
builder.add_node("aggregate_and_save", aggregate_and_save)

builder.add_edge(START, "parse_transcript")
builder.add_conditional_edges("parse_transcript", session_orchestrator, ["process_volley"])
builder.add_edge("process_volley", "aggregate_and_save")
builder.add_edge("aggregate_and_save", END)

app = builder.compile()

if __name__ == "__main__":
    import sys
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Process transcript graphs using AutoMISC")
    parser.add_argument("--input-file", required=True, help="Path to the input transcript file")
    parser.add_argument("--output-file", help="Path to the output yaml file")
    parser.add_argument("--model", type=str, help="Model name to use")
    parser.add_argument("--context-turns", type=int, help="Number of context turns")
    parser.add_argument("--provider-kwargs", type=str, help="JSON-encoded string of provider kwargs")
    parser.add_argument("--max-tokens", type=int, help="Maximum number of output tokens")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    
    if args.model:
        MODEL_NAME = args.model
    if args.context_turns is not None:
        NUM_CONTEXT_TURNS = args.context_turns
    if args.provider_kwargs:
        PROVIDER_KWARGS = json.loads(args.provider_kwargs)
    if args.max_tokens is not None:
        MAX_TOKENS = args.max_tokens
        
    # Default output path is in the same directory as transcript
    default_output = Path(input_file).parent / "output_annotated.yaml"
    output_file = args.output_file if args.output_file else str(default_output)
    
    try:
        logging.info(f"Starting processing for {input_file} (Model: {MODEL_NAME}, Context Turns: {NUM_CONTEXT_TURNS})")
        app.invoke({"file_path": input_file, "output_path": output_file})
        logging.info(f"Processing complete. Output saved to {output_file}")
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
