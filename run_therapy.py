"""
Example usage of the therapy conversation LangGraph.
Runs a complete 4-session therapy program.
"""
import datetime
import logging
import os
import signal
import sys
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Phoenix observability - connects to standalone Phoenix server
# Start Phoenix separately with: phoenix serve
# Then open http://localhost:6006 in your browser
from openinference.semconv.trace import SpanAttributes
from src.utils.tracing import (
    setup_phoenix_tracing,
    flush_phoenix_tracing,
    tracer as _root_tracer,
    SessionTurnFilter,
)

tracer_provider = setup_phoenix_tracing()

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from src.graphs import SingleSessionGraphBuilder, MultiSessionGraphBuilder
from src.graphs.visualization import save_graph_visualization
from src.graphs.graph_config import MultiSessionState, LatentTrajectoryEntry, RECURSION_LIMIT
from src.analytics import save_final_state
from src.therapist.therapist_config import generate_miin_injection_schedule
from src.patient.patient_init import initialize_patient
from src.patient.patient_config import PATIENT_IDX
from src.utils.cost_tracker import cost_tracker
from src.graphs.graph_config import NUM_SESSIONS

# Maximum wall-clock time for the entire run (seconds)
# 4 sessions × ~17 turns × ~30s per turn ≈ 35 min, with generous margin
MAX_RUN_DURATION_SECONDS = int(os.getenv("MAX_RUN_DURATION_SECONDS", "7200"))  # 2 hours default

class RunTimeoutError(Exception):
    """Raised when the graph run exceeds the wall-clock time limit."""
    pass


def _timeout_handler(signum, frame):
    raise RunTimeoutError(
        f"Run exceeded wall-clock time limit of {MAX_RUN_DURATION_SECONDS}s"
    )

# Configure logging for observability
log_filename = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join("logs", log_filename)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [S%(session)s|T%(turn)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filepath),
    ],
)
# Attach filter so %(session)s and %(turn)s are always available
_session_turn_filter = SessionTurnFilter()
for _handler in logging.root.handlers:
    _handler.addFilter(_session_turn_filter)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_filepath}")


if __name__ == "__main__":
    # Resume an existing run by passing its thread_id, or start fresh
    resume_thread_id = os.getenv("RESUME_THREAD_ID")
    
    # SQLite checkpoint DB persists across crashes
    checkpoint_db = os.path.join("logs", "checkpoints.sqlite")
    
    with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
      try:
        thread_id = resume_thread_id or str(uuid.uuid4())
        logger.info(f"Starting {NUM_SESSIONS}-session therapy program (thread_id={thread_id})")
        
        # Build single-session graph
        store = InMemoryStore()
        single_builder = SingleSessionGraphBuilder(store=store)
        single_graph = single_builder.build()
        logger.info("Single-session graph built")
        
        # Save single-session graph visualization
        save_graph_visualization(single_graph, "single_session_graph.png")
        
        # Build multi-session graph with checkpointing
        multi_builder = MultiSessionGraphBuilder(single_graph, checkpointer=checkpointer, store=store)
        multi_graph = multi_builder.build()
        logger.info("Multi-session graph built")
        
        # Save multi-session graph visualization
        save_graph_visualization(multi_graph, "multi_session_graph.png")
        
        invoke_config = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {
                "single_session_graph": single_graph,
                "thread_id": thread_id
            }
        }
        
        store_file = os.path.join("logs", f"store_{thread_id}.json")
        
        if resume_thread_id:
            # Resume from checkpoint — pass None as input to continue
            logger.info(f"Resuming from checkpoint (thread_id={resume_thread_id})")
            checkpoint = checkpointer.get(invoke_config)
            if checkpoint is None:
                logger.error(f"No checkpoint found for thread_id={resume_thread_id}")
                sys.exit(1)
            
            from src.utils.store import load_store
            load_store(store, store_file)
            logger.info(f"Loaded store from {store_file}")
            
            logger.info(f"Checkpoint found, resuming...")
            initial_state = None
        else:
            # Fresh run — initialize patient profile
            cost_tracker.reset()
            patient = initialize_patient(PATIENT_IDX)
            logger.info(f"Patient {PATIENT_IDX} profile initialized")

            # Build initial state
            miin_injection_schedule = generate_miin_injection_schedule()
            logger.info(f"MIIN injection schedule: {miin_injection_schedule}")
            
            session_1_is_miin_target = 1 in miin_injection_schedule
            session_1_injection_turns = miin_injection_schedule.get(1, [])

            init_lvs = patient.latent_variables.model_dump()
            initial_trajectory_entry: LatentTrajectoryEntry = {
                "step_index": 0,
                "component": "initial_state",
                "session_number": 1,
                "turn_number": 0,
                "anger": float(init_lvs["anger"]),
                "self_efficacy": float(init_lvs["self_efficacy"]),
                "problem_recognition": float(init_lvs["problem_recognition"]),
                "motivational_readiness": float(init_lvs["motivational_readiness"]),
                "details": {"program_initial_baseline": True},
            }
            
            initial_state: MultiSessionState = {
                "thread_id": thread_id,
                "patient_id": PATIENT_IDX,
                "current_session_number": 1,
                "miin_injection_schedule": miin_injection_schedule,
                "all_session_turns": [],
                "patient_latent_variables": init_lvs,
                "patient_profile_data": patient.profile_data.model_dump(),
                "all_latent_variable_logs": [],
                "latent_variable_trajectory": [initial_trajectory_entry],
                "past_event_titles": [],
                "bse_trigger_classes": [],
                "current_session": {
                    "messages": [],
                    "current_speaker": "",
                    "session_ended": False,
                    "session_end_reason": None,
                    "turn_count": 0,
                    "patient_system_prompt": "",
                    "session_number": 1,
                    "current_session_turns": [],
                    "miin_injection_turn_numbers": session_1_injection_turns,
                    "miin_injection_active": session_1_is_miin_target,
                    "last_used_techniques": [],
                    "patient_latent_variables": init_lvs,
                    "latent_variable_logs": [],
                    "session_latent_trajectory": [],
                },
            }
        
        # Execute graph
        logger.info(f"Executing multi-session graph ({NUM_SESSIONS} sessions)...")
        start_time = datetime.datetime.now()
        
        # Set wall-clock timeout to prevent indefinite hangs
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(MAX_RUN_DURATION_SECONDS)
        
        therapist_type = os.getenv("THERAPIST_TYPE", "expert")
        try:
            # Root span: groups all {NUM_SESSIONS} run_single_session spans (and their
            # children) into a single trace in Phoenix instead of having
            # each session appear as a separate top-level trace.
            with _root_tracer.start_as_current_span("therapy_program") as root_span:
                root_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
                root_span.set_attribute("therapy.thread_id", thread_id)
                root_span.set_attribute("therapy.sessions_count", NUM_SESSIONS)
                root_span.set_attribute("therapy.therapist_type", therapist_type)
                
                trace_id = format(root_span.get_span_context().trace_id, "032x")
                if initial_state is None:
                    initial_state = {"therapy_program_trace_id": trace_id}
                else:
                    initial_state["therapy_program_trace_id"] = trace_id
                    if "current_session" in initial_state:
                        initial_state["current_session"]["therapy_program_trace_id"] = trace_id

                final_state = multi_graph.invoke(
                    initial_state,
                    config=invoke_config
                )
        finally:
            signal.alarm(0)  # Cancel the alarm
            
            from src.utils.store import dump_store
            dump_store(store, store_file)
            logger.info(f"Saved store to {store_file}")
        
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        logger.info(f"Multi-session therapy program completed and it took {duration}")
        cost_tracker.log_total()

        # Save final state for analysis
        saved_path = save_final_state(final_state)
        logger.info(f"Final state saved to: {saved_path}")
        logger.info(f"To analyze results, run: python analyze_run.py {saved_path}")

        from src.utils.config import (
            CAUSAL_RECORD_BASELINE,
            CAUSAL_BASELINE_DIR,
            CAUSAL_FIXED_CONTEXT_PROBE,
        )
        if CAUSAL_RECORD_BASELINE:
            from src.causal_study.baseline_recorder import save_causal_baseline_trace
            trace_saved_path = save_causal_baseline_trace(
                final_state=final_state,
                patient_id=PATIENT_IDX,
                output_dir=CAUSAL_BASELINE_DIR,
                store=store,
            )
            logger.info(f"Causal baseline trace saved to: {trace_saved_path}")

        if CAUSAL_FIXED_CONTEXT_PROBE:
            logger.info(
                f"CAUSAL_FIXED_CONTEXT_PROBE is enabled. Running fixed-context probe for Patient {PATIENT_IDX}..."
            )
            from experiments.run_fixed_context_probe import run_probe_for_patient_id
            run_probe_for_patient_id(
                patient_id=PATIENT_IDX,
                baseline_dir=CAUSAL_BASELINE_DIR,
            )

        flush_phoenix_tracing()
      except RunTimeoutError as e:
        logger.error(f"Run timed out: {e}")
        cost_tracker.log_total()
        logger.info(f"To resume this run: RESUME_THREAD_ID={thread_id} python example_usage.py")
        sys.exit(2)
      except Exception as e:
        logger.exception(f"Fatal error: {e}")
        cost_tracker.log_total()
        logger.info(f"To resume this run: RESUME_THREAD_ID={thread_id} python example_usage.py")
        sys.exit(1)