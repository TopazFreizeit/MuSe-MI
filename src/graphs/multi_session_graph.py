"""
Multi-session therapy conversation graph (4 sessions).
"""
from typing import Dict, Any, Optional
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig

from ..therapist.therapist_rolling_memory import therapist_rolling_memory_node
from .graph_config import MultiSessionState, ConversationState, LatentTrajectoryEntry, RECURSION_LIMIT, NUM_SESSIONS
from ..utils.tracing import enrich_span, traced_span
from ..utils.store import PATIENT_MEMORY_NS, THERAPIST_MEMORY_NS
from ..between_session_events.between_session_event_agent import between_session_event_node
from ..between_session_events.between_session_config import ENABLE_BETWEEN_SESSION_EVENTS
from ..patient_state_manager.rolling_memory_agent import rolling_memory_node
from ..patient.patient_summary_memory import (
    patient_summary_memory_node,
    patient_summary_inter_sessions_node,
)
from ..cami_therapist_baseline import cami_therapist_memory_node, cami_therapist_inter_sessions_node
from ..consistent_client_baseline import consistent_client_memory_node, consistent_client_inter_sessions_node
from ..vanilla_baseline import vanilla_client_memory_node, vanilla_client_inter_sessions_node
from ..patient_psi_baseline import patient_psi_session_summary_node, patient_psi_inter_sessions_node
from ..simpatient_baseline import (
    simpatient_global_scores_node,
    simpatient_session_summary_node,
    simpatient_between_session_event_node,
    simpatient_inter_sessions_node,
)
logger = logging.getLogger(__name__)


@traced_span("run_single_session")
def run_single_session_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    """
    Adapter node that runs a single therapy session as a subgraph.

    Nodes inside the subgraph access memory exclusively via runtime.store.

    Args:
        state: Parent MultiSessionState
        config: Configuration including single_session_graph
        runtime: LangGraph Runtime providing access to the compiled Store

    Returns:
        Updated MultiSessionState with current_session results and synced memory
    """
    logger = logging.getLogger(__name__)

    session_number = state['current_session_number']
    logger.info(f"run_single_session_node invoked for session {session_number}")

    # Get single-session graph from config
    configurable = config.get("configurable", {})
    single_session_graph = configurable.get("single_session_graph")

    enrich_span(
        node_name="run_single_session",
        description=f"Execute single therapy session {session_number}/{NUM_SESSIONS}",
        session_number=session_number,
        phase="orchestration",
        purpose="subgraph_execution",
    )

    if not single_session_graph:
        raise ValueError("single_session_graph not provided in config")

    # Guard: patient_latent_variables must be initialised before the graph runs.
    patient_latent_variables = state.get("patient_latent_variables")
    if not patient_latent_variables:
        raise ValueError(
            f"[Session {session_number}] patient_latent_variables is missing or empty in "
            "MultiSessionState. Ensure the patient profile was initialised before invoking "
            "the graph (see example_usage.py → initialize_patient)."
        )

    # ── Prepare subgraph session state ───────────────────────────────────
    session_state = state["current_session"].copy()
    session_state["session_number"] = session_number

    # Set MIIN injection state from the pre-computed schedule
    miin_injection_schedule = state.get("miin_injection_schedule", {})
    session_state["miin_injection_active"] = session_number in miin_injection_schedule
    session_state["miin_injection_turn_numbers"] = miin_injection_schedule.get(session_number, [])

    # Inject current latent variables into session state so subgraph nodes can read/update them
    session_state["patient_latent_variables"] = patient_latent_variables

    initial_log_entry: dict = {
        "turn_number": 0,
        "self_efficacy": patient_latent_variables["self_efficacy"],
        "anger": patient_latent_variables["anger"],
        "problem_recognition": patient_latent_variables["problem_recognition"],
        "motivational_readiness": patient_latent_variables["motivational_readiness"],
    }
    session_state["latent_variable_logs"] = [initial_log_entry]

    initial_trajectory_entry: LatentTrajectoryEntry = {
        "step_index": -1,
        "component": "initial_state",
        "session_number": session_number,
        "turn_number": 0,
        "anger": float(patient_latent_variables["anger"]),
        "self_efficacy": float(patient_latent_variables["self_efficacy"]),
        "problem_recognition": float(patient_latent_variables["problem_recognition"]),
        "motivational_readiness": float(patient_latent_variables["motivational_readiness"]),
        "details": {"session_start_baseline": True},
    }
    session_state["session_latent_trajectory"] = [initial_trajectory_entry]

    subgraph_config = {
        "recursion_limit": RECURSION_LIMIT,
        "callbacks": config.get("callbacks"),
        "configurable": {
            "current_session_number": state["current_session_number"],
            "all_session_turns": state["all_session_turns"],
            "patient_latent_variables": patient_latent_variables,
            "patient_profile_data": state.get("patient_profile_data"),
            "bse_trigger_classes": state.get("bse_trigger_classes"),
        }
    }

    # Invoke subgraph with session state
    final_session_state = single_session_graph.invoke(session_state, config=subgraph_config)

    logger.info(f"Session {session_number} subgraph complete — rolling memory will run to consolidate the end of session.")

    from ..utils.config import (
        ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES,
        DISABLE_COGNITIVE_INTERPRETER_IMPACT,
        VANILLA_BASELINE,
        CONSISTENT_CLIENT_BASELINE,
        SIMPATIENT_BASELINE,
        PATIENT_PSI_BASELINE,
    )
    ci_active = not (
        ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES
        or DISABLE_COGNITIVE_INTERPRETER_IMPACT
        or VANILLA_BASELINE
        or CONSISTENT_CLIENT_BASELINE
        or SIMPATIENT_BASELINE
        or PATIENT_PSI_BASELINE
    )

    parent_trajectory = list(state.get("latent_variable_trajectory", []))
    session_trajectory = final_session_state.get("session_latent_trajectory", [])
    if ci_active and session_trajectory:
        parent_trajectory = parent_trajectory + session_trajectory

    return {
        "current_session": final_session_state,
        "patient_latent_variables": final_session_state["patient_latent_variables"],
        "latent_variable_trajectory": parent_trajectory,
    }


@traced_span("patient_rolling_memory_trigger")
def patient_rolling_memory_trigger(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    """
    Inter-session node that consolidates the remaining turns of the just-completed session
    into structured cross-session memory (facts, motivations, concerns, plans, etc.) and applies
    the Macro-Shift to the patient's latent variables.

    Reads/writes store via runtime.store.
    """
    session_number = state["current_session_number"]
    logger.info(f"patient_rolling_memory_trigger: rolling memory consolidating end of session {session_number}")

    current_session = state["current_session"]
    turn_count = current_session.get("turn_count", 0)
    last_memory_turn = current_session.get("patient_last_rolling_memory_update_turn", 0)

    result = {}
    if turn_count != last_memory_turn:
        result = rolling_memory_node(current_session, config=config, runtime=runtime)
    else:
        logger.info(f"patient_rolling_memory_trigger: skipped, memory already updated on turn {turn_count}")

    # Return updated patient_latent_variables from the rolling memory output (if any shift occurred)
    if result and "patient_latent_variables" in result:
        updated_dict: Dict[str, Any] = {"patient_latent_variables": result["patient_latent_variables"]}
        
        from ..utils.config import (
            ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES,
            DISABLE_COGNITIVE_INTERPRETER_IMPACT,
            VANILLA_BASELINE,
            CONSISTENT_CLIENT_BASELINE,
            SIMPATIENT_BASELINE,
            PATIENT_PSI_BASELINE,
        )
        ci_active = not (
            ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES
            or DISABLE_COGNITIVE_INTERPRETER_IMPACT
            or VANILLA_BASELINE
            or CONSISTENT_CLIENT_BASELINE
            or SIMPATIENT_BASELINE
            or PATIENT_PSI_BASELINE
        )
        if ci_active:
            memory_trajectory_entry: LatentTrajectoryEntry = {
                "step_index": -1,
                "component": "memory",
                "session_before": session_number,
                "session_after": session_number + 1,
                "anger": float(result["patient_latent_variables"]["anger"]),
                "self_efficacy": float(result["patient_latent_variables"]["self_efficacy"]),
                "problem_recognition": float(result["patient_latent_variables"]["problem_recognition"]),
                "motivational_readiness": float(result["patient_latent_variables"]["motivational_readiness"]),
                "details": result.get("macro_shift_details"),
            }
            parent_trajectory = list(state.get("latent_variable_trajectory", []))
            parent_trajectory.append(memory_trajectory_entry)
            updated_dict["latent_variable_trajectory"] = parent_trajectory

        return updated_dict
    return {}


@traced_span("therapist_rolling_memory_trigger")
def therapist_rolling_memory_trigger(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    """
    Inter-session node that consolidates the remaining turns for the therapist.
    Runs after patient session consolidation but before between_session_event.
    """
    session_number = state["current_session_number"]
    logger.info(f"therapist_rolling_memory_trigger: updating memory for session {session_number}")
    
    current_session = state["current_session"]
    turn_count = current_session.get("turn_count", 0)
    last_memory_turn = current_session.get("therapist_last_rolling_memory_update_turn", 0)
    
    if turn_count != last_memory_turn:
        therapist_rolling_memory_node(current_session, config=config, runtime=runtime)
    else:
        logger.info(f"therapist_rolling_memory_trigger: skipped, memory already updated on turn {turn_count}")

    return {}


@traced_span("increment_session")
def increment_session_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Increment session counter and accumulate session data.
    
    Archives completed session data (including summaries)
    and resets current_session for next iteration.
    
    Args:
        state: Parent MultiSessionState
        config: Configuration dictionary
        
    Returns:
        Updated MultiSessionState ready for next session
    """
    logger = logging.getLogger(__name__)
    
    current_session = state["current_session"]
    current_number = state["current_session_number"]
    
    logger.info(f"increment_session_node: archiving session {current_number}")
    
    enrich_span(
        node_name="increment_session",
        description=f"Archive session {current_number} results and prepare session {current_number + 1}",
        session_number=current_number,
        phase="orchestration",
        purpose="session_transition",
        thread_id=config.get("configurable", {}).get("thread_id"),
    )
    
    # Accumulate session data
    session_turns = list(current_session.get("current_session_turns", []))
    from ..utils.config import CAUSAL_RECORD_BASELINE
    if CAUSAL_RECORD_BASELINE:
        psr = current_session.get("patient_state_reasoning")
        if psr:
            for tr in reversed(session_turns):
                if tr.get("speaker") == "patient":
                    tr["session_appraisal_package"] = psr
                    break

    updated_session_turns = state["all_session_turns"] + [session_turns]
    updated_lv_logs = state["all_latent_variable_logs"] + [current_session["latent_variable_logs"]]

    # Latent variables simply carry forward to the next session.
    # Between-Session Event macro-shifts (the "Sleeper Effect") are handled exclusively 
    # by the BetweenSessionEventAgent before this node runs, modifying the parent state.
    lv = state.get("patient_latent_variables", current_session["patient_latent_variables"]).copy()

    # Reset current_session for next iteration
    next_session_number = current_number + 1
    miin_injection_schedule = state.get("miin_injection_schedule", {})
    next_session_is_miin_target = next_session_number in miin_injection_schedule
    next_session_injection_turns = miin_injection_schedule.get(next_session_number, [])
    reset_session: ConversationState = {
        "messages": [],
        "current_speaker": "",
        "session_ended": False,
        "session_end_reason": None,
        "turn_count": 0,
        "patient_system_prompt": "",
        "session_number": next_session_number,
        "current_session_turns": [],
        "miin_injection_turn_numbers": next_session_injection_turns,
        "miin_injection_active": next_session_is_miin_target,
        "last_used_techniques": [],
        "patient_latent_variables": lv,
        "latent_variable_logs": [],
        "session_latent_trajectory": [],
        "consistent_client_state": current_session.get("consistent_client_state", {}),
        "cami_therapist_state": current_session.get("cami_therapist_state", {}),
    }

    logger.info(f"Session {current_number} archived. Preparing for session {next_session_number}")

    return {
        "current_session_number": current_number + 1,
        "all_session_turns": updated_session_turns,
        "all_latent_variable_logs": updated_lv_logs,
        "current_session": reset_session,
    }


def should_continue_therapy(state: Dict[str, Any]) -> str:
    """
    Conditional edge to determine if therapy should continue or end.
    
    Args:
        state: Parent MultiSessionState
        
    Returns:
        "continue" if more sessions remain, "end" if therapy is complete
    """
    logger = logging.getLogger(__name__)
    
    current_number = state["current_session_number"]
    
    if current_number <= NUM_SESSIONS:
        logger.info(f"Therapy continuing: session {current_number - 1}/{NUM_SESSIONS} completed")
        return "continue"
    else:
        logger.info(f"Therapy complete: all {NUM_SESSIONS} sessions finished")
        return "end"


def route_after_run_single_session(state: Dict[str, Any]) -> str:
    """
    Conditional edge after run_single_session.
    Skips memory consolidation, clinician note, and BSE on the final session.
    """
    logger = logging.getLogger(__name__)
    current_number = state["current_session_number"]
    
    if current_number >= NUM_SESSIONS:
        logger.info(f"Routing: final session {current_number} complete, skipping inter-session memory and events.")
        return "increment_session"
    
    from ..utils.config import (
        CONSISTENT_CLIENT_BASELINE,
        VANILLA_BASELINE,
        SIMPATIENT_BASELINE,
        PATIENT_PSI_BASELINE,
        DISABLE_PATIENT_MEMORY,
    )
    if SIMPATIENT_BASELINE:
        return "simpatient_global_scores"
    if PATIENT_PSI_BASELINE:
        return "patient_psi_session_summary"
    if VANILLA_BASELINE:
        return "vanilla_client_memory"
    if CONSISTENT_CLIENT_BASELINE:
        return "consistent_client_memory"
    if DISABLE_PATIENT_MEMORY:
        return "patient_summary_memory"
    
    return "patient_rolling_memory_trigger"

def route_after_patient_inter_session_nodes(state: Dict[str, Any]) -> str:
    from ..utils.config import CAMI_THERAPIST_BASELINE
    if CAMI_THERAPIST_BASELINE:
        return "cami_therapist_memory"
    return "therapist_rolling_memory_trigger"

def route_after_single_session(state: Dict[str, Any]) -> str:
    """
    Conditional edge after run_single_session to determine next inter-session node.

    Routes to the between-session event generator if enabled and not the final session,
    otherwise skips directly to increment.

    Args:
        state: Parent MultiSessionState

    Returns:
        "event" if event generation enabled and session < NUM_SESSIONS,
        "skip" otherwise (final session or events disabled)
    """
    logger = logging.getLogger(__name__)

    from ..utils.config import (
        CONSISTENT_CLIENT_BASELINE,
        VANILLA_BASELINE,
        SIMPATIENT_BASELINE,
        PATIENT_PSI_BASELINE,
        ABLATE_LIFE_EVENT_SIMULATOR,
    )
    if SIMPATIENT_BASELINE:
        logger.info("Routing: SimPatient baseline manages its own between-session events, skipping life event simulator.")
        return "skip"
    if PATIENT_PSI_BASELINE:
        logger.info("Routing: patient_psi baseline used, skipping life event simulator.")
        return "skip"
    if CONSISTENT_CLIENT_BASELINE:
        logger.info("Routing: consistent baseline client used, skipping life event simulator.")
        return "skip"
    if VANILLA_BASELINE:
        logger.info("Routing: vanilla baseline used, skipping life event simulator.")
        return "skip"
    if ABLATE_LIFE_EVENT_SIMULATOR:
        logger.info("Routing: ablation study active, skipping life event simulator.")
        return "skip"

    current_number = state["current_session_number"]
    if current_number < NUM_SESSIONS and ENABLE_BETWEEN_SESSION_EVENTS:
        logger.info(f"Routing: between-session event generation for session {current_number}")
        return "event"
    logger.info(f"Routing: skipping inter-session nodes (session={current_number}, event={ENABLE_BETWEEN_SESSION_EVENTS})")
    return "skip"


class MultiSessionGraphBuilder:
    """
    Builder for multi-session (4 sessions) therapy graph.
    
    Creates a parent LangGraph StateGraph that orchestrates 4 sequential
    therapy sessions, accumulating results across sessions.
    """
    
    def __init__(self, single_session_graph, checkpointer=None, store=None, log_level: int = logging.INFO):
        """
        Initialize the multi-session graph builder.

        Args:
            single_session_graph: Compiled single-session graph to use as subgraph
            checkpointer: Optional LangGraph checkpointer for state persistence.
                          If None, uses MemorySaver (in-memory, lost on crash).
                          Pass a SqliteSaver or PostgresSaver for disk persistence.
            store: LangGraph Store for cross-session memory (diary, decisional balance,
                   therapist memory lists).  Defaults to InMemoryStore if not provided.
                   Pass an InMemoryStore pre-populated with initial values for the run.
            log_level: Logging level for graph operations
        """
        self.single_session_graph = single_session_graph
        self.checkpointer = checkpointer
        self.store = store if store is not None else InMemoryStore()
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
    
    def build(self):
        """
        Build and compile the multi-session therapy graph.

        Graph structure:
            START -> run_single_session -> patient_session_consolidation -> clinical_documenter -> (conditional)
                                                                                ├─> life_event_simulator -> increment_session (events enabled, session < 4)
                                                                                └─> increment_session (final session or events disabled)
                                                                                        -> (conditional)
                                                                                            ├─> run_single_session (loop)
                                                                                            └─> END

        Between-session event runs after sessions 1–3 only when ENABLE_BETWEEN_SESSION_EVENTS is true.

        Returns:
            Compiled StateGraph for 4-session therapy
        """
        self.logger.info("Building multi-session therapy graph")

        # Initialize parent graph with MultiSessionState
        workflow = StateGraph(MultiSessionState)

        # Add nodes
        workflow.add_node("run_single_session", run_single_session_node)
        workflow.add_node("patient_summary_memory", patient_summary_memory_node)
        workflow.add_node("patient_summary_inter_sessions", patient_summary_inter_sessions_node)
        workflow.add_node("consistent_client_memory", consistent_client_memory_node)
        workflow.add_node("consistent_client_inter_sessions", consistent_client_inter_sessions_node)
        workflow.add_node("vanilla_client_memory", vanilla_client_memory_node)
        workflow.add_node("vanilla_client_inter_sessions", vanilla_client_inter_sessions_node)
        workflow.add_node("patient_psi_session_summary", patient_psi_session_summary_node)
        workflow.add_node("patient_psi_inter_sessions", patient_psi_inter_sessions_node)
        workflow.add_node("simpatient_global_scores", simpatient_global_scores_node)
        workflow.add_node("simpatient_session_summary", simpatient_session_summary_node)
        workflow.add_node("simpatient_between_session_event", simpatient_between_session_event_node)
        workflow.add_node("simpatient_inter_sessions", simpatient_inter_sessions_node)
        workflow.add_node("cami_therapist_memory", cami_therapist_memory_node)
        workflow.add_node("cami_therapist_inter_sessions", cami_therapist_inter_sessions_node)
        workflow.add_node("patient_rolling_memory_trigger", patient_rolling_memory_trigger)
        workflow.add_node("therapist_rolling_memory_trigger", therapist_rolling_memory_trigger)
        workflow.add_node("life_event_simulator", between_session_event_node)
        workflow.add_node("increment_session", increment_session_node)

        self.logger.info("Nodes added to parent graph")

        # Set entry point
        workflow.set_entry_point("run_single_session")

        # Conditional edge from run_single_session: memory or skip to increment on final session
        workflow.add_conditional_edges(
            "run_single_session",
            route_after_run_single_session,
            {
                "patient_rolling_memory_trigger": "patient_rolling_memory_trigger",
                "patient_summary_memory": "patient_summary_memory",
                "consistent_client_memory": "consistent_client_memory",
                "vanilla_client_memory": "vanilla_client_memory",
                "patient_psi_session_summary": "patient_psi_session_summary",
                "simpatient_global_scores": "simpatient_global_scores",
                "increment_session": "increment_session",
            }
        )

        # Patient summary memory edges (DISABLE_PATIENT_MEMORY ablation)
        workflow.add_edge("patient_summary_memory", "patient_summary_inter_sessions")
        workflow.add_conditional_edges("patient_summary_inter_sessions", route_after_patient_inter_session_nodes, {
            "cami_therapist_memory": "cami_therapist_memory",
            "therapist_rolling_memory_trigger": "therapist_rolling_memory_trigger"
        })

        # Patient-Ψ baseline edges
        workflow.add_edge("patient_psi_session_summary", "patient_psi_inter_sessions")
        workflow.add_conditional_edges("patient_psi_inter_sessions", route_after_patient_inter_session_nodes, {
            "cami_therapist_memory": "cami_therapist_memory",
            "therapist_rolling_memory_trigger": "therapist_rolling_memory_trigger"
        })

        # SimPatient baseline edges
        workflow.add_edge("simpatient_global_scores", "simpatient_session_summary")
        workflow.add_edge("simpatient_session_summary", "simpatient_between_session_event")
        workflow.add_edge("simpatient_between_session_event", "simpatient_inter_sessions")
        workflow.add_conditional_edges("simpatient_inter_sessions", route_after_patient_inter_session_nodes, {
            "cami_therapist_memory": "cami_therapist_memory",
            "therapist_rolling_memory_trigger": "therapist_rolling_memory_trigger"
        })

        # Consistent client edges
        workflow.add_edge("consistent_client_memory", "consistent_client_inter_sessions")
        workflow.add_conditional_edges("consistent_client_inter_sessions", route_after_patient_inter_session_nodes, {
            "cami_therapist_memory": "cami_therapist_memory",
            "therapist_rolling_memory_trigger": "therapist_rolling_memory_trigger"
        })

        # Vanilla baseline edges
        workflow.add_edge("vanilla_client_memory", "vanilla_client_inter_sessions")
        workflow.add_conditional_edges("vanilla_client_inter_sessions", route_after_patient_inter_session_nodes, {
            "cami_therapist_memory": "cami_therapist_memory",
            "therapist_rolling_memory_trigger": "therapist_rolling_memory_trigger"
        })

        # Direct edge: patient_rolling_memory_trigger -> therapist_rolling_memory_trigger or cami
        workflow.add_conditional_edges("patient_rolling_memory_trigger", route_after_patient_inter_session_nodes, {
            "cami_therapist_memory": "cami_therapist_memory",
            "therapist_rolling_memory_trigger": "therapist_rolling_memory_trigger"
        })
        
        # Cami therapist edges
        workflow.add_edge("cami_therapist_memory", "cami_therapist_inter_sessions")
        workflow.add_conditional_edges("cami_therapist_inter_sessions", route_after_single_session, {
            "event": "life_event_simulator",
            "skip": "increment_session"
        })

        # Conditional edge from therapist_rolling_memory_trigger: event or skip to increment
        workflow.add_conditional_edges(
            "therapist_rolling_memory_trigger",
            route_after_single_session,
            {
                "event": "life_event_simulator",
                "skip": "increment_session",
            }
        )

        # Direct edge from life_event_simulator to increment
        workflow.add_edge("life_event_simulator", "increment_session")

        # Conditional edge from increment to continue or end
        workflow.add_conditional_edges(
            "increment_session",
            should_continue_therapy,
            {
                "continue": "run_single_session",
                "end": END
            }
        )

        compiled_graph = workflow.compile(checkpointer=self.checkpointer, store=self.store)
        self.logger.info("Multi-session graph compiled successfully")

        return compiled_graph