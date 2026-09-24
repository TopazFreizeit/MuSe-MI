"""
Single-session therapy conversation graph.
"""
from typing import Dict, Any
import logging
from langgraph.graph import StateGraph, END

from ..patient.patient_agent import patient_node
from ..consistent_client_baseline.client_node import consistent_client_node
from ..vanilla_baseline.client_node import vanilla_client_node
from ..simpatient_baseline.client_node import simpatient_client_node
from ..patient_psi_baseline.client_node import patient_psi_client_node
from ..cami_therapist_baseline.therapist_node import cami_therapist_node
from ..therapist.therapist_agent import therapist_node
from ..therapist.mina_therapist_agent import mina_therapist_node
from ..session_end_identifier.session_end_detector import end_session_node
from ..patient_state_manager import unified_state_update_node, rolling_memory_node
from ..therapist.therapist_rolling_memory import therapist_rolling_memory_node
from ..causal_study import causal_shuffled_bank_node
from ..patient.patient_config import PATIENT_CONTEXT_WINDOW_TURNS
from ..therapist.therapist_config import CONTEXT_WINDOW as THERAPIST_CONTEXT_WINDOW
from .graph_config import ConversationState, MAX_TURNS
from ..utils.config import ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES

logger = logging.getLogger(__name__)

def therapist_router(state: Dict[str, Any]) -> str:
    """Routes to the correct therapist node based on the MIIN injection schedule."""
    from ..utils.config import CAMI_THERAPIST_BASELINE
    if CAMI_THERAPIST_BASELINE:
        return "cami_therapist"

    turn_number = state.get("turn_count", 0) + 1
    miin_active = state.get("miin_injection_active", False)
    injection_turns = state.get("miin_injection_turn_numbers", [])
    
    if miin_active and turn_number in injection_turns:
        return "adversarial_therapist"
    return "therapist"


def should_end_session(state: Dict[str, Any]) -> str:
    """
    Conditional edge function to determine next node.

    Args:
        state: Current conversation state (ConversationState)

    Returns:
        Next node name: "end_session", "patient_rolling_memory", "therapist_rolling_memory", "patient", "consistent_client", "therapist", "cami_therapist" or "adversarial_therapist"
    """
    logger = logging.getLogger(__name__)
    session_number = state.get("session_number", 1)
    logger.info(f"[Session {session_number}] Routing: checking session state")

    # Check if session already ended
    if state["session_ended"]:
        logger.info(f"[Session {session_number}] Routing: session_ended flag set, routing to end_session")
        return "end_session"

    # Check turn limit (safety)
    if state["turn_count"] >= MAX_TURNS:
        logger.warning(f"[Session {session_number}] Routing: max turns reached ({MAX_TURNS}), ending session")
        return "end_session"

    from ..utils.config import (
        CONSISTENT_CLIENT_BASELINE,
        CAMI_THERAPIST_BASELINE,
        VANILLA_BASELINE,
        SIMPATIENT_BASELINE,
        PATIENT_PSI_BASELINE,
        DISABLE_PATIENT_MEMORY,
    )
    # Route to opposite speaker
    if state["current_speaker"] == "therapist":
        if VANILLA_BASELINE:
            next_speaker = "vanilla_client"
        elif CONSISTENT_CLIENT_BASELINE:
            next_speaker = "consistent_client"
        elif SIMPATIENT_BASELINE:
            next_speaker = "simpatient_client"
        elif PATIENT_PSI_BASELINE:
            next_speaker = "patient_psi_client"
        else:
            next_speaker = "patient"
    else:
        next_speaker = "therapist"
    
    if next_speaker == "therapist" and state["turn_count"] > 0:
        trigger_patient = (
            not CONSISTENT_CLIENT_BASELINE
            and not VANILLA_BASELINE
            and not SIMPATIENT_BASELINE
            and not PATIENT_PSI_BASELINE
            and not DISABLE_PATIENT_MEMORY
            and state["turn_count"] % (PATIENT_CONTEXT_WINDOW_TURNS * 2) == 0
        )
        trigger_therapist = (
            not CAMI_THERAPIST_BASELINE 
            and state["turn_count"] % (THERAPIST_CONTEXT_WINDOW * 2) == 0 
            and state["turn_count"] != state.get("therapist_last_rolling_memory_update_turn", 0)
        )

        if trigger_patient:
            logger.info(f"[Session {session_number}] Routing: Turn {state['turn_count']} reached, triggering patient_rolling_memory")
            return "patient_rolling_memory"

        if trigger_therapist:
            logger.info(f"[Session {session_number}] Routing: Turn {state['turn_count']} reached, triggering therapist_rolling_memory")
            return "therapist_rolling_memory"

    if next_speaker == "therapist":
        logger.info(f"[Session {session_number}] Routing: session continues, next speaker is therapist. Routing via therapist_router.")
        return therapist_router(state)

    logger.info(f"[Session {session_number}] Routing: session continues, next speaker: {next_speaker}")
    return next_speaker

# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_patient_rolling_memory(state: Dict[str, Any]) -> str:
    """Routes back to the correct therapist node after patient rolling memory, chaining therapist memory if needed."""
    from ..utils.config import CAMI_THERAPIST_BASELINE
    
    if not CAMI_THERAPIST_BASELINE and state["turn_count"] > 0 and state["turn_count"] % (THERAPIST_CONTEXT_WINDOW * 2) == 0:
        return "therapist_rolling_memory"
    return therapist_router(state)


def route_after_therapist_rolling_memory(state: Dict[str, Any]) -> str:
    """Routes back to the correct therapist node after therapist rolling memory."""
    return therapist_router(state)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class SingleSessionGraphBuilder:
    """
    Builder for single-session therapy conversation graph.

    Creates a LangGraph StateGraph that manages one therapy session from
    start to finish, including therapist-patient exchanges, session end
    detection, and session evaluation.
    """

    def __init__(self, store=None, log_level: int = logging.INFO):
        """
        Initialize the graph builder.

        Args:
            store: LangGraph Store to compile into the graph so that nodes
                   accessing ``runtime.store`` receive the shared store instance.
                   Must be the same Store passed to MultiSessionGraphBuilder.
            log_level: Logging level for graph operations
        """
        self.store = store
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def build(self):
        """
        Build and compile the single-session therapy graph.
        """
        self.logger.info("Building single-session therapy graph")

        # Initialize graph
        workflow = StateGraph(ConversationState)

        # ------------------------------------------------------------------
        # Add nodes
        # ------------------------------------------------------------------
        workflow.add_node("therapist", therapist_node)
        workflow.add_node("cami_therapist", cami_therapist_node)
        workflow.add_node("adversarial_therapist", mina_therapist_node)
        workflow.add_node("patient", patient_node)
        workflow.add_node("consistent_client", consistent_client_node)
        workflow.add_node("vanilla_client", vanilla_client_node)
        workflow.add_node("simpatient_client", simpatient_client_node)
        workflow.add_node("patient_psi_client", patient_psi_client_node)
        workflow.add_node("end_session", end_session_node)
        workflow.add_node("cognitive_interpreter", unified_state_update_node)
        workflow.add_node("causal_shuffled_bank", causal_shuffled_bank_node)
        workflow.add_node("patient_rolling_memory", rolling_memory_node)
        workflow.add_node("therapist_rolling_memory", therapist_rolling_memory_node)

        # Set entry point
        workflow.add_conditional_edges(
            "__start__",
            therapist_router,
            {
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
            }
        )

        # Direct edge from end_session to END
        workflow.add_edge("end_session", END)

        # ------------------------------------------------------------------
        # After therapist: route to cognitive_interpreter or bypass
        # ------------------------------------------------------------------
        def route_after_therapist(state):
            from ..utils.config import (
                CONSISTENT_CLIENT_BASELINE,
                ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES,
                DISABLE_COGNITIVE_INTERPRETER_IMPACT,
                VANILLA_BASELINE,
                SIMPATIENT_BASELINE,
                DISABLE_COGNITIVE_INTERPRETER_IMPACT,
                VANILLA_BASELINE,
                SIMPATIENT_BASELINE,
                PATIENT_PSI_BASELINE,
                CAUSAL_SHUFFLED_ROLLOUT,
            )
            if VANILLA_BASELINE:
                return should_end_session(state)
            if CONSISTENT_CLIENT_BASELINE:
                return should_end_session(state)
            if SIMPATIENT_BASELINE:
                return should_end_session(state)
            if PATIENT_PSI_BASELINE:
                return should_end_session(state)
            if ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES:
                return should_end_session(state)
            if DISABLE_COGNITIVE_INTERPRETER_IMPACT:
                return should_end_session(state)
            if CAUSAL_SHUFFLED_ROLLOUT:
                return "causal_shuffled_bank"
            return "cognitive_interpreter"

        workflow.add_conditional_edges(
            "therapist",
            route_after_therapist,
            {
                "patient": "patient",
                "consistent_client": "consistent_client",
                "vanilla_client": "vanilla_client",
                "simpatient_client": "simpatient_client",
                "patient_psi_client": "patient_psi_client",
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
                "cognitive_interpreter": "cognitive_interpreter",
                "causal_shuffled_bank": "causal_shuffled_bank",
            }
        )
        
        workflow.add_conditional_edges(
            "cami_therapist",
            route_after_therapist,
            {
                "patient": "patient",
                "consistent_client": "consistent_client",
                "vanilla_client": "vanilla_client",
                "simpatient_client": "simpatient_client",
                "patient_psi_client": "patient_psi_client",
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
                "cognitive_interpreter": "cognitive_interpreter",
                "causal_shuffled_bank": "causal_shuffled_bank",
            }
        )
        
        workflow.add_conditional_edges(
            "adversarial_therapist",
            route_after_therapist,
            {
                "patient": "patient",
                "consistent_client": "consistent_client",
                "vanilla_client": "vanilla_client",
                "simpatient_client": "simpatient_client",
                "patient_psi_client": "patient_psi_client",
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
                "cognitive_interpreter": "cognitive_interpreter",
                "causal_shuffled_bank": "causal_shuffled_bank",
            }
        )

        # ------------------------------------------------------------------
        # After patient_state_manager: delegate to should_end_session
        # ------------------------------------------------------------------
        workflow.add_conditional_edges(
            "cognitive_interpreter",
            should_end_session,
            {
                "patient": "patient",
                "consistent_client": "consistent_client",
                "vanilla_client": "vanilla_client",
                "simpatient_client": "simpatient_client",
                "patient_psi_client": "patient_psi_client",
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
            },
        )

        workflow.add_conditional_edges(
            "causal_shuffled_bank",
            should_end_session,
            {
                "patient": "patient",
                "consistent_client": "consistent_client",
                "vanilla_client": "vanilla_client",
                "simpatient_client": "simpatient_client",
                "patient_psi_client": "patient_psi_client",
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
            },
        )

        # ------------------------------------------------------------------
        # After patient: delegate to should_end_session 
        # ------------------------------------------------------------------
        workflow.add_conditional_edges(
            "patient",
            should_end_session,
            {
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
            }
        )
        
        workflow.add_conditional_edges(
            "consistent_client",
            should_end_session,
            {
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
            }
        )

        workflow.add_conditional_edges(
            "vanilla_client",
            should_end_session,
            {
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
            }
        )

        workflow.add_conditional_edges(
            "simpatient_client",
            should_end_session,
            {
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
            }
        )

        workflow.add_conditional_edges(
            "patient_psi_client",
            should_end_session,
            {
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "end_session": "end_session",
                "patient_rolling_memory": "patient_rolling_memory",
                "therapist_rolling_memory": "therapist_rolling_memory",
            }
        )

        workflow.add_conditional_edges(
            "patient_rolling_memory",
            route_after_patient_rolling_memory,
            {
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
                "therapist_rolling_memory": "therapist_rolling_memory",
            }
        )

        workflow.add_conditional_edges(
            "therapist_rolling_memory",
            route_after_therapist_rolling_memory,
            {
                "therapist": "therapist",
                "cami_therapist": "cami_therapist",
                "adversarial_therapist": "adversarial_therapist",
            }
        )

        compiled_graph = workflow.compile(store=self.store)
        self.logger.info("Single-session graph compiled successfully")

        return compiled_graph
