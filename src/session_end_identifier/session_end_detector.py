"""
Session end identifier logic.
Sessions end when MAX_TURNS is reached (reason code 2).
"""
import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from ..utils.tracing import enrich_span, traced_span
# Configure logging
logger = logging.getLogger(__name__)


class SessionEndDetector:
    """Provides session-end reason descriptions."""

    @classmethod
    def get_reason_description(cls, reason_code: int) -> str:
        """
        Get a human-readable description of the session end reason.

        Args:
            reason_code: The reason code (currently only 2)

        Returns:
            Description of the reason
        """
        reasons = {
            2: "Max turns reached",
        }
        return reasons.get(reason_code, "Unknown reason")


@traced_span("end_session")
def end_session_node(state: Dict, config: RunnableConfig) -> Dict:
    """
    Node that marks session as ended and determines the end reason.
    
    Reason codes:
        2 — Max turns reached (sole termination mechanism)
    
    Args:
        state: Current conversation state (ConversationState)
        config: Configuration including parent context
        
    Returns:
        State with session_ended and session_end_reason updated
    """
    session_number = state.get('session_number', 1)
    turn_count = state.get('turn_count', 0)
    
    enrich_span(
        node_name="end_session",
        description=f"End session {session_number} after {turn_count} turns",
        session_number=session_number,
        turn_number=turn_count,
        phase="conversation",
        purpose="session_termination",
    )
    
    # Sessions end via max turns only (reason code 2)
    session_end_reason = 2
    reason_desc = SessionEndDetector.get_reason_description(session_end_reason)
    
    logger.info(f"[Session {session_number}] session ended: turns={turn_count}, messages={len(state['messages'])}, reason={reason_desc}")
    
    return {
        "session_ended": True,
        "session_end_reason": session_end_reason
    }
