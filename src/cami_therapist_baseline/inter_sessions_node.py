import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from src.utils.store import THERAPIST_MEMORY_NS

logger = logging.getLogger(__name__)

def cami_therapist_inter_sessions_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    logger.info("cami_therapist_inter_sessions_node: transferring memory to parent state.")
    return {}
