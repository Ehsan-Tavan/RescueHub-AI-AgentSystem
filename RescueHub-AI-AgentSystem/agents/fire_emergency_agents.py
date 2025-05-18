from langgraph.graph import StateGraph, END
import logging
from typing import Dict

from .state import State
from nodes import get_conversation_node

logger = logging.getLogger(__name__)


def create_fire_emergency_agent(model_config: Dict[str, str]):
    logger.info("Creating fire emergency agent.")

    # Initialize graph
    workflow = StateGraph(State)

    # Add conversation node
    logger.info("Adding conversation node.")
    workflow.add_node("conversation", get_conversation_node(model_config))

    # Set direct path
    workflow.set_entry_point("conversation")
    workflow.add_edge("conversation", END)

    # Compile
    app = workflow.compile(debug=False)
    logger.info("Conversation graph created")

    return app