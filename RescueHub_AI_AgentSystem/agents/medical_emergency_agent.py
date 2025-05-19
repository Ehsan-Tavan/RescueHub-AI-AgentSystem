from langgraph.graph import StateGraph, END
import logging
from typing import Dict

from .state import State
from RescueHub_AI_AgentSystem.nodes import get_conversation_node, get_history_node

logger = logging.getLogger(__name__)


def create_medical_emergency_agent(model_config: Dict[str, str]):
    logger.info("Creating medical emergency agent.")

    # Initialize graph
    workflow = StateGraph(State)

    # Get nodes
    conversation_node = get_conversation_node(model_config, agent_name="medical_emergency_agent")
    history_node = get_history_node()

    # Add nodes to workflow
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("history", history_node)

    # Set edges
    workflow.set_entry_point("conversation")
    workflow.add_edge("conversation", "history")
    workflow.add_edge("history", END)

    # Compile
    app = workflow.compile(debug=False)
    logger.info("Conversation graph created")

    return app
