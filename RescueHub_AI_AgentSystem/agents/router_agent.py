from langgraph.graph import StateGraph, END
import logging
from typing import Dict

from .state import State
from RescueHub_AI_AgentSystem.nodes import get_conversation_node, get_history_node
from RescueHub_AI_AgentSystem.agents import create_fire_emergency_agent, create_medical_emergency_agent

logger = logging.getLogger(__name__)


def route_by_agent(state: State) -> str:
    if state.get("agent_name") == "fire_emergency_agent":
        return "fire"
    elif state.get("agent_name") == "medical_emergency_agent":
        return "medical"
    else:
        return "router"


def create_router_agent(model_config: Dict[str, str]):
    logger.info("Creating router agent.")

    # Initialize graph
    workflow = StateGraph(State)

    # Get nodes
    conversation_node = get_conversation_node(model_config, agent_name="router_agent")
    history_node = get_history_node()

    fire_agent = create_fire_emergency_agent(model_config)
    medical_agent = create_medical_emergency_agent(model_config)

    # Add nodes to workflow
    workflow.add_node("router_conversation", conversation_node)
    workflow.add_node("router_history", history_node)
    workflow.add_node("fire_agent", fire_agent)
    workflow.add_node("medical_agent", medical_agent)

    workflow.add_conditional_edges("router_conversation", route_by_agent, {
        "router": "router_history",
        "fire": "fire_agent",
        "medical": "medical_agent"
    })

    # workflow.add_edge("router_history", "router_conversation")
    workflow.set_entry_point("router_conversation")

    workflow.add_edge("router_history", END)

    # Compile
    app = workflow.compile(debug=False)
    logger.info("Conversation graph created")

    return app
