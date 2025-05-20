from langgraph.graph import StateGraph, END
import logging
from typing import Dict

from .state import State
from RescueHub_AI_AgentSystem.nodes import get_conversation_node, get_history_node
from RescueHub_AI_AgentSystem.agents import create_fire_emergency_agent, create_medical_emergency_agent

logger = logging.getLogger(__name__)


def entry_router(state: State) -> dict:
    logger.info(f"[ENTRY] Starting state: {state}")
    detected_agents = state.get("agent_name", [])
    if not detected_agents:
        return {
            "agent_name": ["router"],
        }

    return {}


def end_point(state: State) -> dict:
    logger.info(f"End point: {state}")
    return {}


def route_by_agent(state: State) -> str:
    detected_agents = state.get("agent_name", [])

    last_agent = detected_agents[-1]

    if last_agent == "fire_emergency_agent":
        return "fire"
    elif last_agent == "medical_emergency_agent":
        return "medical"
    else:
        return "router"


# def check_agent_changing(state: State) -> str:
#     detected_agents = state.get("agent_name", [])
#     print("detected_agents")
#     print(detected_agents)
#     if len(detected_agents) <= 2:
#         return "not_change"
#     if detected_agents[-1] != detected_agents[-2]:
#         print(detected_agents[-1])
#         return detected_agents[-1]
#     else:
#         return "not_change"


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
    workflow.add_node("entry_router", entry_router)
    workflow.add_node("end_point", end_point)
    workflow.add_node("router_conversation", conversation_node)
    workflow.add_node("router_history", history_node)
    workflow.add_node("fire_agent", fire_agent)
    workflow.add_node("medical_agent", medical_agent)

    workflow.add_conditional_edges("entry_router", route_by_agent, {
        "router": "router_conversation",
        "fire": "fire_agent",
        "medical": "medical_agent"
    })

    workflow.add_conditional_edges("router_conversation", route_by_agent, {
        "router": "router_history",
        "fire": "fire_agent",
        "medical": "medical_agent"
    })

    # workflow.add_conditional_edges("fire_agent", check_agent_changing, {
    #     "not_change": "end_point",
    #     "fire_emergency_agent": "end_point",
    #     "medical_emergency_agent": "medical_agent"
    # })
    #
    # workflow.add_conditional_edges("medical_agent", check_agent_changing, {
    #     "not_change": "end_point",
    #     "fire_emergency_agent": "fire_agent",
    #     "medical_emergency_agent": "end_point"
    # })

    workflow.set_entry_point("entry_router")
    # workflow.add_edge("router_conversation", "router_history")
    workflow.add_edge("router_history", "end_point")
    workflow.add_edge("end_point", END)

    # Compile
    app = workflow.compile(debug=False)
    logger.info("Conversation graph created")

    return app
