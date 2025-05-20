from typing import Dict
import json
import os
from datetime import datetime
from langgraph.graph import StateGraph, END
import logging

from .state import State
from RescueHub_AI_AgentSystem.nodes import get_conversation_node, get_history_node, get_exit_summary_node
from RescueHub_AI_AgentSystem.agents import create_fire_emergency_agent, create_medical_emergency_agent

logger = logging.getLogger(__name__)


def is_exit_input(state: State) -> bool:
    user_input = state.get("query", "").strip().lower()
    return user_input in {"exit", "quit"}


def route_from_router_conversation(state: State) -> str:
    if is_exit_input(state):
        return "summary_node"
    return route_by_agent(state)


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


def mock_save_ticket_tool(state: State) -> dict:
    summary = state.get("summary", {})
    logger.info(f"[TICKET TOOL] Saving ticket with summary: {summary}")

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"ticket_{timestamp}.json"
    save_path = os.path.join("saved_tickets", filename)

    # Ensure the output directory exists
    os.makedirs("saved_tickets", exist_ok=True)

    # Save the summary as a JSON file
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    ticket_id = f"TICKET_{timestamp}"

    return {
        "ticket_id": ticket_id,
        "message": f"Ticket saved to {filename}"
    }


def create_router_agent(model_config: Dict[str, str]):
    logger.info("Creating router agent.")

    # Initialize graph
    workflow = StateGraph(State)

    # Get nodes
    conversation_node = get_conversation_node(model_config, agent_name="router_agent")
    summary_node = get_exit_summary_node(model_config)
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
    workflow.add_node("summary_node", summary_node)
    workflow.add_node("save_ticket_tool", mock_save_ticket_tool)

    workflow.add_conditional_edges("entry_router", route_from_router_conversation, {
        "summary_node": "summary_node",
        "router": "router_conversation",
        "fire": "fire_agent",
        "medical": "medical_agent"
    })

    workflow.set_entry_point("entry_router")
    workflow.add_edge("router_conversation", "router_history")
    workflow.add_edge("router_history", "end_point")
    workflow.add_edge("summary_node", "save_ticket_tool")
    workflow.add_edge("save_ticket_tool", "end_point")
    workflow.add_edge("fire_agent", "end_point")
    workflow.add_edge("medical_agent", "end_point")
    workflow.add_edge("end_point", END)

    # Compile
    app = workflow.compile(debug=False)
    logger.info("Conversation graph created")

    plot = app.get_graph().draw_mermaid_png()
    with open("RescueHub_AI_AgentSystem.png", "wb") as fp:
        fp.write(plot)

    return app
