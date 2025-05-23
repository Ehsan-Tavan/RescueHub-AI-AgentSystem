import argparse
import yaml

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from RescueHub_AI_AgentSystem.utils import setup_logger
from RescueHub_AI_AgentSystem.agents import create_router_agent

if __name__ == "__main__":
    LOGGER = setup_logger()
    PARSER = argparse.ArgumentParser(description="RescueHub_AI_AgentSystem")
    PARSER.add_argument("-c", "--config", default=None, type=str,
                        help="Config file path (default: None)")
    ARGS = PARSER.parse_args()

    if ARGS.config is None:
        raise ValueError("The config argument should be set!")

    CONFIG = yaml.safe_load(open(ARGS.config))
    LOGGER.debug(CONFIG)

    # Create the agent (LangGraph app)
    FIRE_AGENT = create_router_agent(model_config=CONFIG["generative_model"])

    # Initialize history
    history: list[BaseMessage] = []
    agent_names: list[str] = []

    print("💬 Fire Emergency Agent is ready! Type 'exit' to stop.\n")

    while True:
        query = input("🧑 You: ")
        # if query.lower() in {"exit", "quit"}:
        #     print("👋 Conversation ended.")
        #     break

        # Build input state
        state = {
            "query": query,
            "chat_history": history,
            "agent_name": agent_names,
        }

        # Invoke the LangGraph app
        state = FIRE_AGENT.invoke(state)

        if query.lower() in {"exit", "quit"}:
            print("👋 Conversation ended.")
            break

        # Print and update history
        answer = state["answer"]
        agent_name = agent_names[-1] if agent_names else "router"

        print(f"🤖 Agent ({agent_name}): {answer}\n")

        # Update history with new messages
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer))
        agent_names.append(state["agent_name"][-1])
