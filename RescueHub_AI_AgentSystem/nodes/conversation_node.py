from typing import Dict, List
import logging

from RescueHub_AI_AgentSystem.prompts import (get_fire_emergency_prompt, get_medical_emergency_prompt,
                                              get_router_agent_prompt)
from RescueHub_AI_AgentSystem.agents import State
from .load_generative_model import load_generative_model

logger = logging.getLogger(__name__)


class ConversationNode:
    def __init__(self, llm_chain, agent_name):
        self.runnable = llm_chain
        self.agent_name = agent_name

    def __call__(self, state: State) -> Dict[str, List[str]]:
        logger.debug({
            "message": f"The LLM is generating conversation response. {self.agent_name}",
            "state": state
        })

        response = self.runnable.invoke({
            "history": state.get("chat_history", []),
            "question": state["query"]
        })

        content = response.content.strip()

        agent_names = state.get("agent_name", [])

        if "[[agent_name:" in content:
            try:
                agent_name = content.split("[[agent_name:")[1].split("]]")[0].strip()
                content = content.replace(f"[[agent_name:{agent_name}]]", "").strip()
                agent_names.append(agent_name)
            except Exception:
                pass

        return {
            "answer": content,
            "agent_name": agent_names,
        }


def get_conversation_node(
        model_config: Dict[str, str],
        agent_name: str
) -> ConversationNode:
    logger.info("Getting the conversation node")

    agent_name_2_prompt_function = {
        "fire_emergency_agent": get_fire_emergency_prompt,
        "medical_emergency_agent": get_medical_emergency_prompt,
        "router_agent": get_router_agent_prompt,
    }

    conversation_prompt = agent_name_2_prompt_function[agent_name]()

    # Load the model
    model = load_generative_model(model_config)

    # Create the chain
    conversation_chain = conversation_prompt | model

    # Create and return the node
    conversation_node = ConversationNode(conversation_chain, agent_name)
    logger.info("The conversation node has been created")
    return conversation_node
