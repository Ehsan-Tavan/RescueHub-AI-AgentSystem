from typing import Dict
import logging

from RescueHub_AI_AgentSystem.prompts import get_fire_emergency_prompt, get_medical_emergency_prompt
from RescueHub_AI_AgentSystem.agents import State
from .load_generative_model import load_generative_model

logger = logging.getLogger(__name__)


class ConversationNode:
    def __init__(self, llm_chain):
        self.runnable = llm_chain

    def __call__(self, state: State) -> Dict[str, str]:
        logger.debug({
            "message": "The LLM is generating conversation response",
            "state": state
        })

        response = self.runnable.invoke({
            "history": state.get("chat_history", []),
            "question": state["query"]
        })

        return {"answer": response.content}


def get_conversation_node(
        model_config: Dict[str, str],
        agent_name: str
) -> ConversationNode:
    logger.info("Getting the conversation node")

    agent_name_2_prompt_function = {
        "fire_emergency_agent": get_fire_emergency_prompt,
        "medical_emergency_agent": get_medical_emergency_prompt,
    }

    conversation_prompt = agent_name_2_prompt_function[agent_name]()

    # Load the model
    model = load_generative_model(model_config)

    # Create the chain
    conversation_chain = conversation_prompt | model

    # Create and return the node
    conversation_node = ConversationNode(conversation_chain)
    logger.info("The conversation node has been created")
    return conversation_node
