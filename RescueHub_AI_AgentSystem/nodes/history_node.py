from typing import Dict, List
import logging
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage

from RescueHub_AI_AgentSystem.agents import State

logger = logging.getLogger(__name__)


class HistoryNode:
    def __call__(self, state: State) -> Dict[str, List[BaseMessage]]:
        logger.debug({
            "message": "The history is updated.",
            "state": state
        })

        history = state.get("chat_history", [])
        updated_history = history + [
            HumanMessage(content=state["query"]),
            AIMessage(content=state["answer"]),
        ]
        return {
            "chat_history": updated_history
        }


def get_history_node(
) -> HistoryNode:
    return HistoryNode()
