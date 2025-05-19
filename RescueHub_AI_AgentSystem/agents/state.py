from typing import TypedDict, List
from langchain_core.messages import BaseMessage


class State(TypedDict):
    query: str
    answer: str
    chat_history: List[BaseMessage]
