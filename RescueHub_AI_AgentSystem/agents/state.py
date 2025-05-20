from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage


class State(TypedDict):
    query: str
    answer: str
    chat_history: Annotated[List[BaseMessage], operator.add]
    agent_name: List[str]
