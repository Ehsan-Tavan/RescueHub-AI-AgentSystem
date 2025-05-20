from typing import Dict
import logging

from RescueHub_AI_AgentSystem.prompts import get_exit_summary_prompt
from RescueHub_AI_AgentSystem.agents import State
from .load_generative_model import load_generative_model

logger = logging.getLogger(__name__)


class ExitSummaryNode:
    def __init__(self, llm_chain):
        self.runnable = llm_chain

    def __call__(self, state: State) -> Dict:
        logger.debug({
            "message": "Generating exit summary from chat history.",
            "state": state
        })

        response = self.runnable.invoke({
            "history": state.get("chat_history", [])
        })

        try:
            # Eval response as dictionary (be cautious in production)
            summary_dict = eval(response.content.strip())
            if isinstance(summary_dict, dict):
                return {"summary": summary_dict}
        except Exception as e:
            logger.error(f"Failed to parse summary: {e}")

        return {"summary":
            {
                "incident_type": "unknown",
                "location": "",
                "patient_condition": "",
                "requires_medical_emergency": "no",
                "requires_fire_response": "no",
                "hazards": []
            }
        }


def get_exit_summary_node(model_config: Dict[str, str]) -> ExitSummaryNode:
    logger.info("Getting the exit summary node")

    prompt = get_exit_summary_prompt()
    model = load_generative_model(model_config)
    summary_chain = prompt | model

    return ExitSummaryNode(summary_chain)
