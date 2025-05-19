import os
import logging
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

ChatOpenAI.model_rebuild()


logger = logging.getLogger(__name__)


def load_generative_model(model_config) -> BaseChatOpenAI:
    logger.debug({
        "message": "loading the model",
        "model_config": model_config,
        "temperature": model_config["temperature"]
    })

    llm = ChatOpenAI(
        model=model_config["name"],
        temperature=model_config["temperature"],
        api_key=os.getenv("OPENAI_API_KEY", model_config.get("api_key", None)),
        base_url=model_config["base_url"],
        max_retries=model_config.get('max_retries', 2),
    )

    return llm