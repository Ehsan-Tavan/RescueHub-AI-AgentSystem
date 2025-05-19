import argparse
import yaml

from RescueHub_AI_AgentSystem.utils import setup_logger
from RescueHub_AI_AgentSystem.agents import create_fire_emergency_agent

if __name__ == "__main__":
    LOGGER = setup_logger()
    PARSER = argparse.ArgumentParser(
        description="RescueHub_AI_AgentSystem"
    )
    PARSER.add_argument("-c", "--config", default=None, type=str,
                        help="Config file path (default: None)")

    ARGS = PARSER.parse_args()

    if ARGS.config is None:
        raise ValueError("The config argument should be set!")

    CONFIG = yaml.safe_load(open(ARGS.config))
    LOGGER.debug(CONFIG)
    FIRE_AGENT = create_fire_emergency_agent(model_config=CONFIG["generative_model"])
    RESPONSE = FIRE_AGENT.invoke({
    "query": "سلام! چطور میتونم کمک کنم؟"
    })
    print(RESPONSE)
