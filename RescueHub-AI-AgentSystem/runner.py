import argparse
import yaml

from utils import setup_logger

if __name__ == "__main__":
    LOGGER = setup_logger()
    PARSER = argparse.ArgumentParser(
        description='Ocr Wrapper'
    )
    PARSER.add_argument("-c", "--config", default=None, type=str,
                        help="Config file path (default: None)")

    ARGS = PARSER.parse_args()

    if ARGS.config is None:
        raise ValueError("The config argument should be set!")

    CONFIG = yaml.safe_load(open(ARGS.config))
    LOGGER.info(CONFIG)
