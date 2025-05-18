import logging
import os
from datetime import datetime
from pythonjsonlogger import jsonlogger


def setup_logger(log_dir: str = "../logs", log_level: int = logging.INFO, logger_name: str = "rescuehub"):
    """
    Create a logger that outputs to both the console and a timestamped file using JSON formatting.

    Args:
        log_dir: Directory to save log files.
        log_level: Logging level (e.g., logging.INFO).
        logger_name: Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{logger_name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent log duplication if root logger is used

    # Clear previous handlers if any (important when re-running in notebooks or scripts)
    if logger.hasHandlers():
        logger.handlers.clear()

    # JSON formatter
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream (console) handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
