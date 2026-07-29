import logging
import sys

from .config import config


def setup_logging():
    logger = logging.getLogger("interview_copilot")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    # Disable other loggers if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    return logger


logger = setup_logging()


def log_sensitive(msg: str, *args, **kwargs):
    """Log sensitive data only if privacy mode is disabled and text logging is enabled."""
    if not config.PRIVACY_MODE and config.TEXT_LOGGING_ENABLED:
        logger.info(msg, *args, **kwargs)
