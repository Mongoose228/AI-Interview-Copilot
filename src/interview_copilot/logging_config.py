import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import config
from .paths import get_logs_dir


def setup_logging():
    logger = logging.getLogger("interview_copilot")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    logging.getLogger("httpx").setLevel(logging.WARNING)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        log_path = get_logs_dir() / "copilot.log"
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        # Fall back to stdout-only if file logging cannot be set up
        stream_handler.emit(
            logging.LogRecord(
                name="interview_copilot",
                level=logging.WARNING,
                pathname=__file__,
                lineno=0,
                msg=f"File logging unavailable: {e}",
                args=(),
                exc_info=None,
            )
        )

    return logger


logger = setup_logging()


def log_sensitive(msg: str, *args, **kwargs):
    """Log sensitive data only when obfuscation is disabled.

    NOTE: This controls LOCAL log output only. Transcripts and candidate profile
    are always sent to the OpenRouter API for LLM suggestions regardless of this
    setting. See gui/privacy_banner.py for the user-facing disclosure.
    """
    if not config.LOG_OBFUSCATION_ENABLED:
        logger.info(msg, *args, **kwargs)
