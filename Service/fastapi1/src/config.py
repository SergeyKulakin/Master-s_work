from pathlib import Path

from pydantic import BaseModel, BaseSettings



class AppConfig(BaseSettings):
    """Service configuration."""

    host: str = '0.0.0.0'
    port: int = 8001


class LogConfig(BaseModel):
    """Logging configuration to be set for the server."""

    LOGGER_NAME: str = "LGBM"
    LOG_FORMAT: str = "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    LOG_LEVEL: str = "INFO"

    LOG_PATH: Path = Path(f"logs/{LOGGER_NAME}.log")

    version = 1
    disable_existing_loggers = False
    formatters = {
        "default": {
            "format": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers = {
        "streamHandler": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "fileHandler": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": LOG_PATH,
            "backupCount": 10,
            "maxBytes": 1024 * 1024 * 10
        }
    }
    loggers = {
        f"{LOGGER_NAME}": {"handlers": ["streamHandler", "fileHandler"], "level": LOG_LEVEL},
    }