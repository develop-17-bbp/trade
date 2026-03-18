"""
Centralized logging configuration.
Writes logs to DATA_ROOT/logs/ instead of the repo directory.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from .paths import LOG_DIR, ensure_dirs


def configure_logging(level: str = "INFO", log_to_file: bool = True,
                      app_name: str = "trade") -> None:
    """
    Configure root logger for the entire application.
    - Console: colored output to stdout
    - File: rotating file handler at LOG_DIR/trade.log (10MB, 5 backups)
    """
    ensure_dirs()

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = []

    # -- Console handler --
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handlers.append(console)

    # -- Rotating file handler --
    if log_to_file:
        log_file = LOG_DIR / f"{app_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(file_handler)

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"[LOGGING] Configured: level={level}, log_dir={LOG_DIR}"
    )
