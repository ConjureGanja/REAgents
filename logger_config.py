"""
Structured logging configuration.
Call setup_logging() once at startup; all modules then use logging.getLogger(__name__).
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    root = logging.getLogger()
    # Guard: only configure once; re-calling just adjusts the console level.
    if root.handlers:
        root.handlers[0].setLevel(level)
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "re_agent.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)

    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "ultralytics", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("Logging initialised — file: %s", log_path)
