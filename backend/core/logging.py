"""Logging configuration. Never log passwords, tokens or full user text."""
from __future__ import annotations

import logging
import sys

LOGGER_NAME = "humanizer"


def setup_logging(level: str = "INFO") -> logging.Logger:
    root = logging.getLogger()
    if not any(getattr(h, "_humanizer", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        handler._humanizer = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level.upper())
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(LOGGER_NAME)


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
