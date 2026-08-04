"""Logging setup that plays nicely with uvicorn's own handlers."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Install a single stderr handler for the ``reed`` logger tree."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger("reed").setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger = logging.getLogger("reed")
    logger.setLevel(level.upper())
    logger.handlers = [handler]
    logger.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``reed`` namespace."""
    suffix = name.removeprefix("reed.")
    return logging.getLogger(f"reed.{suffix}" if suffix else "reed")
