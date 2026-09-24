"""Logging setup: everything goes to stderr so stdout stays clean for harnesses."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: str | None = None) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("traffictrak")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    logger.setLevel((level or os.environ.get("TRAFFICTRAK_LOG", "WARNING")).upper())
    return logger
