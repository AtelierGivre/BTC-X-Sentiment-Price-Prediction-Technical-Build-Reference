"""Shared logger: console + logs/pipeline.log."""
from __future__ import annotations

import logging
import sys

from config import LOGS

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        root.addHandler(stream)
        file = logging.FileHandler(LOGS / "pipeline.log", encoding="utf-8")
        file.setFormatter(fmt)
        root.addHandler(file)
        _CONFIGURED = True
    return logging.getLogger(name)
