from __future__ import annotations

import logging
import sys
from typing import Optional

from app.core.config import get_settings


def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure root logger once.
    """
    settings = get_settings()
    log_level = (level or settings.log_level or "INFO").upper()

    root = logging.getLogger()
    root.setLevel(log_level)

    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
