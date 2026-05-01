"""Hugging Face runtime setup.

Configures optional authentication and suppresses noisy unauthenticated
warnings that can appear during model downloads.
"""
from __future__ import annotations

import logging
import os
import warnings

from loguru import logger

from app.config import get_settings


class _SuppressHFUnauthWarning(logging.Filter):
    """Filter only the known unauthenticated HF Hub warning message."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - tiny helper
        message = record.getMessage().lower()
        if "unauthenticated requests to the hf hub" in message:
            return False
        if "please set a hf_token" in message:
            return False
        return True


def configure_hf_runtime() -> None:
    """Apply Hugging Face auth and warning preferences from settings/.env."""
    settings = get_settings()
    token = settings.hf_token.strip()

    if token and token != "your_hf_token_here":
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        logger.info("HF_TOKEN loaded ✓")

    # Disable Xet-backed transfer by default in this project to avoid noisy
    # unauthenticated warnings in local/dev setups.
    if settings.hf_hub_disable_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    warning_filter = _SuppressHFUnauthWarning()
    for logger_name in ("huggingface_hub", "hf_xet"):
        logging.getLogger(logger_name).addFilter(warning_filter)

    warnings.filterwarnings(
        "ignore",
        message=r".*unauthenticated requests to the HF Hub.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Please set a HF_TOKEN.*",
    )
