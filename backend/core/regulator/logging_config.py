"""Shim — forwards to canonical backend.logging_config."""

from backend.logging_config import get_logger  # noqa: F401

__all__ = ["get_logger"]
