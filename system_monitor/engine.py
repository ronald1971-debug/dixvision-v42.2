"""
system_monitor/engine.py
DIX VISION v42.2 — Dyon monitoring engine (manifest-canonical §6).

Thin canonical wrapper around ``execution.engine.DyonEngine``.
"""
from __future__ import annotations


def get_system_monitor():  # noqa: D401 - façade
    """Return the singleton Dyon engine."""
    from execution.engine import get_dyon_engine

    return get_dyon_engine()
