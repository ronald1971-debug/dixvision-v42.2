"""
core/bootstrap/loader.py
Deterministic dotted-path module loader used by the bootstrap graph.
"""
from __future__ import annotations

import importlib
from types import ModuleType


def load_module(dotted_path: str) -> ModuleType:
    """Import and return the module at ``dotted_path``.

    Separate wrapper so the bootstrap graph has a single testable surface.
    """
    return importlib.import_module(dotted_path)
