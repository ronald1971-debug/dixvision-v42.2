"""
core/contracts/intelligence.py
DIX VISION v42.2 — intelligence Protocol Contract
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IIntelligence(Protocol):
    """Protocol: intelligence contract."""

    def evaluate(self,data:dict) -> Any:
        """Produce trading decision. TODO: implement in concrete class."""
        ...

    def learn(self,sample:Any) -> Any:
        """Ingest learning sample. TODO: implement in concrete class."""
        ...
