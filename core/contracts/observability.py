"""
core/contracts/observability.py
DIX VISION v42.2 — observability Protocol Contract
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IObservability(Protocol):
    """Protocol: observability contract."""

    def observe(self,name:str,value:float) -> Any:
        """Record metric. TODO: implement in concrete class."""
        ...

    def increment(self,counter:str,labels:dict) -> Any:
        """Increment counter. TODO: implement in concrete class."""
        ...
