"""
core/contracts/execution.py
DIX VISION v42.2 — execution Protocol Contract
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IExecution(Protocol):
    """Protocol: execution contract."""

    def process_tick(self,market_data:dict) -> Any:
        """Process market tick. TODO: implement in concrete class."""
        ...

    def start(self) -> Any:
        """Start engine. TODO: implement in concrete class."""
        ...

    def stop(self) -> Any:
        """Stop engine. TODO: implement in concrete class."""
        ...
