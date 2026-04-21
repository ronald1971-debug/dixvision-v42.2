"""
core/contracts/State.py
DIX VISION v42.2 — State Protocol Contract
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IState(Protocol):
    """Protocol: State contract."""

    def get_state(self) -> Any:
        """Return SystemState snapshot. TODO: implement in concrete class."""
        ...

    def set_mode(self,mode:str) -> Any:
        """Set system mode. TODO: implement in concrete class."""
        ...
