"""
core/contracts/persistence.py
DIX VISION v42.2 — persistence Protocol Contract
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IPersistence(Protocol):
    """Protocol: persistence contract."""

    def save(self,event:str) -> Any:
        """Save snapshot. TODO: implement in concrete class."""
        ...

    def restore(self) -> Any:
        """Restore from snapshot. TODO: implement in concrete class."""
        ...
