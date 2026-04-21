"""
core/contracts/time.py
DIX VISION v42.2 — time Protocol Contract
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ITime(Protocol):
    """Protocol: time contract."""

    def now(self) -> Any:
        """Return TimeStamp. TODO: implement in concrete class."""
        ...

    def now_with_seq(self) -> Any:
        """Return (utc, seq) tuple. TODO: implement in concrete class."""
        ...
