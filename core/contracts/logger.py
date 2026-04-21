"""
core/contracts/logger.py
DIX VISION v42.2 — logger Protocol Contract
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ILogger(Protocol):
    """Protocol: logger contract."""

    def info(self,msg:str,**kw) -> Any:
        """Log INFO. TODO: implement in concrete class."""
        ...

    def error(self,msg:str,**kw) -> Any:
        """Log ERROR. TODO: implement in concrete class."""
        ...

    def critical(self,msg:str,**kw) -> Any:
        """Log CRITICAL. TODO: implement in concrete class."""
        ...
