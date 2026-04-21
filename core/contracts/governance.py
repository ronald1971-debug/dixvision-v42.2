"""
core/contracts/governance.py
DIX VISION v42.2 — governance Protocol Contract
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IGovernance(Protocol):
    """Protocol: governance contract."""

    def evaluate(self,request:Any) -> Any:
        """Evaluate action request. TODO: implement in concrete class."""
        ...

    def evaluate_boot(self,state:Any) -> Any:
        """Evaluate boot readiness. TODO: implement in concrete class."""
        ...
