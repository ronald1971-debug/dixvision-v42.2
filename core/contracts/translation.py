"""
core/contracts/translation.py
DIX VISION v42.2 — translation Protocol Contract
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ITranslation(Protocol):
    """Protocol: translation contract."""

    def translate(self,payload:dict) -> Any:
        """Translate intent to schema. TODO: implement in concrete class."""
        ...

    def validate(self,schema:Any) -> Any:
        """Validate schema. TODO: implement in concrete class."""
        ...
