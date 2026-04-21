"""
cockpit — DIX VISION v42.2 Operator Dashboard.

Re-exports the FastAPI app so that ``uvicorn cockpit:app`` continues to work.
"""
from __future__ import annotations

try:
    from .app import app  # noqa: F401
except Exception:  # pragma: no cover — optional FastAPI dep
    app = None  # type: ignore[assignment]
