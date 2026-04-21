"""
security/audit_trail.py
Thin wrapper so security callers don't touch the ledger directly.
"""
from __future__ import annotations

from typing import Any


def audit(sub_type: str, source: str, payload: dict[str, Any]) -> None:
    try:
        from state.ledger.event_store import append_event

        append_event("SECURITY", sub_type, source, payload)
    except Exception:
        pass
