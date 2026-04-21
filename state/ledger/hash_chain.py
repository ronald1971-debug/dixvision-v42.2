"""
state/ledger/hash_chain.py
DIX VISION v42.2 — Cryptographic Hash Chain Verifier
"""
from __future__ import annotations

from state.ledger.event_store import get_event_store


def verify_full_chain() -> tuple[bool, str]:
    try:
        ok = get_event_store().verify_chain()
        return ok, "chain_valid" if ok else "chain_tampered"
    except Exception as e:
        return False, f"verification_error:{e}"
