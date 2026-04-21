"""
system_monitor/checks/data_integrity_check.py
Verifies ledger hash-chain integrity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrityResult:
    ok: bool
    detail: str


def check_data_integrity() -> IntegrityResult:
    from state.ledger.hash_chain import verify_full_chain

    ok, msg = verify_full_chain()
    return IntegrityResult(ok, msg)
