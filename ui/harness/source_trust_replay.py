"""Paper-S6 source-trust promotion ledger replay.

Extracted from ``ui.server`` as part of the P1.2 harness god-object
refactor. Walks every authority row in chronological order and
applies ``OPERATOR_SOURCE_TRUST_PROMOTED`` / ``..._DEMOTED`` payloads
to the in-memory overlay store so promotions recorded before a
restart survive the bounce. Replay is fail-soft per-row: a malformed
payload (missing fields, bad enum value, non-int ts_ns) is skipped
rather than aborting the whole replay so one corrupted historical
row cannot take down boot.

INV-15: replay is purely a function of ``(ledger contents, store)``;
no clock or PRNG is consulted. The caller passes the ledger writer
(which holds the authoritative in-memory mirror) so the same path is
used in tests with an in-memory ledger and in production with a
SQLite-backed one.
"""

from __future__ import annotations

from core.contracts.signal_trust import SignalTrust
from core.contracts.source_trust_promotions import (
    DEMOTION_LEDGER_KIND,
    PROMOTION_LEDGER_KIND,
    SourceTrustPromotionStore,
    is_promotable_target,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)


def replay_source_trust_promotions(
    *,
    ledger_writer: LedgerAuthorityWriter,
    store: SourceTrustPromotionStore,
) -> None:
    """Rebuild *store* from the authority ledger (Paper-S6)."""

    for entry in ledger_writer.read():
        if entry.kind == PROMOTION_LEDGER_KIND:
            payload = entry.payload
            try:
                source_id = str(payload["source_id"])
                target_trust = SignalTrust(str(payload["target_trust"]))
                requestor = str(payload.get("requestor", "operator"))
                reason = str(payload.get("reason", ""))
                ts_ns = int(payload.get("ts_ns", entry.ts_ns))
            except (KeyError, ValueError, TypeError):
                continue
            if not source_id or not is_promotable_target(target_trust):
                continue
            try:
                store.promote(
                    source_id=source_id,
                    target_trust=target_trust,
                    requestor=requestor,
                    reason=reason,
                    ts_ns=ts_ns,
                )
            except ValueError:
                continue
        elif entry.kind == DEMOTION_LEDGER_KIND:
            payload = entry.payload
            source_id = str(payload.get("source_id", ""))
            if not source_id:
                continue
            store.demote(source_id)


__all__ = ("replay_source_trust_promotions",)
