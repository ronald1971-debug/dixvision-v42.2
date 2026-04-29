"""Silent-fallback audit emitter for SCVS Phase 3 (rule **SCVS-10**).

Whenever an engine swaps to a fallback data source — for any reason
(stale upstream, schema mismatch, AI provider degraded, etc.) — it
MUST emit a ``SOURCE_FALLBACK_ACTIVATED`` :class:`SystemEvent` so the
governance ledger has an explicit record of the substitution. The
"silent fallback" pattern (swap without record) is the precise thing
SCVS-10 forbids.

This module is the single canonical constructor for that event so the
emission shape stays identical across engines.

INV-15 — pure / deterministic. Caller supplies ``now_ns``; nothing
inside this module reads a clock or PRNG.
"""

from __future__ import annotations

from collections.abc import Mapping

from core.contracts.events import SystemEvent, SystemEventKind

SOURCE = "system_engine.scvs.fallback_audit"


def make_fallback_event(
    *,
    now_ns: int,
    failed_source_id: str,
    fallback_source_id: str,
    reason: str,
    detail: Mapping[str, str] | None = None,
) -> SystemEvent:
    """Construct a SCVS-10 ``SOURCE_FALLBACK_ACTIVATED`` system event.

    The event payload is intentionally narrow — just enough for the
    governance ledger to attribute the substitution.
    """

    if not failed_source_id:
        raise ValueError("failed_source_id is required")
    if not fallback_source_id:
        raise ValueError("fallback_source_id is required")
    if failed_source_id == fallback_source_id:
        raise ValueError(
            "failed_source_id and fallback_source_id must differ "
            "(a source cannot fall back to itself)"
        )
    if not reason:
        raise ValueError("reason is required (SCVS-10 forbids silent swaps)")

    payload: dict[str, str] = {
        "failed_source_id": failed_source_id,
        "fallback_source_id": fallback_source_id,
        "reason": reason,
    }
    if detail:
        for k, v in detail.items():
            if k in payload:
                raise ValueError(f"detail key {k!r} collides with reserved field")
            payload[k] = str(v)

    return SystemEvent(
        ts_ns=now_ns,
        sub_kind=SystemEventKind.SOURCE_FALLBACK_ACTIVATED,
        source=SOURCE,
        payload=payload,
    )


__all__ = ["make_fallback_event"]
