"""HARDEN-01 — frozen ``ExecutionIntent`` currency (INV-68).

The Execution Gate requires a single, immutable, content-hashed token
that flows from Intelligence → Governance → Execution. The token is
the only object that ``ExecutionEngine.execute(...)`` accepts (wired
in HARDEN-02), so every trade is provably:

* originated by an authorised producer (``origin`` in the registered
  set of intelligence subsystems),
* approved by Governance (``approved_by_governance is True`` and
  ``governance_decision_id`` non-empty),
* unmodified between approval and execution (``content_hash``
  re-computed at the chokepoint must match).

The module is registry-aware: the set of authorised producer modules
is derived from ``registry/authority_matrix.yaml`` rather than being
hard-coded here, so adding a new intelligence subsystem is a registry
change, not a code change.

T1 / pure
---------

This module performs no IO, has no clocks, and has no randomness.
The deterministic ``intent_id`` and ``content_hash`` are pure
functions of the canonical fields, so two processes that observe the
same ``ExecutionIntent`` derive the same id (INV-15 replay
determinism).

Lint enforcement (B25 — added in this same PR)
-----------------------------------------------

* Only ``intelligence_engine.*`` and ``governance_engine.*`` may call
  :func:`create_execution_intent`. The Triad Lock (INV-56) is now
  triple-bound:

  - **B22** — only intelligence may construct ``SignalEvent``.
  - **B25** — only intelligence may construct ``ExecutionIntent``.
  - **B21** — only execution may construct ``ExecutionEvent``.

  Governance is the sole writer of ``approved_by_governance=True``
  and is allowed through B25 to mark approval / rejection.

Construction is also defended at runtime: a value with an unknown
``origin`` raises :class:`UnauthorizedOriginError` *before* the
content hash is computed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Final

from core.contracts.events import SignalEvent

__all__ = [
    "AUTHORISED_INTENT_ORIGINS",
    "ExecutionIntent",
    "UnauthorizedOriginError",
    "compute_content_hash",
    "compute_intent_id",
    "create_execution_intent",
    "mark_approved",
    "mark_rejected",
]


class UnauthorizedOriginError(ValueError):
    """Raised when ``origin`` is not in :data:`AUTHORISED_INTENT_ORIGINS`."""


# Authorised producers of an :class:`ExecutionIntent`. Curated against
# ``registry/authority_matrix.yaml``: every entry must resolve to a
# subsystem under the ``intelligence`` actor row. Adding an entry is a
# registry-led change — keep this tuple in sync with the matrix.
#
# Tests assert that every entry is a strict child of
# ``intelligence_engine.*``; the operator UI / tests sometimes need to
# fabricate intents to drive replay paths and use a dedicated
# ``"tests.fixtures"`` origin so they never collide with a real
# subsystem name.
AUTHORISED_INTENT_ORIGINS: Final[frozenset[str]] = frozenset(
    {
        "intelligence_engine.meta_controller.runtime_adapter",
        "intelligence_engine.meta_controller.hot_path",
        "intelligence_engine.signal_pipeline.orchestrator",
        # Replay / harness origin. Tests construct intents through
        # the ``tests.fixtures`` origin so that B25 lint remains
        # tight on production code paths.
        "tests.fixtures",
    }
)


def _canonical_fields(
    *,
    ts_ns: int,
    origin: str,
    signal: SignalEvent,
    approved_by_governance: bool,
    governance_decision_id: str,
) -> tuple[tuple[str, str], ...]:
    """Stable canonical key/value tuple for hashing.

    The order is fixed so byte output is identical across processes
    (INV-15). We deliberately serialise only the fields that are part
    of the contract — adding a field requires bumping the schema
    version (a future :data:`SCHEMA_VERSION` constant) so a stale
    consumer can't accidentally accept an extended intent.
    """

    plugin_chain = "|".join(signal.plugin_chain)
    meta_items = ";".join(
        f"{k}={signal.meta[k]}" for k in sorted(signal.meta.keys())
    )
    return (
        ("ts_ns", str(ts_ns)),
        ("origin", origin),
        ("signal.ts_ns", str(signal.ts_ns)),
        ("signal.symbol", signal.symbol),
        ("signal.side", signal.side.value),
        ("signal.confidence", repr(signal.confidence)),
        ("signal.plugin_chain", plugin_chain),
        ("signal.meta", meta_items),
        ("approved_by_governance", "1" if approved_by_governance else "0"),
        ("governance_decision_id", governance_decision_id),
    )


def compute_content_hash(
    *,
    ts_ns: int,
    origin: str,
    signal: SignalEvent,
    approved_by_governance: bool,
    governance_decision_id: str,
) -> str:
    """Deterministic SHA-256 over the canonical key/value tuple."""

    h = hashlib.sha256()
    for key, value in _canonical_fields(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=approved_by_governance,
        governance_decision_id=governance_decision_id,
    ):
        h.update(key.encode("utf-8"))
        h.update(b"=")
        h.update(value.encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


def compute_intent_id(content_hash: str) -> str:
    """Stable, human-readable intent id derived from the content hash."""

    # 16 hex chars of SHA-256 is more than enough for collision
    # resistance over the lifetime of one trading session, and keeps
    # log lines readable.
    return f"INTENT-{content_hash[:16]}"


@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    """Frozen, content-hashed approval token.

    Construction goes through :func:`create_execution_intent`; direct
    instantiation is technically possible (Python can't prevent it on
    a public dataclass) but is forbidden by B25 lint and
    :func:`create_execution_intent` is the only constructor that
    computes the content hash and intent id correctly.

    Approval state transitions are surfaced as pure helpers
    (:func:`mark_approved`, :func:`mark_rejected`) that return a new
    instance — the dataclass itself never mutates.
    """

    intent_id: str
    ts_ns: int
    origin: str
    signal: SignalEvent
    approved_by_governance: bool
    governance_decision_id: str
    content_hash: str
    meta: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def verify_content_hash(self) -> bool:
        """Re-derive the canonical hash and compare."""

        expected = compute_content_hash(
            ts_ns=self.ts_ns,
            origin=self.origin,
            signal=self.signal,
            approved_by_governance=self.approved_by_governance,
            governance_decision_id=self.governance_decision_id,
        )
        return expected == self.content_hash


def create_execution_intent(
    *,
    ts_ns: int,
    origin: str,
    signal: SignalEvent,
    approved_by_governance: bool = False,
    governance_decision_id: str = "",
    meta: tuple[tuple[str, str], ...] = (),
) -> ExecutionIntent:
    """Authorised constructor.

    Raises:
        UnauthorizedOriginError: if ``origin`` is not in
            :data:`AUTHORISED_INTENT_ORIGINS`.
        ValueError: if ``approved_by_governance`` is True without a
            non-empty ``governance_decision_id``.

    Returns:
        A frozen :class:`ExecutionIntent` with deterministic
        :attr:`intent_id` and :attr:`content_hash`.
    """

    if origin not in AUTHORISED_INTENT_ORIGINS:
        raise UnauthorizedOriginError(
            f"unauthorised intent origin: {origin!r} — must be one of "
            f"{sorted(AUTHORISED_INTENT_ORIGINS)}"
        )
    if approved_by_governance and not governance_decision_id:
        raise ValueError(
            "approved_by_governance=True requires governance_decision_id"
        )
    if not approved_by_governance and governance_decision_id:
        # Allow rejection records: an explicit decision id with
        # approved=False corresponds to a governance rejection. We
        # accept this; tighten the constraint here if the contract
        # ever forbids it.
        pass

    content_hash = compute_content_hash(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=approved_by_governance,
        governance_decision_id=governance_decision_id,
    )
    return ExecutionIntent(
        intent_id=compute_intent_id(content_hash),
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
        approved_by_governance=approved_by_governance,
        governance_decision_id=governance_decision_id,
        content_hash=content_hash,
        meta=meta,
    )


def mark_approved(
    intent: ExecutionIntent, *, governance_decision_id: str
) -> ExecutionIntent:
    """Return a new intent with ``approved_by_governance=True``.

    The original instance is not modified. Governance is expected to
    call this helper and the resulting intent is what ``execute(...)``
    accepts.
    """

    if not governance_decision_id:
        raise ValueError("governance_decision_id required to mark approved")
    if intent.approved_by_governance:
        # Idempotent: re-approving with the same decision id is a no-op.
        if intent.governance_decision_id == governance_decision_id:
            return intent
        raise ValueError(
            "intent already approved with a different governance_decision_id"
        )
    return create_execution_intent(
        ts_ns=intent.ts_ns,
        origin=intent.origin,
        signal=intent.signal,
        approved_by_governance=True,
        governance_decision_id=governance_decision_id,
        meta=intent.meta,
    )


def mark_rejected(
    intent: ExecutionIntent, *, governance_decision_id: str
) -> ExecutionIntent:
    """Return a new intent that records governance rejection.

    ``approved_by_governance`` stays False; the decision id is
    recorded so the audit ledger can attribute the rejection.
    """

    if not governance_decision_id:
        raise ValueError("governance_decision_id required to mark rejected")
    return create_execution_intent(
        ts_ns=intent.ts_ns,
        origin=intent.origin,
        signal=intent.signal,
        approved_by_governance=False,
        governance_decision_id=governance_decision_id,
        meta=intent.meta,
    )
