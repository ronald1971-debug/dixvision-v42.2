"""HARDEN-05 — harness governance approval shim (INV-68).

Production code paths construct an :class:`ExecutionIntent` inside an
intelligence subsystem and approve it through the live governance
pipeline; only then does the intent reach
:meth:`ExecutionEngine.execute`. The harness (``ui.server``) and a
handful of replay tests need a deterministic, single-call equivalent
that does the same thing without spinning up the full async control
loop.

This module exposes exactly one helper,
:func:`approve_signal_for_execution`, that:

1. Re-stamps the input :class:`SignalEvent` so it carries the canonical
   ``produced_by_engine`` for its origin (HARDEN-03 wire contract).
2. Constructs a frozen :class:`ExecutionIntent` via the B25-allowed
   :func:`create_execution_intent` factory.
3. Marks the intent approved through :func:`mark_approved`, recording
   a deterministic ``governance_decision_id`` derived from the signal
   timestamp so the result is reproducible across replays (INV-15).

The helper does **not** call ``execute`` — the caller does, so the
chokepoint stays exactly where HARDEN-02 placed it
(:meth:`ExecutionEngine.execute`). All this module does is collapse
"build intent + governance approves" into one deterministic call.

Lint scope:
* B25 allows ``governance_engine.*`` to call the
  :func:`create_execution_intent` factory, so this module is the
  legitimate home for the approval shim.
* B1 — does not import any other engine package.
"""

from __future__ import annotations

from typing import Final

from core.contracts.events import SignalEvent
from core.contracts.execution_intent import (
    AUTHORISED_INTENT_ORIGINS,
    ExecutionIntent,
    create_execution_intent,
    mark_approved,
)

__all__ = [
    "DEFAULT_HARNESS_ORIGIN",
    "HARNESS_DECISION_ID_PREFIX",
    "approve_signal_for_execution",
]


DEFAULT_HARNESS_ORIGIN: Final[str] = (
    "intelligence_engine.signal_pipeline.orchestrator"
)
"""Default origin stamped on harness-approved intents.

Must be a member of :data:`AUTHORISED_INTENT_ORIGINS`. The default
matches the production signal-pipeline subsystem so the matrix's
intelligence actor row covers it without a per-harness exception."""


HARNESS_DECISION_ID_PREFIX: Final[str] = "harness:auto"
"""Prefix for synthetic governance_decision_ids emitted by the shim.

The auditor can grep this prefix to identify intents that bypassed
the live governance loop. Production governance never emits this
prefix."""


def approve_signal_for_execution(
    signal: SignalEvent,
    *,
    ts_ns: int,
    origin: str = DEFAULT_HARNESS_ORIGIN,
    decision_id: str | None = None,
) -> ExecutionIntent:
    """Build + approve an :class:`ExecutionIntent` in one deterministic call.

    Args:
        signal: The :class:`SignalEvent` to wrap. The signal's
            ``ts_ns`` and identity are preserved verbatim — the helper
            does not mutate the signal.
        ts_ns: Monotonic timestamp stamped onto the intent. The
            harness sources this from its replay clock so the result
            is reproducible.
        origin: Authorised intent origin. Defaults to
            :data:`DEFAULT_HARNESS_ORIGIN`. Must be a member of
            :data:`AUTHORISED_INTENT_ORIGINS`; the underlying factory
            raises :class:`UnauthorizedOriginError` otherwise.
        decision_id: Override the synthetic
            ``governance_decision_id``. When unset (the common case),
            a deterministic id is derived from ``ts_ns`` so two
            replays of the same input produce byte-identical intents
            (INV-15).

    Returns:
        A frozen :class:`ExecutionIntent` with
        ``approved_by_governance=True``, ready for
        :meth:`ExecutionEngine.execute`.
    """

    if origin not in AUTHORISED_INTENT_ORIGINS:
        # Re-raised here as well so the harness gets a stable error
        # path even if a caller passes a bare string. The factory
        # would raise too, but doing the check up-front keeps the
        # ``mark_approved`` call below from running with a bad value.
        from core.contracts.execution_intent import UnauthorizedOriginError

        raise UnauthorizedOriginError(
            f"unauthorised harness origin: {origin!r}"
        )
    intent = create_execution_intent(
        ts_ns=ts_ns,
        origin=origin,
        signal=signal,
    )
    return mark_approved(
        intent,
        governance_decision_id=(
            decision_id
            if decision_id is not None
            else f"{HARNESS_DECISION_ID_PREFIX}:{ts_ns}"
        ),
    )
