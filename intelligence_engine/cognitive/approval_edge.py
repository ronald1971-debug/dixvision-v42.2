"""Wave-03 PR-5 — the *only* path that promotes a cognitive proposal to the bus.

The chat graph (PR-3) and chat runtime (PR-4) never emit
``SignalEvent`` directly. When the assistant reply contains a
``propose`` block, the runtime queues an
:class:`ApprovalRequestApi` via the
:class:`~intelligence_engine.cognitive.approval_queue.ApprovalQueue`.
The queue stays inert — no event hits the bus — until the operator
clicks **Approve** in the dashboard, which calls into
:func:`approve` here.

This module is the single place where a cognitive-origin
``SignalEvent`` is constructed. Two invariants are enforced at this
edge:

* **HARDEN-03 stamp** — every emitted event carries
  ``produced_by_engine="intelligence_engine.cognitive"``. The B26
  lint (`tools.authority_lint`) statically forbids this prefix from
  appearing on any other ``SignalEvent`` construction site, so this
  module is the only legitimate origin.
* **HARDEN-02 chain** — the emitted event flows out through the
  ``signal_emitter`` callable seam, which the HTTP layer in
  ``ui.server`` binds to ``intelligence.process(sig) ->
  execution.execute(...)``. The execute side already enforces
  ``AuthorityGuard.assert_can_execute`` on every intent.

Isolation contract (B1): this module lives under
``intelligence_engine.cognitive.*`` and may not import
``governance_engine.*`` or ``system_engine.*``. The ledger write and
bus emission are handed in as callable seams.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from core.contracts.api.cognitive_chat_approvals import (
    ApprovalDecisionRequest,
    ApprovalRequestApi,
    ApprovalSideApi,
    ApprovalStatusApi,
)
from core.contracts.events import Side, SignalEvent
from intelligence_engine.cognitive.approval_queue import (
    ApprovalAlreadyDecidedError,
    ApprovalNotFoundError,
    ApprovalQueue,
)

__all__ = [
    "ApprovalAlreadyDecidedError",
    "ApprovalEdge",
    "ApprovalNotFoundError",
    "COGNITIVE_PRODUCED_BY_ENGINE",
    "LedgerAppend",
    "SignalEmitter",
]


COGNITIVE_PRODUCED_BY_ENGINE = "intelligence_engine.cognitive"
"""Provenance string stamped onto every cognitive-origin ``SignalEvent``.

The B26 authority lint rule pins this string: any other module that
constructs a ``SignalEvent`` with ``produced_by_engine`` starting
with ``"intelligence_engine.cognitive"`` is a violation."""


SignalEmitter = Callable[[SignalEvent], None]
"""Bus emission seam — the HTTP layer wires this to the live bus."""


LedgerAppend = Callable[[str, Mapping[str, str]], None]
"""Audit ledger seam — same shape as the cognitive saver's adapter."""


def _side_from_api(side: ApprovalSideApi) -> Side:
    """Translate the wire enum to the canonical :class:`Side`.

    ``HOLD`` is unreachable here — the queue refuses HOLD proposals
    on submit and ``approve`` re-checks the row's status — but the
    branch exists so a future relaxation of the parser can't silently
    push an invalid side onto the bus."""

    if side is ApprovalSideApi.BUY:
        return Side.BUY
    if side is ApprovalSideApi.SELL:
        return Side.SELL
    raise ValueError(
        f"approval edge cannot emit a SignalEvent for side={side!r}"
    )


@dataclass
class ApprovalEdge:
    """The chokepoint between the approval queue and the signal bus.

    Construction takes three seams (queue, emitter, ledger) so the
    edge stays B1-clean and trivially testable:

    * ``queue`` — owns the ``PENDING/APPROVED/REJECTED`` lifecycle.
    * ``signal_emitter`` — production binding pushes onto the bus
      (``IntelligenceEngine.process``); tests pass a recording stub.
    * ``ledger_append`` — production binding writes to the
      ``LedgerAuthorityWriter``; tests pass a recording stub.

    The ``ts_ns`` callable is injected so the emitted ``SignalEvent``
    timestamp is deterministic in replay (INV-15)."""

    queue: ApprovalQueue
    signal_emitter: SignalEmitter
    ledger_append: LedgerAppend
    ts_ns: Callable[[], int]

    def approve(
        self,
        *,
        request_id: str,
        decision: ApprovalDecisionRequest,
    ) -> tuple[ApprovalRequestApi, SignalEvent]:
        """Flip the row to APPROVED and emit the resulting ``SignalEvent``.

        The returned tuple gives the route handler everything it
        needs for the response: the decided request (for the API
        contract) and the event that landed on the bus (for the
        ``emitted_signal_id`` cross-reference).

        On lifecycle errors (unknown id, already decided) raises the
        exact exception the queue raised — the route handler maps
        ``ApprovalNotFoundError`` → 404 and
        ``ApprovalAlreadyDecidedError`` → 409.
        """

        # Snapshot the proposal *before* flipping the row so the
        # emit-side side effects (bus, ledger) see the same data the
        # operator approved. This keeps the audit row consistent
        # with what the dashboard showed when the click happened.
        pending = self.queue.get(request_id)
        if pending.status is not ApprovalStatusApi.PENDING:
            raise ApprovalAlreadyDecidedError(
                f"approval {request_id!r} is already "
                f"{pending.status.value}"
            )

        decided = self.queue.decide(
            request_id=request_id,
            approved=True,
            decided_by=decision.decided_by,
        )
        ts = self.ts_ns()
        meta: dict[str, str] = {
            "rationale": pending.proposal.rationale,
            "approval_id": request_id,
            "thread_id": pending.thread_id,
            "decided_by": decision.decided_by,
        }
        if decision.note:
            meta["approval_note"] = decision.note
        sig = SignalEvent(
            ts_ns=ts,
            symbol=pending.proposal.symbol,
            side=_side_from_api(pending.proposal.side),
            confidence=pending.proposal.confidence,
            plugin_chain=("cognitive_chat",),
            meta=meta,
            produced_by_engine=COGNITIVE_PRODUCED_BY_ENGINE,
        )
        # Ledger first, bus second — if the bus wiring raises we
        # still have a "operator approved this" record. The reverse
        # ordering would let an ack land without provenance if the
        # ledger writer failed.
        # PR-7: the ledger row's ``ts_ns`` is the queue's decision
        # timestamp (``decided.decided_at_ts_ns``), not the edge's
        # emit ``ts``. The two clocks can diverge in production —
        # the queue stamps via ``time.time_ns`` while the edge uses
        # the harness monotonic counter — and the projection in
        # :mod:`approval_projection` reads this field straight back
        # into ``decided_at_ts_ns`` on rehydrate. Using the edge's
        # counter here would corrupt the field across restarts.
        decided_ts = (
            decided.decided_at_ts_ns
            if decided.decided_at_ts_ns is not None
            else ts
        )
        self.ledger_append(
            "OPERATOR_APPROVED_SIGNAL",
            {
                "approval_id": request_id,
                "thread_id": pending.thread_id,
                "symbol": pending.proposal.symbol,
                "side": pending.proposal.side.value,
                "confidence": f"{pending.proposal.confidence:.6f}",
                "decided_by": decision.decided_by,
                "ts_ns": str(decided_ts),
            },
        )
        self.signal_emitter(sig)
        return decided, sig

    def reject(
        self,
        *,
        request_id: str,
        decision: ApprovalDecisionRequest,
    ) -> ApprovalRequestApi:
        """Flip the row to REJECTED — no event hits the bus.

        The rejection is recorded on the audit ledger (so the chain
        captures every operator decision, not just approvals) and
        the queue row is returned to the route handler.
        """

        pending = self.queue.get(request_id)
        if pending.status is not ApprovalStatusApi.PENDING:
            raise ApprovalAlreadyDecidedError(
                f"approval {request_id!r} is already "
                f"{pending.status.value}"
            )
        decided = self.queue.decide(
            request_id=request_id,
            approved=False,
            decided_by=decision.decided_by,
        )
        # PR-7: the ledger row carries the queue's decision
        # timestamp so :mod:`approval_projection` rehydrates a
        # byte-identical ``decided_at_ts_ns``. ``self.ts_ns()`` is
        # only retained to drive ``self.signal_emitter`` callers
        # that expect a per-call edge timestamp; the rejection path
        # has no emitter so ``ts`` is unused here.
        decided_ts = (
            decided.decided_at_ts_ns
            if decided.decided_at_ts_ns is not None
            else self.ts_ns()
        )
        payload: dict[str, str] = {
            "approval_id": request_id,
            "thread_id": pending.thread_id,
            "symbol": pending.proposal.symbol,
            "side": pending.proposal.side.value,
            "confidence": f"{pending.proposal.confidence:.6f}",
            "decided_by": decision.decided_by,
            "ts_ns": str(decided_ts),
        }
        if decision.note:
            payload["approval_note"] = decision.note
        self.ledger_append("OPERATOR_REJECTED_SIGNAL", payload)
        return decided
