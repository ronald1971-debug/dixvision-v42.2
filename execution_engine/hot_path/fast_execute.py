"""Fast-execute hot path — EXEC-11 / T1-pure.

Takes one :class:`SignalEvent`, a frozen :class:`RiskSnapshot`, and a
mark price; returns a :class:`HotPathDecision` describing whether the
signal becomes an :class:`ExecutionEvent` and at what price/qty.

T1 rules (authority_lint):

* No imports from ``governance_engine``, ``intelligence_engine``,
  ``system_engine``, or any offline engine.
* No clocks, no ``random``, no IO.
* All inputs explicit; all outputs deterministic functions of inputs.

The hot path is the per-tick gate. Slow-path bookkeeping (lifecycle
FSM, ledger audit) is the responsibility of :class:`ExecutionEngine`
and Governance.

Single-backend (Python)
-----------------------

Reviewer #3 (audit v3, item 1) flagged the dual-backend Python+Rust
state as a determinism hazard: INV-15 / TEST-01 are only as strong
as the equivalence between the two backends, and the polyglot
revival had no shadow-equivalence harness. The conservative
resolution per the user's directive was to delete the Rust crates
and revisit after a 30-day shadow-mode window
(see ``docs/rust_revival_schedule.yaml`` and
``tools/rust_revival_reminder.py`` for the revival reminder).

This module is therefore Python-only. ``FastExecutor`` no longer
takes a ``prefer_rust`` parameter; the gate decision always runs
through :meth:`FastExecutor._execute_python` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)
from core.contracts.risk import RiskSnapshot


class HotPathOutcome(StrEnum):
    APPROVED = "APPROVED"
    REJECTED_RISK_STALE = "REJECTED_RISK_STALE"
    REJECTED_NO_MARK = "REJECTED_NO_MARK"
    REJECTED_LIMIT = "REJECTED_LIMIT"
    REJECTED_HOLD = "REJECTED_HOLD"
    REJECTED_LOW_CONFIDENCE = "REJECTED_LOW_CONFIDENCE"


@dataclass(frozen=True, slots=True)
class HotPathDecision:
    """Result of one :meth:`FastExecutor.execute` call.

    Always carries the canonical :class:`ExecutionEvent` so callers
    write a single row to the bus regardless of approval.
    """

    outcome: HotPathOutcome
    event: ExecutionEvent
    risk_version: int


class FastExecutor:
    """T1-pure executor.

    Args:
        max_staleness_ns: Reject if ``signal.ts_ns - snapshot.ts_ns``
            exceeds this delta. Default ``2_000_000_000`` (2 s).
        default_qty: Order qty when ``signal.meta['qty']`` is absent.
    """

    name: str = "fast_executor"
    spec_id: str = "EXEC-11"

    def __init__(
        self,
        *,
        max_staleness_ns: int = 2_000_000_000,
        default_qty: float = 1.0,
    ) -> None:
        if max_staleness_ns <= 0:
            raise ValueError("max_staleness_ns must be > 0")
        if default_qty <= 0.0:
            raise ValueError("default_qty must be > 0")
        self._max_staleness_ns = max_staleness_ns
        self._default_qty = default_qty
        self._counter: int = 0

    def execute(
        self,
        *,
        signal: SignalEvent,
        snapshot: RiskSnapshot,
        mark_price: float,
    ) -> HotPathDecision:
        return self._execute_python(
            signal=signal,
            snapshot=snapshot,
            mark_price=mark_price,
        )

    # ------------------------------------------------------------------
    # Pure-Python backend (canonical reference)
    # ------------------------------------------------------------------

    def _execute_python(
        self,
        *,
        signal: SignalEvent,
        snapshot: RiskSnapshot,
        mark_price: float,
    ) -> HotPathDecision:
        # Halted: fail fast, deterministically.
        if snapshot.halted:
            return self._reject(
                signal,
                snapshot,
                outcome=HotPathOutcome.REJECTED_LIMIT,
                reason="halted",
                price=mark_price if mark_price > 0.0 else 0.0,
            )

        # Stale risk → SAFE-style halt.
        if signal.ts_ns - snapshot.ts_ns > self._max_staleness_ns:
            return self._reject(
                signal,
                snapshot,
                outcome=HotPathOutcome.REJECTED_RISK_STALE,
                reason="risk_stale",
                price=0.0,
            )

        if mark_price <= 0.0:
            return self._reject(
                signal,
                snapshot,
                outcome=HotPathOutcome.REJECTED_NO_MARK,
                reason="no_mark",
                price=0.0,
            )

        if signal.confidence < snapshot.max_signal_confidence:
            return self._reject(
                signal,
                snapshot,
                outcome=HotPathOutcome.REJECTED_LOW_CONFIDENCE,
                reason="confidence_floor",
                price=mark_price,
            )

        if signal.side is Side.HOLD:
            return self._reject(
                signal,
                snapshot,
                outcome=HotPathOutcome.REJECTED_HOLD,
                reason="hold_signal",
                price=mark_price,
            )

        qty = self._qty_for(signal)
        cap = snapshot.cap_for(signal.symbol)
        if cap is not None and qty > cap:
            return self._reject(
                signal,
                snapshot,
                outcome=HotPathOutcome.REJECTED_LIMIT,
                reason="qty_above_cap",
                price=mark_price,
            )

        return self._build_approved(signal, snapshot, price=mark_price, qty=qty)

    # -- internals ---------------------------------------------------------

    def _build_approved(
        self,
        signal: SignalEvent,
        snapshot: RiskSnapshot,
        *,
        price: float,
        qty: float,
    ) -> HotPathDecision:
        self._counter += 1
        order_id = f"HP-{self._counter:08d}"
        event = ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            price=price,
            status=ExecutionStatus.APPROVED,
            venue="hot_path",
            order_id=order_id,
            meta={"risk_version": str(snapshot.version)},
            produced_by_engine="execution_engine",
        )
        return HotPathDecision(
            outcome=HotPathOutcome.APPROVED,
            event=event,
            risk_version=snapshot.version,
        )

    def _qty_for(self, signal: SignalEvent) -> float:
        meta_qty = signal.meta.get("qty")
        if meta_qty is None:
            return self._default_qty
        try:
            value = float(meta_qty)
        except (TypeError, ValueError):
            return self._default_qty
        if value <= 0.0:
            return self._default_qty
        return value

    def _reject(
        self,
        signal: SignalEvent,
        snapshot: RiskSnapshot,
        *,
        outcome: HotPathOutcome,
        reason: str,
        price: float,
    ) -> HotPathDecision:
        event = ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=0.0,
            price=price,
            status=ExecutionStatus.REJECTED,
            venue="hot_path",
            order_id="",
            meta={"reason": reason, "risk_version": str(snapshot.version)},
            produced_by_engine="execution_engine",
        )
        return HotPathDecision(
            outcome=outcome,
            event=event,
            risk_version=snapshot.version,
        )


__all__ = [
    "FastExecutor",
    "HotPathDecision",
    "HotPathOutcome",
    "RiskSnapshot",
]
