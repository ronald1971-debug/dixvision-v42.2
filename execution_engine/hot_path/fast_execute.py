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

Polyglot dual backend
---------------------

When the ``dixvision_py_execution`` PyO3 wheel is importable, the
gate decision is delegated to the Rust crate
``dixvision-execution::fast_execute``. The wheel is OPTIONAL: if it
is not installed (Python-only test runs, operator boxes that haven't
built it, CI matrix entries that intentionally exercise the Python
fallback) this module runs the pure-Python implementation below.

Both backends are byte-equivalent — the Rust port owns no logic that
the Python reference doesn't, and ``tests/test_fast_execute_parity.py``
exercises every branch under both. Side effects (the order-id
counter, ``ExecutionEvent`` / ``HotPathDecision`` construction, the
``signal.meta["qty"]`` fallback ladder) live in Python in either
mode; the FFI seam moves only the pure decision function.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)
from core.contracts.risk import RiskSnapshot

try:  # pragma: no cover — exercised by the parity test under both modes.
    from dixvision_py_execution import decide_gate_py as _RUST_DECIDE_GATE
except ImportError:  # pragma: no cover
    _RUST_DECIDE_GATE = None


def _rust_backend_available() -> bool:
    """Public predicate: is the Rust gate backend importable?

    Used by parity tests (and the wave-04 LIVE-flip checklist) to
    verify the wheel is present in environments where it should be.
    """

    return _RUST_DECIDE_GATE is not None


class HotPathOutcome(StrEnum):
    APPROVED = "APPROVED"
    REJECTED_RISK_STALE = "REJECTED_RISK_STALE"
    REJECTED_NO_MARK = "REJECTED_NO_MARK"
    REJECTED_LIMIT = "REJECTED_LIMIT"
    REJECTED_HOLD = "REJECTED_HOLD"
    REJECTED_LOW_CONFIDENCE = "REJECTED_LOW_CONFIDENCE"


# Stable mapping of audit reasons to HotPathOutcome. Public so the
# Rust seam can be re-tagged without re-encoding the contract here.
_REASON_TO_OUTCOME: Final[dict[str, HotPathOutcome]] = {
    "halted": HotPathOutcome.REJECTED_LIMIT,
    "risk_stale": HotPathOutcome.REJECTED_RISK_STALE,
    "no_mark": HotPathOutcome.REJECTED_NO_MARK,
    "confidence_floor": HotPathOutcome.REJECTED_LOW_CONFIDENCE,
    "hold_signal": HotPathOutcome.REJECTED_HOLD,
    "qty_above_cap": HotPathOutcome.REJECTED_LIMIT,
}


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
        prefer_rust: When ``True`` (default) and the
            ``dixvision_py_execution`` wheel is importable, the gate
            decision is computed in Rust. Set to ``False`` to force
            the pure-Python path (e.g. for parity comparison or
            debugging).
    """

    name: str = "fast_executor"
    spec_id: str = "EXEC-11"

    def __init__(
        self,
        *,
        max_staleness_ns: int = 2_000_000_000,
        default_qty: float = 1.0,
        prefer_rust: bool = True,
    ) -> None:
        if max_staleness_ns <= 0:
            raise ValueError("max_staleness_ns must be > 0")
        if default_qty <= 0.0:
            raise ValueError("default_qty must be > 0")
        self._max_staleness_ns = max_staleness_ns
        self._default_qty = default_qty
        self._counter: int = 0
        self._use_rust = prefer_rust and _rust_backend_available()

    @property
    def using_rust_backend(self) -> bool:
        """True iff this executor will dispatch the gate to Rust."""

        return self._use_rust

    def execute(
        self,
        *,
        signal: SignalEvent,
        snapshot: RiskSnapshot,
        mark_price: float,
    ) -> HotPathDecision:
        if self._use_rust:
            return self._execute_rust(
                signal=signal,
                snapshot=snapshot,
                mark_price=mark_price,
            )
        return self._execute_python(
            signal=signal,
            snapshot=snapshot,
            mark_price=mark_price,
        )

    # ------------------------------------------------------------------
    # Rust backend
    # ------------------------------------------------------------------

    def _execute_rust(
        self,
        *,
        signal: SignalEvent,
        snapshot: RiskSnapshot,
        mark_price: float,
    ) -> HotPathDecision:
        # Resolve qty + cap on the Python side so the FFI seam stays
        # primitive-only (no Python dict crosses the boundary).
        qty = self._qty_for(signal)
        cap = snapshot.cap_for(signal.symbol)

        assert _RUST_DECIDE_GATE is not None  # _use_rust gate above
        outcome_str, reason, price = _RUST_DECIDE_GATE(
            signal_ts_ns=signal.ts_ns,
            signal_confidence=signal.confidence,
            signal_side=signal.side.value,
            snapshot_version=snapshot.version,
            snapshot_ts_ns=snapshot.ts_ns,
            snapshot_halted=snapshot.halted,
            snapshot_max_signal_confidence=snapshot.max_signal_confidence,
            cap=cap,
            mark_price=mark_price,
            max_staleness_ns=self._max_staleness_ns,
            qty=qty,
        )

        if outcome_str == HotPathOutcome.APPROVED.value:
            return self._build_approved(signal, snapshot, price=price, qty=qty)

        # Re-tag onto the canonical Python enum via the audit reason.
        outcome = _REASON_TO_OUTCOME[reason]
        return self._reject(
            signal,
            snapshot,
            outcome=outcome,
            reason=reason,
            price=price,
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
