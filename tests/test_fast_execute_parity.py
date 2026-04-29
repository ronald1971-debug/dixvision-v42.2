"""Polyglot parity — Python vs Rust backend for the hot-path gate.

Both backends MUST agree on every observable: outcome, audit reason,
event price, event qty, and the canonical ``ExecutionEvent`` /
``HotPathDecision`` shape.

The Rust path runs only when the ``dixvision_py_execution`` PyO3
wheel is importable. When the wheel is absent (Python-only test run,
operator box that hasn't built it) the Rust mixin's tests are
skipped instead of failing, so the same suite is the source of truth
for both deployment shapes.

Each scenario is run once for the canonical assertions (outcome,
status, qty, price, reasons, risk_version) under both backends. The
assertions are intentionally redundant with
``test_execution_hot_path`` so that a future Python regression that
doesn't touch the Rust crate still trips this file.
"""

from __future__ import annotations

import pytest

from core.contracts.events import ExecutionStatus, Side, SignalEvent
from execution_engine.hot_path.fast_execute import (
    FastExecutor,
    HotPathDecision,
    HotPathOutcome,
    RiskSnapshot,
    _rust_backend_available,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _signal(
    *,
    ts_ns: int = 1_000_000_000,
    symbol: str = "BTC-USD",
    side: Side = Side.BUY,
    confidence: float = 0.9,
    qty: str | None = None,
) -> SignalEvent:
    meta = {"qty": qty} if qty is not None else {}
    return SignalEvent(
        ts_ns=ts_ns,
        symbol=symbol,
        side=side,
        confidence=confidence,
        meta=meta,
    )


def _snapshot(**overrides) -> RiskSnapshot:
    base = {
        "version": 1,
        "ts_ns": 1_000_000_000,
        "max_position_qty": 5.0,
        "max_signal_confidence": 0.5,
    }
    base.update(overrides)
    return RiskSnapshot(**base)


# ---------------------------------------------------------------------------
# Backend mixin — runs the same scenarios under each backend
# ---------------------------------------------------------------------------


class _BackendCases:
    """Shared parametric test surface.

    Subclasses set ``prefer_rust`` and the test runner picks the
    backend up via :class:`FastExecutor`'s constructor flag. The
    class attribute is a strict bool because test discovery resolves
    it at class instantiation time.
    """

    prefer_rust: bool

    def _executor(
        self,
        *,
        max_staleness_ns: int = 2_000_000_000,
        default_qty: float = 1.0,
    ) -> FastExecutor:
        fx = FastExecutor(
            max_staleness_ns=max_staleness_ns,
            default_qty=default_qty,
            prefer_rust=self.prefer_rust,
        )
        # Sanity guard: the Rust subclass must actually be on the
        # Rust backend in this environment, otherwise the test is a
        # no-op masquerading as parity coverage.
        assert fx.using_rust_backend is self.prefer_rust
        return fx

    # ---- approved ---------------------------------------------------

    def test_approved_emits_filled_event(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(),
            snapshot=_snapshot(),
            mark_price=50_000.0,
        )
        assert isinstance(decision, HotPathDecision)
        assert decision.outcome is HotPathOutcome.APPROVED
        assert decision.event.status is ExecutionStatus.APPROVED
        assert decision.event.qty == 1.0
        assert decision.event.price == 50_000.0
        assert decision.event.venue == "hot_path"
        assert decision.event.meta["risk_version"] == "1"
        assert decision.risk_version == 1
        assert decision.event.order_id == "HP-00000001"

    def test_approved_uses_meta_qty_when_present_and_valid(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(qty="2.5"),
            snapshot=_snapshot(),
            mark_price=42_000.0,
        )
        assert decision.outcome is HotPathOutcome.APPROVED
        assert decision.event.qty == 2.5
        assert decision.event.price == 42_000.0

    def test_approved_falls_back_when_meta_qty_invalid(self) -> None:
        fx = self._executor(default_qty=3.0)
        for bad in ("not-a-number", "0", "-2.5", None):
            decision = fx.execute(
                signal=_signal(qty=bad),
                snapshot=_snapshot(),
                mark_price=99.0,
            )
            assert decision.outcome is HotPathOutcome.APPROVED, bad
            assert decision.event.qty == 3.0, bad

    def test_confidence_at_floor_passes(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(confidence=0.5),
            snapshot=_snapshot(max_signal_confidence=0.5),
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.APPROVED

    def test_unbounded_cap_allows_any_qty(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(qty="999999"),
            snapshot=_snapshot(max_position_qty=None),
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.APPROVED
        assert decision.event.qty == 999_999.0

    def test_per_symbol_cap_overrides_default(self) -> None:
        fx = self._executor()
        snap = _snapshot(
            max_position_qty=10.0, symbol_caps={"BTC-USD": 0.1}
        )
        decision = fx.execute(
            signal=_signal(qty="1.0"),
            snapshot=snap,
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_LIMIT
        assert decision.event.meta["reason"] == "qty_above_cap"

    # ---- rejection branches ----------------------------------------

    def test_rejected_when_halted(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(),
            snapshot=_snapshot(halted=True),
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_LIMIT
        assert decision.event.status is ExecutionStatus.REJECTED
        assert decision.event.meta["reason"] == "halted"
        assert decision.event.price == 100.0

    def test_halted_with_zero_mark_returns_zero_price(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(),
            snapshot=_snapshot(halted=True),
            mark_price=0.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_LIMIT
        assert decision.event.meta["reason"] == "halted"
        assert decision.event.price == 0.0

    def test_halted_wins_over_stale(self) -> None:
        # Both halted and stale apply; halted is checked first.
        fx = self._executor(max_staleness_ns=1_000_000)
        decision = fx.execute(
            signal=_signal(ts_ns=2_000_000_000),
            snapshot=_snapshot(ts_ns=1_000_000_000, halted=True),
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_LIMIT
        assert decision.event.meta["reason"] == "halted"

    def test_rejected_when_stale(self) -> None:
        fx = self._executor(max_staleness_ns=1_000_000)
        decision = fx.execute(
            signal=_signal(ts_ns=2_000_000_000),
            snapshot=_snapshot(ts_ns=1_000_000_000),
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_RISK_STALE
        assert decision.event.meta["reason"] == "risk_stale"
        assert decision.event.price == 0.0

    def test_stale_threshold_is_exclusive(self) -> None:
        # delta == max_staleness_ns is allowed.
        fx = self._executor(max_staleness_ns=1_000_000)
        decision = fx.execute(
            signal=_signal(ts_ns=1_001_000_000),
            snapshot=_snapshot(ts_ns=1_000_000_000),
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.APPROVED

    def test_rejected_when_no_mark(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(),
            snapshot=_snapshot(),
            mark_price=0.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_NO_MARK
        assert decision.event.meta["reason"] == "no_mark"
        assert decision.event.price == 0.0

    def test_rejected_when_negative_mark(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(),
            snapshot=_snapshot(),
            mark_price=-1.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_NO_MARK

    def test_rejected_when_low_confidence(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(confidence=0.49),
            snapshot=_snapshot(max_signal_confidence=0.5),
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_LOW_CONFIDENCE
        assert decision.event.meta["reason"] == "confidence_floor"
        assert decision.event.price == 100.0

    def test_rejected_when_hold(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(side=Side.HOLD),
            snapshot=_snapshot(),
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_HOLD
        assert decision.event.meta["reason"] == "hold_signal"

    def test_rejected_when_qty_above_cap(self) -> None:
        fx = self._executor()
        decision = fx.execute(
            signal=_signal(qty="10.0"),
            snapshot=_snapshot(max_position_qty=5.0),
            mark_price=100.0,
        )
        assert decision.outcome is HotPathOutcome.REJECTED_LIMIT
        assert decision.event.meta["reason"] == "qty_above_cap"

    def test_counter_increments_only_on_approve(self) -> None:
        fx = self._executor()
        # Reject first — counter should not bump.
        rejected = fx.execute(
            signal=_signal(side=Side.HOLD),
            snapshot=_snapshot(),
            mark_price=100.0,
        )
        assert rejected.event.order_id == ""
        # Approve next — counter starts at 1.
        approved = fx.execute(
            signal=_signal(),
            snapshot=_snapshot(),
            mark_price=100.0,
        )
        assert approved.event.order_id == "HP-00000001"


# ---------------------------------------------------------------------------
# Concrete classes — one per backend.
# ---------------------------------------------------------------------------


class TestFastExecuteParityPython(_BackendCases):
    """Pure-Python reference path. Always runs."""

    prefer_rust = False


@pytest.mark.skipif(
    not _rust_backend_available(),
    reason="dixvision_py_execution wheel not installed",
)
class TestFastExecuteParityRust(_BackendCases):
    """Rust PyO3 backend. Runs only when the wheel is importable."""

    prefer_rust = True


# ---------------------------------------------------------------------------
# Cross-backend scalar parity — same inputs, byte-equal outputs.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _rust_backend_available(),
    reason="dixvision_py_execution wheel not installed",
)
def test_cross_backend_outputs_are_byte_equal() -> None:
    """Sweep every gate branch and compare both backends row-by-row."""

    scenarios = [
        # (label, signal_kwargs, snapshot_kwargs, mark_price)
        ("approved_default_qty", {}, {}, 50_000.0),
        ("approved_meta_qty", {"qty": "2.5"}, {}, 100.0),
        (
            "approved_unbounded_cap",
            {"qty": "1000"},
            {"max_position_qty": None},
            10.0,
        ),
        (
            "rejected_halted",
            {},
            {"halted": True},
            100.0,
        ),
        (
            "rejected_halted_no_mark",
            {},
            {"halted": True},
            0.0,
        ),
        (
            "rejected_stale",
            {"ts_ns": 5_000_000_000},
            {"ts_ns": 1_000_000_000},
            100.0,
        ),
        ("rejected_no_mark", {}, {}, 0.0),
        (
            "rejected_low_confidence",
            {"confidence": 0.1},
            {"max_signal_confidence": 0.5},
            100.0,
        ),
        ("rejected_hold", {"side": Side.HOLD}, {}, 100.0),
        (
            "rejected_qty_above_cap",
            {"qty": "10.0"},
            {"max_position_qty": 5.0},
            100.0,
        ),
        (
            "approved_confidence_at_floor",
            {"confidence": 0.5},
            {"max_signal_confidence": 0.5},
            100.0,
        ),
    ]

    for label, sig_kw, snap_kw, mark in scenarios:
        py_fx = FastExecutor(prefer_rust=False)
        rs_fx = FastExecutor(prefer_rust=True)
        signal = _signal(**sig_kw)
        snapshot = _snapshot(**snap_kw)

        py = py_fx.execute(signal=signal, snapshot=snapshot, mark_price=mark)
        rs = rs_fx.execute(signal=signal, snapshot=snapshot, mark_price=mark)

        assert py.outcome == rs.outcome, label
        assert py.risk_version == rs.risk_version, label
        assert py.event.status == rs.event.status, label
        assert py.event.price == rs.event.price, label
        assert py.event.qty == rs.event.qty, label
        assert py.event.symbol == rs.event.symbol, label
        assert py.event.side == rs.event.side, label
        assert py.event.venue == rs.event.venue, label
        # order_id format includes a per-executor counter; both
        # backends start fresh so the first approved row is always
        # HP-00000001 for either.
        assert py.event.order_id == rs.event.order_id, label
        assert dict(py.event.meta) == dict(rs.event.meta), label


def test_using_rust_backend_flag_reflects_wheel_presence() -> None:
    fx = FastExecutor()
    assert fx.using_rust_backend is _rust_backend_available()


def test_prefer_rust_false_forces_python_even_when_wheel_present() -> None:
    fx = FastExecutor(prefer_rust=False)
    assert fx.using_rust_backend is False
