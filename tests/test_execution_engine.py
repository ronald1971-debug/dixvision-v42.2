"""Phase E1 — ExecutionEngine end-to-end + paper broker tests."""

from __future__ import annotations

import time

import pytest

from core.contracts.events import (
    EventKind,
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)
from core.contracts.governance import SystemMode
from core.contracts.market import MarketTick
from execution_engine.adapters import BrokerAdapter, PaperBroker
from execution_engine.engine import (
    ExecutionEngine,
    LegacyExecutionPathRemovedError,
)
from governance_engine.harness_approver import (
    approve_signal_for_execution,
)
from intelligence_engine.engine import IntelligenceEngine


def _intent_for(sig: SignalEvent, *, ts_ns: int | None = None):
    """Build a governance-approved intent for a test signal.

    HARDEN-05 — the legacy ``ExecutionEngine.process`` path is gone;
    every test that wants a fill goes through the same
    ``execute(intent)`` chokepoint as production.
    """
    return approve_signal_for_execution(
        sig, ts_ns=ts_ns if ts_ns is not None else sig.ts_ns
    )

# ---------------------------------------------------------------------------
# PaperBroker unit tests
# ---------------------------------------------------------------------------


def test_paper_broker_implements_protocol():
    assert isinstance(PaperBroker(), BrokerAdapter)


def test_paper_broker_buy_fills_at_mark_when_no_slippage():
    broker = PaperBroker()
    sig = SignalEvent(ts_ns=10, symbol="EURUSD", side=Side.BUY, confidence=0.7)
    out = broker.submit(sig, mark_price=1.10)
    assert isinstance(out, ExecutionEvent)
    assert out.status is ExecutionStatus.FILLED
    assert out.side is Side.BUY
    assert out.price == pytest.approx(1.10)
    assert out.qty == pytest.approx(1.0)
    assert out.venue == "paper"
    assert out.order_id.startswith("PAPER-")


def test_paper_broker_sell_applies_slippage():
    broker = PaperBroker(slippage_bps=10.0)  # 0.10 %
    sig = SignalEvent(ts_ns=11, symbol="BTCUSDT", side=Side.SELL, confidence=0.8)
    out = broker.submit(sig, mark_price=50_000.0)
    assert out.status is ExecutionStatus.FILLED
    assert out.side is Side.SELL
    assert out.price == pytest.approx(50_000.0 - 50.0)


def test_paper_broker_hold_rejects():
    broker = PaperBroker()
    sig = SignalEvent(ts_ns=12, symbol="X", side=Side.HOLD, confidence=0.0)
    out = broker.submit(sig, mark_price=100.0)
    assert out.status is ExecutionStatus.REJECTED
    assert out.qty == 0.0


def test_paper_broker_non_positive_mark_fails():
    broker = PaperBroker()
    sig = SignalEvent(ts_ns=13, symbol="X", side=Side.BUY, confidence=0.5)
    out = broker.submit(sig, mark_price=0.0)
    assert out.status is ExecutionStatus.FAILED
    assert out.order_id == ""


def test_paper_broker_meta_qty_overrides_default():
    broker = PaperBroker(default_qty=1.0)
    sig = SignalEvent(
        ts_ns=14,
        symbol="X",
        side=Side.BUY,
        confidence=0.5,
        meta={"qty": "2.5"},
    )
    out = broker.submit(sig, mark_price=10.0)
    assert out.qty == pytest.approx(2.5)


def test_paper_broker_order_ids_are_monotonic():
    broker = PaperBroker()
    sig = SignalEvent(ts_ns=15, symbol="X", side=Side.BUY, confidence=0.5)
    a = broker.submit(sig, mark_price=10.0)
    b = broker.submit(sig, mark_price=10.0)
    assert a.order_id < b.order_id


def test_paper_broker_rejects_negative_slippage():
    with pytest.raises(ValueError):
        PaperBroker(slippage_bps=-1.0)


def test_paper_broker_rejects_non_positive_default_qty():
    with pytest.raises(ValueError):
        PaperBroker(default_qty=0.0)


# ---------------------------------------------------------------------------
# ExecutionEngine integration tests
# ---------------------------------------------------------------------------


def test_execution_engine_no_mark_returns_failed_event():
    engine = ExecutionEngine()
    sig = SignalEvent(ts_ns=20, symbol="EURUSD", side=Side.BUY, confidence=0.7)
    out = engine.execute(_intent_for(sig))
    assert len(out) == 1
    evt = out[0]
    assert isinstance(evt, ExecutionEvent)
    assert evt.status is ExecutionStatus.FAILED
    assert evt.kind is EventKind.EXECUTION


def test_execution_engine_with_mark_fills_buy():
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=21, symbol="EURUSD", bid=1.0998, ask=1.1002, last=1.1)
    )
    sig = SignalEvent(ts_ns=22, symbol="EURUSD", side=Side.BUY, confidence=0.7)
    out = engine.execute(_intent_for(sig))
    assert len(out) == 1
    evt = out[0]
    assert isinstance(evt, ExecutionEvent)
    assert evt.status is ExecutionStatus.FILLED
    assert evt.price == pytest.approx(1.1)


def test_execution_engine_legacy_process_hard_fails():
    """HARDEN-05 — the deprecated ``process`` path raises immediately.

    Any caller still on the old contract must surface as a runtime
    error, not silently degrade via a deprecation warning.
    """
    engine = ExecutionEngine()
    sig = SignalEvent(ts_ns=23, symbol="X", side=Side.BUY, confidence=0.5)
    with pytest.raises(LegacyExecutionPathRemovedError):
        engine.process(sig)


def test_execution_engine_reset_mark_with_zero_last_is_ignored():
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=24, symbol="X", bid=1.0, ask=1.01, last=1.0)
    )
    engine.on_market(
        MarketTick(ts_ns=25, symbol="X", bid=1.0, ask=1.01, last=0.0)
    )
    sig = SignalEvent(ts_ns=26, symbol="X", side=Side.BUY, confidence=0.5)
    out = engine.execute(_intent_for(sig))
    assert out[0].status is ExecutionStatus.FILLED


def test_execution_engine_default_adapter_is_paper():
    engine = ExecutionEngine()
    assert engine.adapter.name == "paper"


# ---------------------------------------------------------------------------
# End-to-end: Intelligence -> Execution
# ---------------------------------------------------------------------------


def test_e2e_intelligence_to_execution_round_trip():
    intelligence = IntelligenceEngine()
    execution = ExecutionEngine()

    execution.on_market(
        MarketTick(ts_ns=100, symbol="BTCUSDT", bid=49_990.0, ask=50_010.0, last=50_000.0)
    )

    sig = SignalEvent(
        ts_ns=101,
        symbol="BTCUSDT",
        side=Side.BUY,
        confidence=0.8,
        plugin_chain=("phase_e1_test",),
    )

    intelligence_out = intelligence.process(sig)
    assert len(intelligence_out) == 1

    execution_events: list = []
    for evt in intelligence_out:
        execution_events.extend(execution.execute(_intent_for(evt)))

    assert len(execution_events) == 1
    exec_evt = execution_events[0]
    assert isinstance(exec_evt, ExecutionEvent)
    assert exec_evt.status is ExecutionStatus.FILLED
    assert exec_evt.side is Side.BUY
    assert exec_evt.price == pytest.approx(50_000.0)
    assert exec_evt.symbol == "BTCUSDT"


def test_e2e_replay_is_deterministic():
    """INV-15 / TEST-01 — same input sequence -> bit-identical output."""

    def run() -> list:
        intelligence = IntelligenceEngine()
        execution = ExecutionEngine(adapter=PaperBroker(slippage_bps=2.5))
        execution.on_market(
            MarketTick(
                ts_ns=200, symbol="EURUSD", bid=1.0998, ask=1.1002, last=1.1
            )
        )
        signals = [
            SignalEvent(ts_ns=200 + i, symbol="EURUSD", side=Side.BUY, confidence=0.6)
            for i in range(5)
        ]
        out: list = []
        for s in signals:
            for ev in intelligence.process(s):
                out.extend(execution.execute(_intent_for(ev)))
        return out

    a = run()
    b = run()
    assert a == b
    assert len(a) == 5


# ---------------------------------------------------------------------------
# PERF-01..02 latency SLO (TEST-06).
# Build plan target: p50 < 1 ms / p99 < 5 ms in CI.
# We assert a generous CI envelope (p50 < 2 ms / p99 < 10 ms) so flaky CI
# doesn't fail on cold caches; the typical machine is well below target.
# ---------------------------------------------------------------------------


def test_execution_engine_latency_slo():
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=300, symbol="X", bid=99.5, ask=100.5, last=100.0)
    )
    sig = SignalEvent(ts_ns=300, symbol="X", side=Side.BUY, confidence=0.7)
    intent = _intent_for(sig)

    # warm-up
    for _ in range(200):
        engine.execute(intent)

    n = 5_000
    samples = [0] * n
    for i in range(n):
        t0 = time.perf_counter_ns()
        engine.execute(intent)
        samples[i] = time.perf_counter_ns() - t0

    samples.sort()
    p50_ms = samples[n // 2] / 1_000_000.0
    p99_ms = samples[int(n * 0.99)] / 1_000_000.0

    assert p50_ms < 2.0, f"p50 too high: {p50_ms:.3f} ms"
    assert p99_ms < 10.0, f"p99 too high: {p99_ms:.3f} ms"


# ---------------------------------------------------------------------------
# Wave-04.6 PR-B — SHADOW = signals-on-execution-off
#
# The mode-effect table in core/contracts/mode_effects.py is the single
# source of truth. Three modes report executions_dispatch=False today
# (SAFE, SHADOW, LOCKED). The Execution Gate must honour the table by
# passing the AuthorityGuard but suppressing the broker side effect and
# returning a synthetic REJECTED ExecutionEvent with a machine-readable
# reason. PAPER, CANARY, LIVE, AUTO must continue to dispatch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [SystemMode.SAFE, SystemMode.SHADOW, SystemMode.LOCKED],
)
def test_execute_suppresses_dispatch_when_mode_effect_blocks(mode):
    """SAFE/SHADOW/LOCKED skip broker dispatch via mode-effect table."""
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=400, symbol="X", bid=99.5, ask=100.5, last=100.0)
    )
    sig = SignalEvent(ts_ns=400, symbol="X", side=Side.BUY, confidence=0.7)
    intent = _intent_for(sig)

    out = engine.execute(intent, current_mode=mode)

    assert len(out) == 1
    ev = out[0]
    assert ev.status is ExecutionStatus.REJECTED
    assert ev.qty == 0.0
    assert ev.price == 0.0
    assert ev.order_id == ""
    assert ev.symbol == "X"
    assert ev.side is Side.BUY
    assert ev.produced_by_engine == "execution_engine"
    assert ev.meta == {"reason": "mode_effect_suppressed", "mode": mode.name}


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.PAPER,
        SystemMode.CANARY,
        SystemMode.LIVE,
        SystemMode.AUTO,
    ],
)
def test_execute_dispatches_when_mode_effect_allows(mode):
    """PAPER/CANARY/LIVE/AUTO route to the broker as before."""
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=401, symbol="X", bid=99.5, ask=100.5, last=100.0)
    )
    sig = SignalEvent(ts_ns=401, symbol="X", side=Side.BUY, confidence=0.7)
    intent = _intent_for(sig)

    out = engine.execute(intent, current_mode=mode)

    assert len(out) == 1
    ev = out[0]
    assert ev.status is ExecutionStatus.FILLED
    assert ev.qty > 0.0
    assert ev.price > 0.0
    assert ev.order_id.startswith("PAPER-")
    assert "reason" not in ev.meta


def test_execute_without_mode_argument_preserves_legacy_behaviour():
    """Callers that omit current_mode dispatch unconditionally.

    Replay tests and harness flows already gate upstream; the optional
    parameter must not regress them.
    """
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=402, symbol="X", bid=99.5, ask=100.5, last=100.0)
    )
    sig = SignalEvent(ts_ns=402, symbol="X", side=Side.BUY, confidence=0.7)
    intent = _intent_for(sig)

    out = engine.execute(intent)

    assert len(out) == 1
    assert out[0].status is ExecutionStatus.FILLED


def test_execute_runs_authority_guard_before_mode_check():
    """Guard failures must not be masked by SHADOW dispatch suppression.

    A signal whose intent fails the AuthorityGuard must still raise
    even if the mode would have suppressed the broker side effect.
    The guard is the first invariant in the gate (HARDEN-02); the mode
    suppression is the second.
    """
    from execution_engine.execution_gate import UnauthorizedActorError

    engine = ExecutionEngine()
    sig = SignalEvent(ts_ns=403, symbol="X", side=Side.BUY, confidence=0.7)
    intent = _intent_for(sig)

    with pytest.raises(UnauthorizedActorError):
        engine.execute(intent, caller="not_execution_engine", current_mode=SystemMode.SHADOW)


# ---------------------------------------------------------------------------
# Wave-04.6 PR-C — equity-based notional cap pure helper.
#
# The canonical interpretation of ``ModeEffect.size_cap_pct`` is *percent
# of account equity* (not percent of broker fill qty). PR-C ships only the
# pure helper :func:`equity_notional_cap_qty`. The runtime application of
# that helper is deliberately *not* in ExecutionEngine — it lives in
# :class:`PolicyEngine` (Wave-04.6 PR-E) as a pre-execution clamp on
# :class:`ExecutionIntent.notional_pct`. Applying the cap post-broker (as
# this PR previously did) would create a position-tracking discrepancy
# with live venues: the venue records the full fill while the system
# records the clamped 1%. ExecutionEngine therefore continues to dispatch
# the broker's chosen qty unchanged in CANARY for now; the canonical
# guarantee is enforced upstream by Governance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.PAPER,
        SystemMode.CANARY,
        SystemMode.LIVE,
        SystemMode.AUTO,
    ],
)
def test_execute_canary_does_not_clamp_post_broker(mode):
    """ExecutionEngine no longer applies a post-broker size cap.

    The CANARY safety guarantee is enforced *pre-execution* by the
    PolicyEngine (Wave-04.6 PR-E). ExecutionEngine in PR-C dispatches
    the broker's chosen qty for every dispatch-allowed mode, with no
    mode-driven mutation of ``ExecutionEvent.qty``.
    """
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=500, symbol="X", bid=99.5, ask=100.5, last=100.0)
    )
    sig = SignalEvent(ts_ns=500, symbol="X", side=Side.BUY, confidence=0.7)
    intent = _intent_for(sig)

    out = engine.execute(intent, current_mode=mode)

    assert len(out) == 1
    ev = out[0]
    assert ev.status is ExecutionStatus.FILLED
    assert ev.qty == pytest.approx(1.0)
    assert "clamped_to_mode_cap" not in ev.meta
    assert "original_qty" not in ev.meta


def test_equity_notional_cap_qty_canary_one_percent():
    """CANARY caps notional at 1% of equity → max_qty = equity * 0.01 / price."""
    from core.contracts.mode_effects import equity_notional_cap_qty

    cap = equity_notional_cap_qty(
        mode=SystemMode.CANARY, equity=100_000.0, price=100.0
    )
    # 100_000 * 0.01 / 100 = 10.0
    assert cap == pytest.approx(10.0)


@pytest.mark.parametrize(
    "mode",
    [SystemMode.LIVE, SystemMode.AUTO],
)
def test_equity_notional_cap_qty_uncapped_modes_return_none(mode):
    """LIVE / AUTO have ``size_cap_pct=None`` → helper returns ``None``."""
    from core.contracts.mode_effects import equity_notional_cap_qty

    assert (
        equity_notional_cap_qty(mode=mode, equity=100_000.0, price=100.0)
        is None
    )


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.PAPER,
        SystemMode.SHADOW,
        SystemMode.SAFE,
        SystemMode.LOCKED,
    ],
)
def test_equity_notional_cap_qty_non_applicable_modes_return_none(mode):
    """Modes with ``size_cap_pct=0.0`` are non-applicable → ``None``.

    PAPER is not an equity-bearing venue; SHADOW/SAFE/LOCKED suppress
    dispatch entirely so the cap is moot. The helper signals this with
    ``None`` so callers do not erroneously clamp to zero.
    """
    from core.contracts.mode_effects import equity_notional_cap_qty

    assert (
        equity_notional_cap_qty(mode=mode, equity=100_000.0, price=100.0)
        is None
    )


def test_equity_notional_cap_qty_rejects_negative_equity():
    from core.contracts.mode_effects import equity_notional_cap_qty

    with pytest.raises(ValueError):
        equity_notional_cap_qty(
            mode=SystemMode.CANARY, equity=-1.0, price=100.0
        )


def test_equity_notional_cap_qty_rejects_non_positive_price():
    from core.contracts.mode_effects import equity_notional_cap_qty

    with pytest.raises(ValueError):
        equity_notional_cap_qty(
            mode=SystemMode.CANARY, equity=100_000.0, price=0.0
        )
    with pytest.raises(ValueError):
        equity_notional_cap_qty(
            mode=SystemMode.CANARY, equity=100_000.0, price=-1.0
        )
