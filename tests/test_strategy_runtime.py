"""Tests for Phase 3 strategy runtime (state machine, regime detector,
scheduler, orchestrator, conflict resolver).

Pure-Python, deterministic — no clocks, no IO.
"""

from __future__ import annotations

import math

import pytest

from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick
from intelligence_engine.strategy_runtime import (
    ConflictResolver,
    MarketRegime,
    RegimeDetector,
    StrategyLifecycleError,
    StrategyOrchestrator,
    StrategyScheduler,
    StrategyState,
    StrategyStateMachine,
)

# ---------------------------------------------------------------------------
# StrategyStateMachine
# ---------------------------------------------------------------------------


def test_strategy_fsm_legal_promotion_path():
    fsm = StrategyStateMachine()
    fsm.propose(strategy_id="alpha", ts_ns=1)
    fsm.transition(
        strategy_id="alpha",
        new_state=StrategyState.CANARY,
        ts_ns=2,
        reason="promote",
    )
    fsm.transition(
        strategy_id="alpha",
        new_state=StrategyState.LIVE,
        ts_ns=3,
        reason="promote",
    )
    rec = fsm.get("alpha")
    assert rec is not None
    assert rec.state is StrategyState.LIVE
    # propose + 2 transitions (SHADOW tier removed by SHADOW-DEMOLITION-02)
    assert len(rec.history) == 3
    assert [r.new for r in rec.history] == [
        StrategyState.PROPOSED,
        StrategyState.CANARY,
        StrategyState.LIVE,
    ]


def test_strategy_fsm_illegal_skip_raises():
    fsm = StrategyStateMachine()
    fsm.propose(strategy_id="alpha", ts_ns=1)
    with pytest.raises(StrategyLifecycleError):
        # Cannot skip CANARY.
        fsm.transition(
            strategy_id="alpha",
            new_state=StrategyState.LIVE,
            ts_ns=2,
            reason="skip",
        )


def test_strategy_fsm_terminal_states_reject_further_transitions():
    fsm = StrategyStateMachine()
    fsm.propose(strategy_id="a", ts_ns=1)
    fsm.transition(
        strategy_id="a",
        new_state=StrategyState.RETIRED,
        ts_ns=2,
        reason="early_withdrawal",
    )
    with pytest.raises(StrategyLifecycleError):
        fsm.transition(
            strategy_id="a",
            new_state=StrategyState.CANARY,
            ts_ns=3,
            reason="undo",
        )


def test_strategy_fsm_canary_can_rollback_to_proposed():
    fsm = StrategyStateMachine()
    fsm.propose(strategy_id="a", ts_ns=1)
    fsm.transition(
        strategy_id="a",
        new_state=StrategyState.CANARY,
        ts_ns=2,
        reason="promote",
    )
    fsm.transition(
        strategy_id="a",
        new_state=StrategyState.PROPOSED,
        ts_ns=3,
        reason="rollback",
    )
    assert fsm.get("a").state is StrategyState.PROPOSED


def test_strategy_fsm_unknown_id_raises():
    fsm = StrategyStateMachine()
    with pytest.raises(KeyError):
        fsm.transition(
            strategy_id="ghost",
            new_state=StrategyState.CANARY,
            ts_ns=1,
            reason="x",
        )


def test_strategy_fsm_replay_determinism():
    def run() -> tuple:
        fsm = StrategyStateMachine()
        fsm.propose(strategy_id="a", ts_ns=1)
        fsm.transition(
            strategy_id="a",
            new_state=StrategyState.CANARY,
            ts_ns=2,
            reason="promote",
        )
        fsm.transition(
            strategy_id="a",
            new_state=StrategyState.LIVE,
            ts_ns=3,
            reason="promote",
        )
        rec = fsm.get("a")
        assert rec is not None
        return tuple((h.ts_ns, h.prev, h.new, h.reason) for h in rec.history)

    assert run() == run()


def test_strategy_fsm_all_in():
    fsm = StrategyStateMachine()
    for sid in ("a", "b", "c"):
        fsm.propose(strategy_id=sid, ts_ns=1)
    fsm.transition(
        strategy_id="a",
        new_state=StrategyState.CANARY,
        ts_ns=2,
        reason="promote",
    )
    proposed = {r.strategy_id for r in fsm.all_in(StrategyState.PROPOSED)}
    canary = {r.strategy_id for r in fsm.all_in(StrategyState.CANARY)}
    assert proposed == {"b", "c"}
    assert canary == {"a"}


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------


def _tick(ts: int, mid: float, spread: float = 0.02) -> MarketTick:
    bid = mid - spread / 2
    ask = mid + spread / 2
    return MarketTick(
        ts_ns=ts, symbol="BTCUSDT", bid=bid, ask=ask, last=mid
    )


def test_regime_detector_unknown_when_too_few_samples():
    det = RegimeDetector(window=8)
    r = det.observe(_tick(1, 100.0))
    assert r.regime is MarketRegime.UNKNOWN
    assert r.sample_count == 1


def test_regime_detector_classifies_trending_up():
    det = RegimeDetector(
        window=16,
        trend_threshold_bps=5.0,
        volatility_threshold_bps=10_000.0,  # disable vol override
    )
    r = None
    for i in range(8):
        r = det.observe(_tick(i + 1, 100.0 + i * 0.05))
    assert r is not None
    assert r.regime is MarketRegime.TRENDING_UP
    assert r.drift_bps > 0.0


def test_regime_detector_classifies_trending_down():
    det = RegimeDetector(
        window=16,
        trend_threshold_bps=5.0,
        volatility_threshold_bps=10_000.0,
    )
    r = None
    for i in range(8):
        r = det.observe(_tick(i + 1, 100.0 - i * 0.05))
    assert r is not None
    assert r.regime is MarketRegime.TRENDING_DOWN


def test_regime_detector_classifies_ranging():
    det = RegimeDetector(
        window=16,
        trend_threshold_bps=200.0,  # very wide trend band
        volatility_threshold_bps=200.0,
    )
    r = None
    for i in range(8):
        # Tiny oscillation around 100.0.
        r = det.observe(_tick(i + 1, 100.0 + (0.001 if i % 2 else -0.001)))
    assert r is not None
    assert r.regime is MarketRegime.RANGING


def test_regime_detector_classifies_volatile_overrides_trend():
    det = RegimeDetector(
        window=8,
        trend_threshold_bps=5.0,
        volatility_threshold_bps=20.0,
    )
    # Big swing → high realised vol.
    prices = [100.0, 100.5, 99.5, 101.0, 99.0, 101.5, 98.5, 102.0]
    r = None
    for i, p in enumerate(prices):
        r = det.observe(_tick(i + 1, p))
    assert r is not None
    assert r.regime is MarketRegime.VOLATILE


def test_regime_detector_unknown_on_crossed_book():
    det = RegimeDetector(window=8)
    bad = MarketTick(
        ts_ns=1, symbol="X", bid=100.0, ask=99.0, last=99.5
    )
    r = det.observe(bad)
    assert r.regime is MarketRegime.UNKNOWN


def test_regime_detector_replay_determinism():
    def run() -> tuple:
        det = RegimeDetector(window=8)
        out = []
        for i in range(20):
            r = det.observe(_tick(i + 1, 100.0 + i * 0.01))
            out.append((r.regime, round(r.drift_bps, 6)))
        return tuple(out)

    assert run() == run()


def test_regime_detector_validates_args():
    with pytest.raises(ValueError):
        RegimeDetector(window=2)
    with pytest.raises(ValueError):
        RegimeDetector(trend_threshold_bps=-1.0)
    with pytest.raises(ValueError):
        RegimeDetector(volatility_threshold_bps=-1.0)


# ---------------------------------------------------------------------------
# StrategyScheduler
# ---------------------------------------------------------------------------


def test_scheduler_cadence_one_fires_every_step():
    sch = StrategyScheduler()
    sch.register(strategy_id="a", cadence=1)
    assert sch.step(1) == ("a",)
    assert sch.step(2) == ("a",)
    assert sch.step(3) == ("a",)
    assert sch.get("a").fires == 3


def test_scheduler_cadence_three_fires_every_third_step():
    sch = StrategyScheduler()
    sch.register(strategy_id="a", cadence=3)
    assert sch.step(1) == ()
    assert sch.step(2) == ()
    assert sch.step(3) == ("a",)
    assert sch.step(4) == ()
    assert sch.step(5) == ()
    assert sch.step(6) == ("a",)


def test_scheduler_validates_args():
    sch = StrategyScheduler()
    with pytest.raises(ValueError):
        sch.register(strategy_id="", cadence=1)
    with pytest.raises(ValueError):
        sch.register(strategy_id="a", cadence=0)
    sch.register(strategy_id="a", cadence=1)
    with pytest.raises(ValueError):
        sch.register(strategy_id="a", cadence=1)


def test_scheduler_deregister_removes_from_step():
    sch = StrategyScheduler()
    sch.register(strategy_id="a", cadence=1)
    sch.register(strategy_id="b", cadence=1)
    sch.deregister("a")
    assert sch.step(1) == ("b",)


# ---------------------------------------------------------------------------
# StrategyOrchestrator
# ---------------------------------------------------------------------------


def _promote_to(fsm: StrategyStateMachine, sid: str, target: StrategyState):
    fsm.propose(strategy_id=sid, ts_ns=1)
    path = [
        StrategyState.CANARY,
        StrategyState.LIVE,
    ]
    target_idx = path.index(target) if target in path else -1
    for i, st in enumerate(path[: target_idx + 1] if target_idx >= 0 else []):
        fsm.transition(
            strategy_id=sid,
            new_state=st,
            ts_ns=2 + i,
            reason="promote",
        )


def test_orchestrator_eligibility_respects_min_state():
    fsm = StrategyStateMachine()
    orc = StrategyOrchestrator(fsm)
    _promote_to(fsm, "a", StrategyState.CANARY)
    _promote_to(fsm, "b", StrategyState.LIVE)
    orc.register(strategy_id="a", min_state=StrategyState.LIVE)
    orc.register(strategy_id="b", min_state=StrategyState.LIVE)
    eligible = orc.eligible(MarketRegime.TRENDING_UP)
    assert eligible == ("b",)


def test_orchestrator_eligibility_respects_allowed_regimes():
    fsm = StrategyStateMachine()
    orc = StrategyOrchestrator(fsm)
    _promote_to(fsm, "trend_only", StrategyState.LIVE)
    _promote_to(fsm, "anywhere", StrategyState.LIVE)
    orc.register(
        strategy_id="trend_only",
        allowed_regimes=(MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN),
        min_state=StrategyState.CANARY,
    )
    orc.register(strategy_id="anywhere", min_state=StrategyState.CANARY)
    assert set(orc.eligible(MarketRegime.TRENDING_UP)) == {
        "trend_only",
        "anywhere",
    }
    assert set(orc.eligible(MarketRegime.RANGING)) == {"anywhere"}


def test_orchestrator_terminal_strategy_never_eligible():
    fsm = StrategyStateMachine()
    orc = StrategyOrchestrator(fsm)
    fsm.propose(strategy_id="a", ts_ns=1)
    fsm.transition(
        strategy_id="a",
        new_state=StrategyState.FAILED,
        ts_ns=2,
        reason="kill",
    )
    orc.register(strategy_id="a", min_state=StrategyState.CANARY)
    assert orc.eligible(MarketRegime.TRENDING_UP) == ()


def test_orchestrator_unregister_drops_from_eligibility():
    fsm = StrategyStateMachine()
    orc = StrategyOrchestrator(fsm)
    _promote_to(fsm, "a", StrategyState.LIVE)
    orc.register(strategy_id="a")
    assert orc.eligible(MarketRegime.TRENDING_UP) == ("a",)
    orc.deregister("a")
    assert orc.eligible(MarketRegime.TRENDING_UP) == ()


# ---------------------------------------------------------------------------
# ConflictResolver
# ---------------------------------------------------------------------------


def _signal(symbol: str, side: Side, conf: float, plugin: str) -> SignalEvent:
    return SignalEvent(
        ts_ns=1,
        symbol=symbol,
        side=side,
        confidence=conf,
        plugin_chain=(plugin,),
        meta={},
    )


def test_resolver_buy_wins_over_weaker_sell():
    r = ConflictResolver()
    out = r.resolve(
        [
            _signal("X", Side.BUY, 0.7, "p1"),
            _signal("X", Side.SELL, 0.3, "p2"),
        ]
    )
    assert len(out) == 1
    coalesced, resolution = out[0]
    assert coalesced.side is Side.BUY
    assert pytest.approx(resolution.net_score, abs=1e-9) == 0.4
    assert pytest.approx(coalesced.confidence, abs=1e-9) == 0.4


def test_resolver_balanced_signals_collapse_to_hold():
    r = ConflictResolver(min_net_score=1e-6)
    out = r.resolve(
        [
            _signal("X", Side.BUY, 0.5, "p1"),
            _signal("X", Side.SELL, 0.5, "p2"),
        ]
    )
    coalesced, _ = out[0]
    assert coalesced.side is Side.HOLD
    assert coalesced.confidence == 0.0


def test_resolver_balanced_signals_collapse_to_hold_default_threshold():
    """Default ``min_net_score=0.0`` must still collapse a perfectly balanced
    BUY/SELL pair to HOLD with zero (non-negative) confidence.

    Regression for Devin Review BUG_pr-review-job-13b551b6da21436ba5d4f4bc83877512_0001:
    previously ``abs(net) < 0.0`` evaluated False for net == 0.0, falling
    through to the SELL branch with ``min(1.0, -0.0) == -0.0``.
    """

    r = ConflictResolver()
    out = r.resolve(
        [
            _signal("X", Side.BUY, 0.5, "p1"),
            _signal("X", Side.SELL, 0.5, "p2"),
        ]
    )
    coalesced, resolution = out[0]
    assert coalesced.side is Side.HOLD
    assert coalesced.confidence == 0.0
    # Guard against negative-zero leaking into downstream consumers.
    assert math.copysign(1.0, coalesced.confidence) > 0.0
    assert resolution.winning_side is Side.HOLD
    assert resolution.net_score == 0.0


def test_resolver_groups_by_symbol():
    r = ConflictResolver()
    out = r.resolve(
        [
            _signal("X", Side.BUY, 0.6, "p1"),
            _signal("Y", Side.SELL, 0.4, "p2"),
            _signal("X", Side.BUY, 0.2, "p3"),
        ]
    )
    assert len(out) == 2
    by_sym = {c.symbol: (c, res) for c, res in out}
    assert by_sym["X"][0].side is Side.BUY
    # 0.6 + 0.2 = 0.8 net BUY for X
    assert pytest.approx(by_sym["X"][1].net_score, abs=1e-9) == 0.8
    assert by_sym["Y"][0].side is Side.SELL


def test_resolver_below_threshold_emits_hold():
    r = ConflictResolver(min_net_score=0.5)
    out = r.resolve(
        [
            _signal("X", Side.BUY, 0.4, "p1"),
            _signal("X", Side.SELL, 0.1, "p2"),
        ]
    )
    coalesced, _ = out[0]
    # Net = 0.3 < 0.5 → HOLD
    assert coalesced.side is Side.HOLD


def test_resolver_plugin_chain_union_preserves_order():
    r = ConflictResolver()
    out = r.resolve(
        [
            _signal("X", Side.BUY, 0.5, "first"),
            _signal("X", Side.BUY, 0.5, "second"),
            _signal("X", Side.BUY, 0.5, "first"),  # duplicate
        ]
    )
    coalesced, _ = out[0]
    assert coalesced.plugin_chain == ("first", "second")


def test_resolver_empty_input_empty_output():
    r = ConflictResolver()
    assert r.resolve([]) == ()


def test_resolver_replay_determinism():
    r = ConflictResolver()
    inputs = [
        _signal("X", Side.BUY, 0.6, "p1"),
        _signal("X", Side.SELL, 0.2, "p2"),
        _signal("Y", Side.SELL, 0.4, "p3"),
    ]
    a = r.resolve(inputs)
    b = r.resolve(inputs)
    assert tuple((s.symbol, s.side, s.confidence) for s, _ in a) == tuple(
        (s.symbol, s.side, s.confidence) for s, _ in b
    )
