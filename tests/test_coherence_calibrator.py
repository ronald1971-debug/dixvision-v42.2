"""Phase 6.T1c reader — coherence calibrator (INV-53 reader side).

Covers:
* Empty input → zero report.
* Belief / pressure / audit / reward aggregates.
* Window bounds filter out-of-range events.
* Realised-direction matching (BUY → +PnL, SELL → −PnL, HOLD skipped).
* Replay determinism (INV-15) — same inputs ⇒ same report.
* INV-47 — raw PnL preserved alongside shaped reward.
* ``to_event`` projects every numeric field into the SystemEvent payload.
"""

from __future__ import annotations

from core.contracts.events import (
    EventKind,
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SystemEvent,
    SystemEventKind,
)
from learning_engine.calibration.coherence_calibrator import (
    CALIBRATION_REPORT_VERSION,
    CALIBRATOR_SOURCE,
    CalibrationReport,
    calibrate_coherence_window,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _belief(ts_ns: int, regime_confidence: float) -> SystemEvent:
    return SystemEvent(
        ts_ns=ts_ns,
        sub_kind=SystemEventKind.BELIEF_STATE_SNAPSHOT,
        source="core.coherence.belief_state",
        payload={
            "regime": "TREND_UP",
            "regime_confidence": f"{regime_confidence:.6f}",
            "consensus_side": "BUY",
            "signal_count": "5",
            "avg_confidence": "0.7",
            "symbols": "EURUSD",
            "version": "v3.3-T1a",
        },
    )


def _pressure(
    ts_ns: int,
    *,
    perf: float = 0.1,
    risk: float = 0.2,
    drift: float = 0.3,
    latency: float = 0.4,
    uncertainty: float = 0.5,
    safety_modifier: float = 0.9,
) -> SystemEvent:
    return SystemEvent(
        ts_ns=ts_ns,
        sub_kind=SystemEventKind.PRESSURE_VECTOR_SNAPSHOT,
        source="core.coherence.performance_pressure",
        payload={
            "perf": f"{perf:.6f}",
            "risk": f"{risk:.6f}",
            "drift": f"{drift:.6f}",
            "latency": f"{latency:.6f}",
            "uncertainty": f"{uncertainty:.6f}",
            "safety_modifier": f"{safety_modifier:.6f}",
            "cross_signal_entropy": "0.10",
            "signal_count": "5",
            "version": "v3.3-T1a",
        },
    )


def _audit(
    ts_ns: int,
    *,
    decision_side: Side,
    decision_confidence: float = 0.6,
    fallback: bool = False,
) -> SystemEvent:
    return SystemEvent(
        ts_ns=ts_ns,
        sub_kind=SystemEventKind.META_AUDIT,
        source="intelligence_engine.meta_controller.runtime_adapter",
        payload={
            "version": "v3.3-T1c",
            "proposed_side": decision_side.value,
            "regime": "TREND_UP",
            "regime_transitioned": "false",
            "elapsed_ns": "100000",
            "decision_side": decision_side.value,
            "decision_size": "0.05",
            "decision_confidence": f"{decision_confidence:.6f}",
            "decision_fallback": "true" if fallback else "false",
        },
    )


def _reward(
    ts_ns: int,
    *,
    raw_pnl: float,
    components: tuple[tuple[str, float], ...],
) -> SystemEvent:
    payload: dict[str, str] = {
        "raw_pnl": f"{raw_pnl:.6f}",
        "shaped_reward": f"{sum(v for _, v in components):.6f}",
        "shaping_version": "v3.3-J3",
    }
    for name, value in components:
        payload[f"c.{name}"] = f"{value:.6f}"
    return SystemEvent(
        ts_ns=ts_ns,
        sub_kind=SystemEventKind.REWARD_BREAKDOWN,
        source="learning_engine.lanes.reward_shaping",
        payload=payload,
    )


def _fill(
    ts_ns: int,
    side: Side,
    realised_pnl: float,
    *,
    status: ExecutionStatus = ExecutionStatus.FILLED,
) -> ExecutionEvent:
    return ExecutionEvent(
        ts_ns=ts_ns,
        symbol="EURUSD",
        side=side,
        qty=1.0,
        price=1.1,
        status=status,
        venue="paper",
        order_id=f"O{ts_ns}",
        meta={"realised_pnl": f"{realised_pnl:.6f}"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_input_yields_zero_report():
    report = calibrate_coherence_window(ts_ns=1_000, events=())
    assert isinstance(report, CalibrationReport)
    assert report.belief_count == 0
    assert report.pressure_count == 0
    assert report.audit_count == 0
    assert report.reward_count == 0
    assert report.audit_match_known is False
    assert report.audit_directional_match_rate == 0.0
    assert report.belief_calibration_gap == 0.0
    assert report.window_start_ns == 1_000
    assert report.window_end_ns == 1_000
    assert report.version == CALIBRATION_REPORT_VERSION


def test_belief_avg_regime_confidence():
    events = (
        _belief(ts_ns=10, regime_confidence=0.40),
        _belief(ts_ns=20, regime_confidence=0.60),
        _belief(ts_ns=30, regime_confidence=0.80),
    )
    report = calibrate_coherence_window(ts_ns=100, events=events)
    assert report.belief_count == 3
    assert abs(report.belief_avg_regime_confidence - 0.60) < 1e-9
    assert report.window_start_ns == 10
    assert report.window_end_ns == 30


def test_pressure_aggregates_each_dimension():
    events = (
        _pressure(
            ts_ns=10,
            perf=0.0,
            risk=0.0,
            drift=0.0,
            latency=0.0,
            uncertainty=0.0,
            safety_modifier=1.0,
        ),
        _pressure(
            ts_ns=20,
            perf=1.0,
            risk=1.0,
            drift=1.0,
            latency=1.0,
            uncertainty=1.0,
            safety_modifier=0.0,
        ),
    )
    report = calibrate_coherence_window(ts_ns=100, events=events)
    assert report.pressure_count == 2
    assert abs(report.pressure_avg_perf - 0.5) < 1e-9
    assert abs(report.pressure_avg_risk - 0.5) < 1e-9
    assert abs(report.pressure_avg_drift - 0.5) < 1e-9
    assert abs(report.pressure_avg_latency - 0.5) < 1e-9
    assert abs(report.pressure_avg_uncertainty - 0.5) < 1e-9
    assert abs(report.pressure_avg_safety_modifier - 0.5) < 1e-9


def test_audit_fallback_rate_and_avg_confidence():
    events = (
        _audit(ts_ns=10, decision_side=Side.BUY, decision_confidence=0.5),
        _audit(
            ts_ns=20,
            decision_side=Side.BUY,
            decision_confidence=0.7,
            fallback=True,
        ),
        _audit(
            ts_ns=30,
            decision_side=Side.SELL,
            decision_confidence=0.9,
            fallback=True,
        ),
    )
    report = calibrate_coherence_window(ts_ns=100, events=events)
    assert report.audit_count == 3
    assert abs(report.audit_avg_decision_confidence - 0.7) < 1e-9
    assert abs(report.audit_fallback_rate - (2 / 3)) < 1e-9


def test_audit_match_unknown_when_no_fills():
    events = (
        _audit(ts_ns=10, decision_side=Side.BUY),
        _audit(ts_ns=20, decision_side=Side.SELL),
    )
    report = calibrate_coherence_window(ts_ns=100, events=events)
    assert report.audit_match_known is False
    assert report.audit_directional_match_rate == 0.0
    assert report.belief_calibration_gap == 0.0


def test_audit_match_buy_correct_when_pnl_positive():
    events = (
        _belief(ts_ns=10, regime_confidence=0.8),
        _audit(ts_ns=10, decision_side=Side.BUY),
    )
    fills = (_fill(ts_ns=12, side=Side.BUY, realised_pnl=1.5),)
    report = calibrate_coherence_window(
        ts_ns=100, events=events, fills=fills
    )
    assert report.audit_match_known is True
    assert report.audit_directional_match_rate == 1.0
    # belief avg 0.8, match rate 1.0 → gap = -0.2 (underconfident).
    assert abs(report.belief_calibration_gap - (-0.2)) < 1e-9


def test_audit_match_sell_correct_when_pnl_negative():
    events = (_audit(ts_ns=10, decision_side=Side.SELL),)
    fills = (_fill(ts_ns=15, side=Side.SELL, realised_pnl=-0.5),)
    report = calibrate_coherence_window(
        ts_ns=100, events=events, fills=fills
    )
    assert report.audit_match_known is True
    assert report.audit_directional_match_rate == 1.0


def test_audit_hold_is_skipped_in_match_rate():
    events = (
        _audit(ts_ns=10, decision_side=Side.HOLD),
        _audit(ts_ns=20, decision_side=Side.BUY),
    )
    fills = (
        _fill(ts_ns=11, side=Side.BUY, realised_pnl=0.0),
        _fill(ts_ns=21, side=Side.BUY, realised_pnl=2.0),
    )
    report = calibrate_coherence_window(
        ts_ns=100, events=events, fills=fills
    )
    # Only the second audit/fill pair contributes; PnL>0, side=BUY → 1/1.
    assert report.audit_match_known is True
    assert report.audit_directional_match_rate == 1.0


def test_audit_match_mixed_directional_accuracy():
    events = (
        _audit(ts_ns=10, decision_side=Side.BUY),
        _audit(ts_ns=20, decision_side=Side.SELL),
        _audit(ts_ns=30, decision_side=Side.BUY),
    )
    fills = (
        _fill(ts_ns=11, side=Side.BUY, realised_pnl=1.0),    # correct
        _fill(ts_ns=21, side=Side.SELL, realised_pnl=1.0),   # WRONG (pos)
        _fill(ts_ns=31, side=Side.BUY, realised_pnl=2.0),    # correct
    )
    report = calibrate_coherence_window(
        ts_ns=100, events=events, fills=fills
    )
    assert report.audit_match_known is True
    assert abs(report.audit_directional_match_rate - (2 / 3)) < 1e-9


def test_audit_zero_pnl_is_skipped():
    events = (_audit(ts_ns=10, decision_side=Side.BUY),)
    fills = (_fill(ts_ns=11, side=Side.BUY, realised_pnl=0.0),)
    report = calibrate_coherence_window(
        ts_ns=100, events=events, fills=fills
    )
    assert report.audit_match_known is False
    assert report.audit_directional_match_rate == 0.0


def test_reward_aggregates_components_and_preserves_raw_pnl():
    events = (
        _reward(
            ts_ns=10,
            raw_pnl=2.5,
            components=(("consensus", 0.5), ("strength", 0.25)),
        ),
        _reward(
            ts_ns=20,
            raw_pnl=-1.0,
            components=(("consensus", -0.2), ("kelly_penalty", -0.1)),
        ),
    )
    report = calibrate_coherence_window(ts_ns=100, events=events)
    assert report.reward_count == 2
    # INV-47: raw PnL preserved alongside shaped reward.
    assert abs(report.reward_total_raw_pnl - 1.5) < 1e-9
    assert abs(report.reward_total_shaped - (0.75 + (-0.30))) < 1e-9
    components = dict(report.reward_components)
    assert abs(components["consensus"] - 0.3) < 1e-9
    assert abs(components["strength"] - 0.25) < 1e-9
    assert abs(components["kelly_penalty"] - (-0.1)) < 1e-9
    # Sorted by name for replay-determinism.
    assert [name for name, _ in report.reward_components] == sorted(
        components.keys()
    )


def test_window_bounds_filter_events():
    events = (
        _belief(ts_ns=10, regime_confidence=0.10),
        _belief(ts_ns=50, regime_confidence=0.50),
        _belief(ts_ns=90, regime_confidence=0.90),
    )
    report = calibrate_coherence_window(
        ts_ns=100,
        events=events,
        window_start_ns=20,
        window_end_ns=80,
    )
    # Only the ts=50 snapshot should fall inside [20, 80].
    assert report.belief_count == 1
    assert abs(report.belief_avg_regime_confidence - 0.5) < 1e-9
    assert report.window_start_ns == 20
    assert report.window_end_ns == 80


def test_unknown_event_kinds_are_ignored():
    events = (
        _belief(ts_ns=10, regime_confidence=0.4),
        SystemEvent(
            ts_ns=20,
            sub_kind=SystemEventKind.HEARTBEAT,
            source="ignored",
            payload={"foo": "bar"},
        ),
    )
    report = calibrate_coherence_window(ts_ns=100, events=events)
    assert report.belief_count == 1
    assert report.pressure_count == 0


def test_partially_filled_counts_as_realised():
    events = (_audit(ts_ns=10, decision_side=Side.BUY),)
    fills = (
        _fill(
            ts_ns=11,
            side=Side.BUY,
            realised_pnl=0.5,
            status=ExecutionStatus.PARTIALLY_FILLED,
        ),
    )
    report = calibrate_coherence_window(
        ts_ns=100, events=events, fills=fills
    )
    assert report.audit_match_known is True
    assert report.audit_directional_match_rate == 1.0


def test_non_terminal_fill_is_ignored():
    events = (_audit(ts_ns=10, decision_side=Side.BUY),)
    fills = (
        _fill(
            ts_ns=11,
            side=Side.BUY,
            realised_pnl=99.0,
            status=ExecutionStatus.PROPOSED,
        ),
    )
    report = calibrate_coherence_window(
        ts_ns=100, events=events, fills=fills
    )
    assert report.audit_match_known is False


def test_replay_determinism_inv15():
    events = (
        _belief(ts_ns=10, regime_confidence=0.4),
        _pressure(ts_ns=10, perf=0.1),
        _audit(ts_ns=10, decision_side=Side.BUY),
        _reward(
            ts_ns=10,
            raw_pnl=1.0,
            components=(("consensus", 0.2), ("strength", 0.1)),
        ),
    )
    fills = (_fill(ts_ns=12, side=Side.BUY, realised_pnl=2.0),)
    a = calibrate_coherence_window(ts_ns=100, events=events, fills=fills)
    b = calibrate_coherence_window(ts_ns=100, events=events, fills=fills)
    assert a == b


def test_to_event_round_trip():
    events = (
        _belief(ts_ns=10, regime_confidence=0.6),
        _pressure(ts_ns=10, latency=0.4),
        _audit(ts_ns=10, decision_side=Side.BUY, decision_confidence=0.7),
        _reward(
            ts_ns=10,
            raw_pnl=0.8,
            components=(("consensus", 0.3),),
        ),
    )
    fills = (_fill(ts_ns=12, side=Side.BUY, realised_pnl=1.0),)
    report = calibrate_coherence_window(
        ts_ns=200, events=events, fills=fills
    )
    ev = report.to_event()
    assert ev.kind is EventKind.SYSTEM
    assert ev.sub_kind is SystemEventKind.CALIBRATION_REPORT
    assert ev.ts_ns == 200
    assert ev.source == CALIBRATOR_SOURCE
    p = ev.payload
    assert p["version"] == CALIBRATION_REPORT_VERSION
    assert p["belief_count"] == "1"
    assert p["pressure_count"] == "1"
    assert p["audit_count"] == "1"
    assert p["reward_count"] == "1"
    assert p["audit_match_known"] == "true"
    assert p["audit_directional_match_rate"] == "1.000000"
    assert p["reward_component_count"] == "1"
    assert p["reward_component__consensus"] == "0.300000"
    assert p["reward_total_raw_pnl"] == "0.800000"


def test_resolved_window_uses_event_bounds_when_unspecified():
    events = (
        _belief(ts_ns=50, regime_confidence=0.4),
        _pressure(ts_ns=80),
    )
    fills = (_fill(ts_ns=120, side=Side.BUY, realised_pnl=1.0),)
    report = calibrate_coherence_window(
        ts_ns=999, events=events, fills=fills
    )
    assert report.window_start_ns == 50
    assert report.window_end_ns == 120
