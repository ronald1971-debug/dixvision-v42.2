"""Tests for HazardThrottleAdapter (P0-2).

The adapter closes the BEHAVIOR-P3 chain: HazardObserver +
compute_throttle + apply_throttle. Until P0-2 the three pieces
existed but nothing wired them together, so apply_throttle() was
never called on the live hot path.
"""

from __future__ import annotations

import pytest

from core.contracts.events import HazardEvent, HazardSeverity
from core.contracts.risk import RiskSnapshot
from system_engine.coupling import (
    HazardThrottleAdapter,
    HazardThrottleConfig,
)


def _baseline(*, halted: bool = False) -> RiskSnapshot:
    return RiskSnapshot(
        version=1,
        ts_ns=1_000,
        max_position_qty=10.0,
        max_signal_confidence=0.0,
        symbol_caps={"BTCUSD": 5.0},
        halted=halted,
    )


def _hazard(
    *,
    code: str = "HAZ-DATA-STALENESS",
    severity: HazardSeverity = HazardSeverity.MEDIUM,
    ts_ns: int = 1_000,
) -> HazardEvent:
    return HazardEvent(
        ts_ns=ts_ns,
        code=code,
        severity=severity,
        detail="test",
        source="system",
        produced_by_engine="system",
    )


def test_adapter_constructs_with_default_config() -> None:
    adapter = HazardThrottleAdapter()
    assert adapter.config.version == HazardThrottleConfig.default().version


def test_no_observations_yields_neutral_snapshot() -> None:
    adapter = HazardThrottleAdapter()
    baseline = _baseline()
    projected = adapter.project(snapshot=baseline, now_ns=2_000)
    assert projected == baseline


def test_observe_then_project_tightens_snapshot() -> None:
    adapter = HazardThrottleAdapter()
    adapter.observe(_hazard(severity=HazardSeverity.MEDIUM, ts_ns=1_500))

    projected = adapter.project(snapshot=_baseline(), now_ns=1_500)

    assert projected.max_position_qty is not None
    assert projected.max_position_qty < 10.0
    assert projected.symbol_caps["BTCUSD"] < 5.0


def test_critical_hazard_halts_snapshot() -> None:
    adapter = HazardThrottleAdapter()
    adapter.observe(_hazard(severity=HazardSeverity.CRITICAL, ts_ns=1_500))

    projected = adapter.project(snapshot=_baseline(), now_ns=1_500)

    assert projected.halted is True


def test_halt_is_monotonic() -> None:
    """A baseline that is already halted stays halted regardless of decision."""

    adapter = HazardThrottleAdapter()
    projected = adapter.project(snapshot=_baseline(halted=True), now_ns=2_000)
    assert projected.halted is True


def test_decayed_observations_stop_throttling() -> None:
    adapter = HazardThrottleAdapter()
    adapter.observe(_hazard(severity=HazardSeverity.MEDIUM, ts_ns=1_000))

    far_future = 1_000 + 10**12
    projected = adapter.project(snapshot=_baseline(), now_ns=far_future)
    assert projected == _baseline()


def test_observe_accepts_raw_hazard_event() -> None:
    adapter = HazardThrottleAdapter()
    adapter.observe(_hazard())
    decision = adapter.current_decision(now_ns=1_500)
    assert decision.contributing_codes == ("HAZ-DATA-STALENESS",)


def test_active_observations_passthrough() -> None:
    adapter = HazardThrottleAdapter()
    adapter.observe(_hazard(ts_ns=1_500))
    active = adapter.active_observations(now_ns=1_500)
    assert len(active) == 1
    assert active[0].code == "HAZ-DATA-STALENESS"


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        HazardThrottleAdapter(capacity=0)
