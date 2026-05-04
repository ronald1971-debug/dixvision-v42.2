"""AUDIT-WIRE.4 regression tests \u2014 SensorArray bound into SystemEngine.

P1-3 closure. ``SystemEngine`` carried no registry of the 12 hazard
sensors before this PR; the primitives existed but were never bound.
This test pins the wiring: the harness builds a SensorArray with all
12 frozen HAZ-XX sensors registered and hands it to ``SystemEngine``;
the engine exposes the same instance on ``self.sensor_array``.
"""

from __future__ import annotations

import pytest

from system_engine.hazard_sensors import SensorArray

EXPECTED_SENSOR_NAMES = frozenset(
    {
        "clock_drift",
        "ws_timeout",
        "exchange_unreachable",
        "stale_data",
        "memory_overflow",
        "latency_spike",
        "heartbeat_missed",
        "risk_snapshot_stale",
        "order_flood",
        "runtime_breaker_open",
        "market_anomaly",
        "system_anomaly",
    }
)


@pytest.fixture()
def state():
    from ui.server import _State

    return _State()


def test_audit_wire_4_state_owns_sensor_array(state):
    assert isinstance(state.sensor_array, SensorArray)
    assert state.system.sensor_array is state.sensor_array


def test_audit_wire_4_all_twelve_haz_sensors_registered(state):
    names = {s.name for s in state.sensor_array.sensors}
    assert names == EXPECTED_SENSOR_NAMES
    assert len(state.sensor_array) == 12


def test_audit_wire_4_check_self_reports_sensor_array_online(state):
    """``SystemEngine.check_self`` previously returned the Phase E0
    'no sensors loaded' string regardless of wiring. With this PR the
    health surface reports the array is online and how many sensors
    are loaded \u2014 a cheap end-to-end signal that the wiring landed."""

    status = state.system.check_self()
    assert "sensor_array online" in status.detail
    assert "12 sensors" in status.detail
