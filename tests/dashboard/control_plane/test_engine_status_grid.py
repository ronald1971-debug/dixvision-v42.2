"""DASH-EG-01 — EngineStatusGrid widget tests."""

from __future__ import annotations

from core.contracts.engine import HealthState, HealthStatus
from dashboard.control_plane.engine_status_grid import EngineStatusGrid


class _FakeEngine:
    def __init__(self, status: HealthStatus) -> None:
        self._status = status

    def check_self(self) -> HealthStatus:
        return self._status


def test_grid_buckets_health_states():
    engines = {
        "execution": _FakeEngine(
            HealthStatus(state=HealthState.OK, detail="all good")
        ),
        "intelligence": _FakeEngine(
            HealthStatus(state=HealthState.DEGRADED, detail="lag")
        ),
        "system": _FakeEngine(
            HealthStatus(state=HealthState.FAIL, detail="halted")
        ),
    }
    grid = EngineStatusGrid(engines=engines)
    rows = grid.snapshot()
    assert tuple(r.engine_name for r in rows) == (
        "execution",
        "intelligence",
        "system",
    )
    assert rows[0].bucket == "alive"
    assert rows[1].bucket == "degraded"
    assert rows[2].bucket == "halted"


def test_grid_handles_missing_check_self():
    grid = EngineStatusGrid(engines={"broken": object()})
    rows = grid.snapshot()
    assert len(rows) == 1
    assert rows[0].bucket == "offline"
    assert "missing check_self" in rows[0].detail


def test_grid_handles_check_self_exception():
    class _Boom:
        def check_self(self) -> HealthStatus:
            raise RuntimeError("kaboom")

    grid = EngineStatusGrid(engines={"boom": _Boom()})
    rows = grid.snapshot()
    assert rows[0].bucket == "offline"
    assert "check_self raised" in rows[0].detail


def test_grid_renders_plugin_state_rows():
    status = HealthStatus(
        state=HealthState.OK,
        detail="ok",
        plugin_states={
            "alpha": {"plug_a": HealthState.OK},
            "beta": {"plug_b": HealthState.DEGRADED},
        },
    )
    grid = EngineStatusGrid(engines={"intelligence": _FakeEngine(status)})
    rows = grid.snapshot()
    plugins = rows[0].plugin_states
    assert ("alpha", "plug_a", "OK") in plugins
    assert ("beta", "plug_b", "DEGRADED") in plugins
