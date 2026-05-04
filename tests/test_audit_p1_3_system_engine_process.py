"""AUDIT-P1.3 regression tests — ``SystemEngine.process`` polls sensors.

Closes the residual gap from PR #193 (AUDIT-WIRE.4): the
:class:`~system_engine.hazard_sensors.sensor_array.SensorArray` was
bound onto :class:`~system_engine.engine.SystemEngine` but never
invoked, so :meth:`process` returned ``()`` regardless of accumulated
hazard state.

After this PR every event flowing through ``SystemEngine.process``
drives the union of *pollable* sensors (those with the canonical
``observe(ts_ns)`` signature). The four sensors with bespoke
``observe(...)`` shapes are intentionally skipped — they ride their
dedicated ingestion paths (NewsFanout, runtime monitor, etc.) and
are documented as such in :func:`SystemEngine.process`.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import pytest

from core.contracts.events import (
    HazardEvent,
    HazardSeverity,
    SystemEvent,
    SystemEventKind,
)
from system_engine.engine import SystemEngine, _is_pollable
from system_engine.hazard_sensors.heartbeat_missed import (
    HeartbeatMissedSensor,
)
from system_engine.hazard_sensors.order_flood import OrderFloodSensor
from system_engine.hazard_sensors.sensor_array import SensorArray

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tick(ts_ns: int) -> SystemEvent:
    """Build a minimal :class:`SystemEvent` heartbeat."""

    return SystemEvent(
        ts_ns=ts_ns,
        sub_kind=SystemEventKind.HEARTBEAT,
        source="tests",
        produced_by_engine="system_engine",
    )


class _StubMultiArgSensor:
    """Sensor whose ``observe`` takes more than ``ts_ns`` — must NOT be polled."""

    name: str = "stub_multi_arg"
    code: str = "HAZ-STUB-MULTI"

    def observe(
        self, *, ts_ns: int, payload: int
    ) -> tuple[HazardEvent, ...]:
        # If this ever fires the test will see HAZ-STUB-MULTI rows.
        return (
            HazardEvent(
                ts_ns=ts_ns,
                code=self.code,
                severity=HazardSeverity.LOW,
                source="tests",
                detail=f"stub fired with payload={payload}",
                meta={},
                produced_by_engine="system_engine",
            ),
        )


# ---------------------------------------------------------------------------
# _is_pollable contract
# ---------------------------------------------------------------------------


def test_pollable_detector_accepts_canonical_signature() -> None:
    sensor = HeartbeatMissedSensor()
    assert _is_pollable(sensor)


def test_pollable_detector_rejects_multi_arg_observe() -> None:
    """A sensor with more than ``ts_ns`` is NOT pollable.

    Pin the negative case so a future refactor can't accidentally
    widen the detector to accept multi-arg shapes (which would call
    bespoke sensors with the wrong arguments and either crash or
    fabricate hazard rows).
    """

    assert not _is_pollable(_StubMultiArgSensor())


def test_pollable_detector_requires_param_name_ts_ns() -> None:
    """Renaming the parameter breaks pollability.

    INV-15 replay determinism — the engine binds ``ts_ns`` by
    keyword via ``observe(ts_ns)``, so a sensor whose first param
    is ``timestamp`` must declare its own ingestion path.
    """

    class _NamedDifferently:
        name = "weird"
        code = "HAZ-WEIRD"

        def observe(self, when: int) -> tuple[HazardEvent, ...]:
            return ()

    assert not _is_pollable(_NamedDifferently())


# ---------------------------------------------------------------------------
# SystemEngine.process invokes pollable sensors
# ---------------------------------------------------------------------------


def test_process_returns_empty_when_no_sensor_array() -> None:
    engine = SystemEngine()
    out = engine.process(_tick(1_000))
    assert out == ()


def test_process_emits_hazard_when_pollable_sensor_fires() -> None:
    """End-to-end: arm a sensor, tick past timeout, see the HAZ row."""

    array = SensorArray()
    sensor = HeartbeatMissedSensor(timeout_ns=1_000_000_000)
    array.register(sensor)
    engine = SystemEngine(sensor_array=array)

    # Arm the sensor with a heartbeat at t=0.
    sensor.on_heartbeat(engine="execution", ts_ns=0)
    # Tick within the timeout — no hazard.
    quiet = engine.process(_tick(500_000_000))
    assert quiet == ()
    # Tick past the timeout — HAZ-07 fires once.
    loud = engine.process(_tick(2_000_000_000))
    assert len(loud) == 1
    haz = loud[0]
    assert haz.code == "HAZ-07"
    assert haz.meta["engine"] == "execution"
    assert haz.severity == HazardSeverity.HIGH


def test_process_returns_pollable_arm_once() -> None:
    """Sensors arm-once; a second tick must not double-emit."""

    array = SensorArray()
    sensor = HeartbeatMissedSensor(timeout_ns=1_000_000_000)
    array.register(sensor)
    engine = SystemEngine(sensor_array=array)

    sensor.on_heartbeat(engine="execution", ts_ns=0)
    first = engine.process(_tick(2_000_000_000))
    assert len(first) == 1
    second = engine.process(_tick(3_000_000_000))
    assert second == ()


def test_process_ignores_multi_arg_sensors() -> None:
    """Sensors with bespoke ``observe`` shapes are NOT polled.

    A misconfigured engine that polled them would either crash on
    the missing keyword argument or fabricate hazards from a
    default — both are unacceptable. Pin the negative path.
    """

    array = SensorArray()
    array.register(_StubMultiArgSensor())
    engine = SystemEngine(sensor_array=array)
    out = engine.process(_tick(1_000_000_000))
    assert out == ()


def test_process_returns_in_registration_order() -> None:
    """When two pollable sensors fire on the same tick, registration order wins.

    Uses two different sensor types (``HeartbeatMissedSensor`` and
    ``OrderFloodSensor``) since the slotted sensor classes don't
    permit renaming the ``name`` attribute on instances.
    """

    array = SensorArray()
    hb = HeartbeatMissedSensor(timeout_ns=1_000_000_000)
    of = OrderFloodSensor(
        window_ns=1_000_000_000,
        max_orders=2,
    )
    array.register(hb)
    array.register(of)
    engine = SystemEngine(sensor_array=array)

    hb.on_heartbeat(engine="execution", ts_ns=0)
    of.record_order(1_500_000_000)
    of.record_order(1_600_000_000)
    of.record_order(1_700_000_000)
    out = engine.process(_tick(2_000_000_000))
    # heartbeat_missed registered first; its HAZ row is at index 0.
    assert out[0].code == "HAZ-07"
    # order_flood registered second; its HAZ row follows.
    assert any(haz.code == "HAZ-09" for haz in out)
    haz_codes = [haz.code for haz in out]
    assert haz_codes.index("HAZ-07") < haz_codes.index("HAZ-09")


# ---------------------------------------------------------------------------
# Process return-type contract
# ---------------------------------------------------------------------------


def test_process_signature_returns_sequence_of_hazard_events() -> None:
    """The engine's contract is :class:`Sequence` of :class:`HazardEvent`.

    Pin the type annotation so a future refactor that widens the
    return type to ``Sequence[Event]`` (which would implicitly
    permit non-hazard events to leak out of the system engine)
    fails this test.
    """

    sig = inspect.signature(SystemEngine.process)
    annotation = sig.return_annotation
    # ``__future__.annotations`` keeps the annotation as a string.
    assert annotation == "Sequence[HazardEvent]" or annotation is Sequence[
        HazardEvent
    ]


# ---------------------------------------------------------------------------
# health surface
# ---------------------------------------------------------------------------


def test_check_self_reports_pollable_count(monkeypatch: pytest.MonkeyPatch) -> None:
    array = SensorArray()
    array.register(HeartbeatMissedSensor())
    array.register(_StubMultiArgSensor())
    engine = SystemEngine(sensor_array=array)
    detail = engine.check_self().detail
    assert "2 sensors" in detail
    assert "1 pollable" in detail
