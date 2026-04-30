"""Pump-level tests for ui.feeds.bls_http (Wave-04.5 PR-3).

The pump is non-deterministic by design (real network), but every
test injects a fake ``post`` and a deterministic ``clock_ns`` so the
control-flow + telemetry contract is exercised without I/O.

Async tests use ``asyncio.run`` rather than pytest-asyncio (the repo
does not depend on it) — mirrors :mod:`tests.test_fred_http_pump`.
"""

from __future__ import annotations

import asyncio
import itertools
import json

import pytest

from core.contracts.macro import MacroObservation
from ui.feeds.bls_http import (
    BLS_API_BASE,
    SOURCE_TAG,
    BLSHTTPPump,
    BLSSeriesSpec,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _payload(
    *,
    status: str = "REQUEST_SUCCEEDED",
    series: list[dict[str, object]] | None = None,
) -> bytes:
    doc: dict[str, object] = {
        "status": status,
        "responseTime": 200,
        "message": [],
        "Results": {"series": series if series is not None else []},
    }
    return json.dumps(doc).encode("utf-8")


def _series_block(
    series_id: str,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    return {"seriesID": series_id, "data": rows}


class _FakeClock:
    """Monotonically increasing fake clock — every call returns ``last + step``."""

    def __init__(self, *, start: int = 1_000_000_000, step: int = 1) -> None:
        self._counter = itertools.count(start, step)

    def __call__(self) -> int:
        return next(self._counter)


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------


def test_pump_rejects_non_positive_poll_interval() -> None:
    with pytest.raises(ValueError, match="poll_interval_s"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="k",
            series=(BLSSeriesSpec(series_id="X"),),
            clock_ns=_FakeClock(),
            poll_interval_s=0.0,
        )


def test_pump_rejects_non_positive_reconnect_delay() -> None:
    with pytest.raises(ValueError, match="reconnect_delay_s"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="k",
            series=(BLSSeriesSpec(series_id="X"),),
            clock_ns=_FakeClock(),
            reconnect_delay_s=0.0,
        )


def test_pump_rejects_max_lt_floor_reconnect_delay() -> None:
    with pytest.raises(ValueError, match=">= reconnect_delay_s"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="k",
            series=(BLSSeriesSpec(series_id="X"),),
            clock_ns=_FakeClock(),
            reconnect_delay_s=10.0,
            reconnect_delay_max_s=5.0,
        )


def test_pump_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="k",
            series=(BLSSeriesSpec(series_id="X"),),
            clock_ns=_FakeClock(),
            source="",
        )


def test_pump_rejects_empty_registration_key() -> None:
    with pytest.raises(ValueError, match="registration_key"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="",
            series=(BLSSeriesSpec(series_id="X"),),
            clock_ns=_FakeClock(),
        )


def test_pump_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="series spec"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="k",
            series=(),
            clock_ns=_FakeClock(),
        )


def test_blsseriesspec_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="series_id"):
        BLSSeriesSpec(series_id="")


def test_pump_rejects_partial_year_range() -> None:
    with pytest.raises(ValueError, match="start_year and end_year must be set together"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="k",
            series=(BLSSeriesSpec(series_id="X"),),
            clock_ns=_FakeClock(),
            start_year=2024,
            end_year=None,
        )
    with pytest.raises(ValueError, match="start_year and end_year must be set together"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="k",
            series=(BLSSeriesSpec(series_id="X"),),
            clock_ns=_FakeClock(),
            start_year=None,
            end_year=2024,
        )


def test_pump_rejects_pre_1900_year() -> None:
    with pytest.raises(ValueError, match=">= 1900"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="k",
            series=(BLSSeriesSpec(series_id="X"),),
            clock_ns=_FakeClock(),
            start_year=1800,
            end_year=2024,
        )


def test_pump_rejects_inverted_year_range() -> None:
    with pytest.raises(ValueError, match="end_year must be >= start_year"):
        BLSHTTPPump(
            sink=lambda _o: None,
            registration_key="k",
            series=(BLSSeriesSpec(series_id="X"),),
            clock_ns=_FakeClock(),
            start_year=2024,
            end_year=2020,
        )


def test_pump_dedups_specs_preserving_order() -> None:
    pump = BLSHTTPPump(
        sink=lambda _o: None,
        registration_key="k",
        series=(
            BLSSeriesSpec(series_id="A"),
            BLSSeriesSpec(series_id="B"),
            BLSSeriesSpec(series_id="A"),
        ),
        clock_ns=_FakeClock(),
    )
    assert pump.series_ids == ("A", "B")


# ---------------------------------------------------------------------------
# initial status snapshot
# ---------------------------------------------------------------------------


def test_pump_initial_status_is_idle() -> None:
    pump = BLSHTTPPump(
        sink=lambda _o: None,
        registration_key="k",
        series=(BLSSeriesSpec(series_id="X"),),
        clock_ns=_FakeClock(),
    )
    s = pump.status()
    assert s.running is False
    assert s.source == SOURCE_TAG
    assert s.series_ids == ("X",)
    assert s.last_poll_ts_ns is None
    assert s.last_observation_ts_ns is None
    assert s.observations_received == 0
    assert s.polls == 0
    assert s.errors == 0


# ---------------------------------------------------------------------------
# happy-path one-cycle run
# ---------------------------------------------------------------------------


def test_pump_run_one_cycle_emits_observations() -> None:
    received: list[MacroObservation] = []
    seen_urls: list[str] = []
    seen_bodies: list[str] = []

    async def fake_post(url: str, body: str) -> bytes:
        seen_urls.append(url)
        seen_bodies.append(body)
        return _payload(
            series=[
                _series_block(
                    "CPIAUCSL",
                    [{"year": "2024", "period": "M01", "value": "319.086"}],
                ),
                _series_block(
                    "UNRATE",
                    [{"year": "2024", "period": "M01", "value": "3.7"}],
                ),
            ],
        )

    pump = BLSHTTPPump(
        sink=received.append,
        registration_key="test-key",
        series=(
            BLSSeriesSpec(
                series_id="CPIAUCSL",
                units="Index 1982-1984=100",
                title="Headline CPI",
            ),
            BLSSeriesSpec(series_id="UNRATE", units="Percent"),
        ),
        clock_ns=_FakeClock(),
        post=fake_post,
        poll_interval_s=0.001,
    )

    async def _scenario() -> None:
        stop = asyncio.Event()

        async def _stopper() -> None:
            while pump.status().polls < 1:
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())

    asyncio.run(_scenario())

    assert seen_urls == [f"{BLS_API_BASE.rstrip('/')}/timeseries/data/"]
    parsed = json.loads(seen_bodies[0])
    assert parsed["registrationkey"] == "test-key"
    assert parsed["seriesid"] == ["CPIAUCSL", "UNRATE"]

    by_id = {o.series_id: o for o in received}
    assert set(by_id) == {"CPIAUCSL", "UNRATE"}
    assert by_id["CPIAUCSL"].units == "Index 1982-1984=100"
    assert by_id["CPIAUCSL"].title == "Headline CPI"
    assert by_id["UNRATE"].units == "Percent"

    s = pump.status()
    assert s.polls >= 1
    assert s.observations_received == 2 * s.polls or s.observations_received >= 2
    assert s.errors == 0
    assert s.last_poll_ts_ns is not None
    assert s.last_observation_ts_ns is not None


# ---------------------------------------------------------------------------
# error / backoff paths
# ---------------------------------------------------------------------------


def test_pump_transport_error_increments_errors_and_backs_off() -> None:
    received: list[MacroObservation] = []

    async def fake_post(url: str, body: str) -> bytes:
        raise OSError("simulated transport failure")

    pump = BLSHTTPPump(
        sink=received.append,
        registration_key="k",
        series=(BLSSeriesSpec(series_id="CPIAUCSL"),),
        clock_ns=_FakeClock(),
        post=fake_post,
        poll_interval_s=0.001,
        reconnect_delay_s=0.001,
    )

    async def _scenario() -> None:
        stop = asyncio.Event()

        async def _stopper() -> None:
            while pump.status().errors < 1:
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())

    asyncio.run(_scenario())

    assert received == []
    s = pump.status()
    assert s.polls >= 1
    assert s.errors >= 1


def test_pump_status_failure_payload_emits_no_observations() -> None:
    received: list[MacroObservation] = []

    async def fake_post(url: str, body: str) -> bytes:
        return _payload(status="REQUEST_NOT_PROCESSED")

    pump = BLSHTTPPump(
        sink=received.append,
        registration_key="k",
        series=(BLSSeriesSpec(series_id="X"),),
        clock_ns=_FakeClock(),
        post=fake_post,
        poll_interval_s=0.001,
    )

    async def _scenario() -> None:
        stop = asyncio.Event()

        async def _stopper() -> None:
            while pump.status().polls < 1:
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())

    asyncio.run(_scenario())

    assert received == []
    s = pump.status()
    assert s.polls >= 1
    assert s.observations_received == 0
    # status="REQUEST_NOT_PROCESSED" is a parse-level no-op (the
    # parser returns ()) — the pump treats it as a successful zero-row
    # cycle, NOT a transport error.
    assert s.errors == 0


def test_pump_sink_exception_increments_errors_but_continues() -> None:
    seen_ids: list[str] = []

    def flaky_sink(obs: MacroObservation) -> None:
        seen_ids.append(obs.series_id)
        if obs.series_id == "CPIAUCSL":
            raise RuntimeError("sink boom")

    async def fake_post(url: str, body: str) -> bytes:
        return _payload(
            series=[
                _series_block(
                    "CPIAUCSL",
                    [{"year": "2024", "period": "M01", "value": "1.0"}],
                ),
                _series_block(
                    "UNRATE",
                    [{"year": "2024", "period": "M01", "value": "3.7"}],
                ),
            ],
        )

    pump = BLSHTTPPump(
        sink=flaky_sink,
        registration_key="k",
        series=(
            BLSSeriesSpec(series_id="CPIAUCSL"),
            BLSSeriesSpec(series_id="UNRATE"),
        ),
        clock_ns=_FakeClock(),
        post=fake_post,
        poll_interval_s=0.001,
    )

    async def _scenario() -> None:
        stop = asyncio.Event()

        async def _stopper() -> None:
            while pump.status().polls < 1:
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())

    asyncio.run(_scenario())

    assert "CPIAUCSL" in seen_ids and "UNRATE" in seen_ids
    s = pump.status()
    assert s.polls >= 1
    # CPIAUCSL raised on the sink-side — must NOT be counted as
    # observations_received; UNRATE must be.
    assert s.observations_received >= 1
    assert s.errors >= 1


# ---------------------------------------------------------------------------
# stop behavior
# ---------------------------------------------------------------------------


def test_pump_stops_promptly_when_stop_is_set_during_sleep() -> None:
    async def fake_post(url: str, body: str) -> bytes:
        return _payload(
            series=[
                _series_block(
                    "X",
                    [{"year": "2024", "period": "M01", "value": "1.0"}],
                ),
            ],
        )

    pump = BLSHTTPPump(
        sink=lambda _o: None,
        registration_key="k",
        series=(BLSSeriesSpec(series_id="X"),),
        clock_ns=_FakeClock(),
        post=fake_post,
        # Long interval — the test would hang if the pump didn't honor stop.
        poll_interval_s=60.0,
    )

    async def _scenario() -> None:
        stop = asyncio.Event()

        async def _stopper() -> None:
            while pump.status().polls < 1:
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())

    asyncio.run(_scenario())
    assert pump.status().running is False


# ---------------------------------------------------------------------------
# determinism / idempotence
# ---------------------------------------------------------------------------


def test_pump_two_cycles_produce_same_observations_per_cycle() -> None:
    received: list[MacroObservation] = []

    async def fake_post(url: str, body: str) -> bytes:
        return _payload(
            series=[
                _series_block(
                    "CPIAUCSL",
                    [{"year": "2024", "period": "M01", "value": "319.086"}],
                ),
            ],
        )

    pump = BLSHTTPPump(
        sink=received.append,
        registration_key="k",
        series=(BLSSeriesSpec(series_id="CPIAUCSL"),),
        clock_ns=_FakeClock(),
        post=fake_post,
        poll_interval_s=0.001,
    )

    async def _scenario() -> None:
        stop = asyncio.Event()

        async def _stopper() -> None:
            while pump.status().polls < 2:
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())

    asyncio.run(_scenario())

    # At least two cycles each emitted one observation (the stopper waits
    # for polls >= 2 then sets stop; a third partial cycle is possible
    # but never required). Same payload twice => observation_date /
    # value / source / series_id all identical between cycles.
    assert len(received) >= 2
    a, b = received[0], received[1]
    assert a.observation_date == b.observation_date
    assert a.value == b.value
    assert a.series_id == b.series_id
    assert a.source == b.source
    # ts_ns advances every clock_ns() call.
    assert b.ts_ns > a.ts_ns
