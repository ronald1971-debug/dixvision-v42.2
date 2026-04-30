"""Tests for the async FRED HTTP pump.

These tests exercise :class:`ui.feeds.fred_http.FredHTTPPump` against
fakes — a stub HTTP fetcher and a deterministic clock — so they run
without network and without ``time.time``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from core.contracts.macro import MacroObservation
from ui.feeds.fred_http import (
    SOURCE_TAG,
    FredHTTPPump,
    FredSeriesSpec,
    MacroFeedStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DGS10_PAYLOAD = b"""{
  "observations": [
    {"date": "2026-04-18", "value": "4.06"},
    {"date": "2026-04-19", "value": "4.12"}
  ]
}"""

_CPI_PAYLOAD = b"""{
  "observations": [
    {"date": "2026-04-01", "value": "311.5"}
  ]
}"""


def _make_clock(start: int = 1) -> Callable[[], int]:
    counter = {"n": start}

    def _clock() -> int:
        v = counter["n"]
        counter["n"] += 1
        return v

    return _clock


def _payload_for(url: str) -> bytes:
    if "DGS10" in url:
        return _DGS10_PAYLOAD
    if "CPIAUCSL" in url:
        return _CPI_PAYLOAD
    raise AssertionError(f"unexpected URL in test: {url}")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_pump_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        FredHTTPPump(
            sink=lambda obs: None,
            api_key="",
            series=[FredSeriesSpec("DGS10")],
            clock_ns=lambda: 1,
        )
    with pytest.raises(ValueError):
        FredHTTPPump(
            sink=lambda obs: None,
            api_key="k",
            series=[],
            clock_ns=lambda: 1,
        )
    with pytest.raises(ValueError):
        FredHTTPPump(
            sink=lambda obs: None,
            api_key="k",
            series=[FredSeriesSpec("DGS10")],
            clock_ns=lambda: 1,
            poll_interval_s=0,
        )
    with pytest.raises(ValueError):
        FredHTTPPump(
            sink=lambda obs: None,
            api_key="k",
            series=[FredSeriesSpec("DGS10")],
            clock_ns=lambda: 1,
            reconnect_delay_s=0,
        )
    with pytest.raises(ValueError):
        FredHTTPPump(
            sink=lambda obs: None,
            api_key="k",
            series=[FredSeriesSpec("DGS10")],
            clock_ns=lambda: 1,
            reconnect_delay_s=10.0,
            reconnect_delay_max_s=5.0,
        )
    with pytest.raises(ValueError):
        FredHTTPPump(
            sink=lambda obs: None,
            api_key="k",
            series=[FredSeriesSpec("DGS10")],
            clock_ns=lambda: 1,
            source="",
        )


def test_series_spec_validation() -> None:
    with pytest.raises(ValueError):
        FredSeriesSpec("")
    with pytest.raises(ValueError):
        FredSeriesSpec("DGS10", limit=0)
    with pytest.raises(ValueError):
        FredSeriesSpec("DGS10", limit=-3)


def test_pump_dedupes_series_in_order() -> None:
    pump = FredHTTPPump(
        sink=lambda obs: None,
        api_key="k",
        series=[
            FredSeriesSpec("DGS10"),
            FredSeriesSpec("CPIAUCSL"),
            FredSeriesSpec("DGS10"),  # dup, should be dropped
        ],
        clock_ns=lambda: 1,
    )
    assert pump.series_ids == ("DGS10", "CPIAUCSL")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_pump_polls_each_series_once_per_cycle_and_emits_observations() -> None:
    received: list[MacroObservation] = []
    fetched_urls: list[str] = []

    async def _fetch(url: str) -> bytes:
        fetched_urls.append(url)
        return _payload_for(url)

    async def _scenario() -> MacroFeedStatus:
        stop = asyncio.Event()
        pump = FredHTTPPump(
            sink=received.append,
            api_key="k",
            series=[FredSeriesSpec("DGS10"), FredSeriesSpec("CPIAUCSL")],
            clock_ns=_make_clock(),
            fetch=_fetch,
            poll_interval_s=0.001,
            reconnect_delay_s=0.001,
        )

        async def _stopper() -> None:
            # Allow at least one full poll cycle (both series), then stop.
            for _ in range(50):
                if pump.status().polls >= 1 and len(received) >= 3:
                    break
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())
        return pump.status()

    status = asyncio.run(_scenario())
    assert status.observations_received >= 3
    assert status.polls >= 1
    assert status.errors == 0
    assert status.source == SOURCE_TAG
    assert status.series_ids == ("DGS10", "CPIAUCSL")
    assert any("DGS10" in u for u in fetched_urls)
    assert any("CPIAUCSL" in u for u in fetched_urls)
    assert any(o.series_id == "DGS10" for o in received)
    assert any(o.series_id == "CPIAUCSL" for o in received)


# ---------------------------------------------------------------------------
# Backoff path
# ---------------------------------------------------------------------------


def test_pump_increments_errors_and_recovers_after_failure() -> None:
    received: list[MacroObservation] = []
    calls: dict[str, int] = {"n": 0}

    async def _fetch(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated fetch boom")
        return _payload_for(url)

    async def _scenario() -> MacroFeedStatus:
        stop = asyncio.Event()
        pump = FredHTTPPump(
            sink=received.append,
            api_key="k",
            series=[FredSeriesSpec("DGS10")],
            clock_ns=_make_clock(),
            fetch=_fetch,
            poll_interval_s=0.001,
            reconnect_delay_s=0.001,
        )

        async def _stopper() -> None:
            for _ in range(200):
                # Keep going until we observed at least one error AND
                # at least one good poll.
                if (
                    pump.status().errors >= 1
                    and pump.status().observations_received >= 1
                ):
                    break
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())
        return pump.status()

    status = asyncio.run(_scenario())
    assert status.errors >= 1
    assert status.observations_received >= 1


def test_pump_per_series_failure_does_not_reraise_when_other_succeeds() -> None:
    received: list[MacroObservation] = []

    async def _fetch(url: str) -> bytes:
        if "DGS10" in url:
            raise RuntimeError("DGS10 down")
        return _payload_for(url)

    async def _scenario() -> MacroFeedStatus:
        stop = asyncio.Event()
        pump = FredHTTPPump(
            sink=received.append,
            api_key="k",
            series=[FredSeriesSpec("DGS10"), FredSeriesSpec("CPIAUCSL")],
            clock_ns=_make_clock(),
            fetch=_fetch,
            poll_interval_s=0.001,
            reconnect_delay_s=0.001,
        )

        async def _stopper() -> None:
            for _ in range(200):
                if pump.status().observations_received >= 1:
                    break
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())
        return pump.status()

    status = asyncio.run(_scenario())
    assert status.observations_received >= 1
    # DGS10 failed every cycle, but the cycle shouldn't have triggered
    # a reconnect-delay backoff because CPIAUCSL succeeded — observable
    # via the per-series error counter.
    assert status.errors >= 1
    assert any(o.series_id == "CPIAUCSL" for o in received)


# ---------------------------------------------------------------------------
# Sink-failure path — pump must continue rather than crash
# ---------------------------------------------------------------------------


def test_pump_isolates_sink_exceptions() -> None:
    sink_calls = {"n": 0}

    def _bad_sink(obs: MacroObservation) -> None:
        sink_calls["n"] += 1
        if sink_calls["n"] == 1:
            raise RuntimeError("sink boom")

    async def _fetch(url: str) -> bytes:
        return _payload_for(url)

    async def _scenario() -> MacroFeedStatus:
        stop = asyncio.Event()
        pump = FredHTTPPump(
            sink=_bad_sink,
            api_key="k",
            series=[FredSeriesSpec("DGS10")],
            clock_ns=_make_clock(),
            fetch=_fetch,
            poll_interval_s=0.001,
            reconnect_delay_s=0.001,
        )

        async def _stopper() -> None:
            for _ in range(200):
                if (
                    pump.status().errors >= 1
                    and pump.status().observations_received >= 1
                ):
                    break
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.gather(pump.run(stop), _stopper())
        return pump.status()

    status = asyncio.run(_scenario())
    # First sink call raised; pump kept going, second observation
    # delivered, errors >= 1, observations_received >= 1.
    assert status.errors >= 1
    assert status.observations_received >= 1


# ---------------------------------------------------------------------------
# Status snapshot is frozen + sane on a fresh pump
# ---------------------------------------------------------------------------


def test_status_on_fresh_pump_is_idle() -> None:
    pump = FredHTTPPump(
        sink=lambda obs: None,
        api_key="k",
        series=[FredSeriesSpec("DGS10")],
        clock_ns=_make_clock(),
    )
    s = pump.status()
    assert s.running is False
    assert s.polls == 0
    assert s.errors == 0
    assert s.observations_received == 0
    assert s.last_poll_ts_ns is None
    assert s.last_observation_ts_ns is None
    assert s.series_ids == ("DGS10",)
    # frozen + slotted contract
    with pytest.raises(AttributeError):
        s.polls = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Stop event short-circuits the regular cadence
# ---------------------------------------------------------------------------


def test_pump_stops_promptly_when_stop_set() -> None:
    received: list[MacroObservation] = []

    async def _fetch(url: str) -> bytes:
        return _payload_for(url)

    async def _scenario() -> MacroFeedStatus:
        stop = asyncio.Event()
        pump = FredHTTPPump(
            sink=received.append,
            api_key="k",
            series=[FredSeriesSpec("DGS10")],
            clock_ns=_make_clock(),
            fetch=_fetch,
            poll_interval_s=60.0,  # long cadence — must be cut short
            reconnect_delay_s=0.001,
        )

        async def _stopper() -> None:
            # Wait until the first poll completed, then stop.
            for _ in range(200):
                if pump.status().polls >= 1:
                    break
                await asyncio.sleep(0.001)
            stop.set()

        await asyncio.wait_for(
            asyncio.gather(pump.run(stop), _stopper()), timeout=5.0
        )
        return pump.status()

    status = asyncio.run(_scenario())
    assert status.polls >= 1
    assert status.running is False
