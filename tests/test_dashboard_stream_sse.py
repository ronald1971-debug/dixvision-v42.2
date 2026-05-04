"""DASH-LIVE-01 — ``/api/dashboard/stream`` SSE endpoint regression tests.

The dashboard's ``realtime.ts`` bridge opens an ``EventSource`` on this
URL and dispatches every ``StreamEvent {channel, ts_iso, payload}`` to
its per-widget listener bus. Before DASH-LIVE-01 the endpoint did not
exist, ``EventSource`` immediately errored, the bridge fell back to the
deterministic mock generator, and the AUDIT-P1.4 amber banner stayed
on permanently. These tests pin the wiring so a regression cannot
silently re-introduce the permanent-mock state.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

import ui.server as ui_server
from ui.server import (
    _sse_channel_for,
    _sse_format,
    _sse_ts_iso_for,
)


def _client() -> TestClient:
    # Re-resolve ``app`` on every call: other tests in the suite reload
    # ``ui.server`` which leaves the module-level ``app`` we imported at
    # collection time pointing at a stale FastAPI instance whose route
    # handlers close over a stale ``STATE``. ``ui_server.app`` always
    # resolves to the live instance.
    return TestClient(ui_server.app)


def _state():
    return ui_server.STATE


def _drain_backfill(client: TestClient) -> bytes:
    """Drain the SSE stream in ``?backfill_only=1`` mode.

    The endpoint emits the current event-queue snapshot then closes the
    connection, so a plain GET reads the entire response body without
    blocking. The production dashboard never sets this flag — it is the
    regression-suite + diagnostic-client mode."""

    resp = client.get("/api/dashboard/stream", params={"backfill_only": "1"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    return resp.content


# ---------------------------------------------------------------------------
# Channel mapping
# ---------------------------------------------------------------------------


def test_channel_alias_market_tick() -> None:
    assert _sse_channel_for({"kind": "MARKET_TICK"}) == "ticks"
    assert _sse_channel_for({"kind": "market_tick"}) == "ticks"


def test_channel_alias_news_substring() -> None:
    assert _sse_channel_for({"kind": "NEWS_ITEM"}) == "news"
    # Substring match for engines that emit ``NEWS_SHOCK`` etc.
    assert _sse_channel_for({"kind": "NEWS_SHOCK"}) == "news"


def test_channel_alias_hazard_prefix() -> None:
    assert _sse_channel_for({"kind": "HAZARD"}) == "hazards"
    assert _sse_channel_for({"kind": "HAZ_LATENCY"}) == "hazards"


def test_channel_alias_unknown_falls_through_to_lowercase_kind() -> None:
    assert _sse_channel_for({"kind": "CUSTOM_KIND"}) == "custom_kind"


def test_channel_alias_missing_kind_defaults_to_event() -> None:
    assert _sse_channel_for({}) == "event"
    assert _sse_channel_for({"kind": ""}) == "event"


# ---------------------------------------------------------------------------
# Timestamp derivation
# ---------------------------------------------------------------------------


def test_ts_iso_derived_from_ts_ns_when_present() -> None:
    iso = _sse_ts_iso_for({"ts_ns": 1_700_000_000_000_000_000})
    # 2023-11-14T22:13:20+00:00
    assert iso.startswith("2023-11-14T22:13:20")


def test_ts_iso_falls_back_to_now_when_missing() -> None:
    iso = _sse_ts_iso_for({})
    # Smoke: still ISO-8601 with explicit UTC offset.
    assert "T" in iso
    assert iso.endswith("+00:00") or iso.endswith("Z")


def test_ts_iso_invalid_ts_ns_falls_back() -> None:
    iso = _sse_ts_iso_for({"ts_ns": "not-a-number"})
    assert "T" in iso


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------


def test_sse_format_emits_canonical_data_line() -> None:
    framed = _sse_format(
        {"channel": "ticks", "ts_iso": "2024-01-01T00:00:00+00:00", "payload": {}}
    )
    assert framed.startswith("data: ")
    assert framed.endswith("\n\n")
    body = framed[len("data: ") : -2]
    parsed = json.loads(body)
    assert parsed["channel"] == "ticks"


# ---------------------------------------------------------------------------
# End-to-end via FastAPI TestClient
# ---------------------------------------------------------------------------


def test_endpoint_emits_text_event_stream_content_type() -> None:
    with _client() as client:
        body = _drain_backfill(client)
    assert b": connected" in body


def test_endpoint_streams_recorded_market_tick() -> None:
    """Posting a tick must surface as a ``ticks`` channel SSE frame.

    This is the regression that was missing before DASH-LIVE-01 — the
    realtime.ts bridge had nowhere to attach so it always fell back to
    the deterministic mock generator.
    """

    with _client() as client:
        with _state().lock:
            baseline_seq = _state().event_seq

        resp = client.post(
            "/api/tick",
            json={
                "symbol": "BTCUSDT",
                "bid": 100.0,
                "ask": 100.1,
                "last": 100.05,
                "ts_ns": 1_700_000_000_000_000_000,
                "venue": "binance",
            },
        )
        assert resp.status_code == 200, resp.text

        with _state().lock:
            assert _state().event_seq > baseline_seq

        body = _drain_backfill(client)

    assert b"data: " in body
    frames = [
        json.loads(line[len("data: ") :])
        for line in body.split(b"\n\n")
        if line.startswith(b"data: ")
    ]
    channels = {ev["channel"] for ev in frames}
    assert "ticks" in channels, channels
    tick_frame = next(ev for ev in frames if ev["channel"] == "ticks")
    assert tick_frame["payload"]["symbol"] == "BTCUSDT"
    assert tick_frame["ts_iso"].startswith("2023-11-14T22:13:20")


def test_endpoint_skips_records_without_seq() -> None:
    """``_sse_event_stream`` must filter records that do not carry a
    monotone ``seq``. Anything else would let infra-only rows confuse
    the last-seq tracking and cause replays of historical events."""

    with _client() as client:
        with _state().lock:
            _state().events.appendleft({"kind": "INFRA_ROW_NO_SEQ"})

        body = _drain_backfill(client)

    frames = [
        json.loads(line[len("data: ") :])
        for line in body.split(b"\n\n")
        if line.startswith(b"data: ")
    ]
    kinds = {ev["payload"].get("kind") for ev in frames}
    assert "INFRA_ROW_NO_SEQ" not in kinds


def test_endpoint_returns_empty_payload_when_event_queue_is_empty() -> None:
    """A fresh harness with no engine activity must still emit the
    initial ``: connected`` comment so the dashboard's ``onopen``
    handler fires and the AUDIT-P1.4 banner clears to ``"live"`` —
    even before any tick has been recorded."""

    with _client() as client:
        with _state().lock:
            _state().events.clear()

        body = _drain_backfill(client)

    assert body.startswith(b": connected\n\n")
