"""HarnessBackgroundTaskManager — SSE bridge owner (P1.3).

Extracted from ``ui.server`` as part of the P1 harness god-object
refactor. The historic SSE bridge — the channel-alias table, the
three per-record helpers, and the ``_sse_event_stream`` async
generator — lived inline in ``ui/server.py`` next to the
``@app.get("/api/dashboard/stream")`` route handler. This module
owns that machinery now; ``ui/server.py`` keeps only the route
handler (so FastAPI route discovery is unaffected) and delegates
to :class:`HarnessBackgroundTaskManager`.

This is a pure code-organisation refactor: zero behaviour change.
The previous module-level names (``_SSE_CHANNEL_ALIASES``,
``_sse_channel_for``, ``_sse_ts_iso_for``, ``_sse_format``,
``_sse_event_stream``) are re-exported by ``ui.server`` so the
existing :mod:`tests.test_dashboard_stream_sse` regression suite,
which imports them directly via
``from ui.server import _sse_channel_for, _sse_format, _sse_ts_iso_for``,
keeps working byte-stable.

INV-15 byte-identical replay, B27 / B28 / INV-71 authority
symmetry, B32 single-mutator FSM, HARDEN-04 / INV-70 freeze
policy, and B7 dashboard-prefix lint are all preserved by
construction (no new typed-event kinds, no new ledger rows, no
new env vars, no changes to the wire format of the SSE bytes).

DASH-LIVE-01 contract (verbatim from the original ``server.py``
docstring):

    ``dashboard2026/src/state/realtime.ts`` opens an
    ``EventSource`` on ``/api/dashboard/stream`` and dispatches
    each ``StreamEvent {channel, ts_iso, payload}`` to the
    per-widget listener bus. If the endpoint is unreachable the
    bridge falls back to a deterministic mock generator and the
    AUDIT-P1.4 banner stays amber.

    The stream is a thin projection over :attr:`_State.events`:
    every recorded event is mapped to a channel name (lowercased
    ``kind`` plus a stable alias table for the widgets the
    legacy mock targeted — ``ticks`` for ``MARKET_TICK``,
    ``news`` for any kind containing ``NEWS``, ``hazards`` for
    ``HazardEvent`` ``HAZ_*`` rows, etc.).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any

from fastapi import Request

from system.time_source import utc_now

SSE_CHANNEL_ALIASES: Mapping[str, str] = {
    "MARKET_TICK": "ticks",
    "SIGNAL": "signals",
    "EXECUTION": "executions",
    "HAZARD": "hazards",
    "MODE_TRANSITION": "mode",
    "OPERATOR_INTENT": "operator",
    "OPERATOR_LEARNING_OVERRIDE_CHANGED": "operator",
    "OPERATOR_SETTINGS_CHANGED": "operator",
    "NEWS_ITEM": "news",
}
"""Stable channel aliases for the widgets the legacy mock generator
targeted (``ticks`` / ``news`` / ``depth``). Anything not in this table
falls through to ``kind.lower()``; never ``""`` so the dashboard side
can always Set-bucket events by channel name."""


def sse_channel_for(record: Mapping[str, Any]) -> str:
    """Map a recorded engine event to its SSE channel name.

    The dashboard bridge buckets events by channel; the rules are
    deterministic and stable across restarts:

    1. ``kind`` is upper-cased and stripped; an empty / missing
       ``kind`` defaults to the literal ``"EVENT"``.
    2. An exact match against :data:`SSE_CHANNEL_ALIASES` wins.
    3. A substring match for ``"NEWS"`` falls through to
       ``"news"`` (covers ``NEWS_SHOCK`` etc).
    4. A ``HAZ`` prefix falls through to ``"hazards"`` (covers
       every ``HAZ_*`` row).
    5. Otherwise the lowercased ``kind`` is returned verbatim.
    """

    kind = str(record.get("kind") or "").upper().strip() or "EVENT"
    if kind in SSE_CHANNEL_ALIASES:
        return SSE_CHANNEL_ALIASES[kind]
    if "NEWS" in kind:
        return "news"
    if kind.startswith("HAZ"):
        return "hazards"
    return kind.lower()


def sse_ts_iso_for(record: Mapping[str, Any]) -> str:
    """Derive the ``ts_iso`` field for a ``StreamEvent``.

    The recorded event dict carries ``ts_ns`` for typed events (the
    canonical wall-clock timestamp under INV-15) and a few infra rows
    omit it. Fall back to ``datetime.now(UTC)`` only for the latter so
    replays of historical events keep their original timestamp.
    """

    ts_ns = record.get("ts_ns")
    if isinstance(ts_ns, int) and ts_ns > 0:
        return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC).isoformat()
    return utc_now().isoformat()


def sse_format(stream_event: Mapping[str, Any]) -> str:
    """Format one ``StreamEvent`` as a single SSE message."""

    return f"data: {json.dumps(stream_event, separators=(',', ':'))}\n\n"


class HarnessBackgroundTaskManager:
    """Owner of the harness's async background machinery.

    Currently scoped to the SSE bridge — the only true asyncio
    background coroutine in the harness. The four live feed
    runners (Binance / CoinDesk / Pump.fun / Raydium) are
    thread-based, not asyncio, and stay where they are (constructed
    by :meth:`ui.server._State._build_live_feeds`).

    The manager is constructed with a callable that returns the
    live ``_State`` so each per-connection async generator picks
    up the *current* event queue and lock, not a stale snapshot.
    This matches the original module-level pattern where
    ``_sse_event_stream`` closed over the module global ``STATE``.
    """

    __slots__ = ("_state_supplier",)

    def __init__(self, state_supplier: Any) -> None:
        # ``state_supplier`` is a zero-arg callable returning the
        # live ``_State``. ``Any`` keeps the typing surface narrow
        # without pulling ``_State`` out of TYPE_CHECKING.
        self._state_supplier = state_supplier

    async def sse_event_stream(
        self,
        request: Request,
        *,
        poll_interval_s: float = 0.25,
        keepalive_every_s: float = 15.0,
        backfill_only: bool = False,
    ) -> AsyncIterator[bytes]:
        """Async generator yielding the canonical SSE byte stream.

        The generator polls the live ``_State.events`` deque (a
        bounded buffer populated by every engine output via
        :meth:`_State.record`) and yields any rows whose ``seq``
        exceeds the last-shipped sequence number. Polling is
        bounded at ``poll_interval_s`` so a quiet harness does not
        burn CPU; long quiets emit a ``: keepalive\\n\\n`` comment
        line every ``keepalive_every_s`` so reverse proxies and
        the browser do not silently time the connection out.

        ``backfill_only`` (driven by the ``?backfill_only=1`` query
        parameter) emits the current event-queue snapshot once and
        then closes the connection. The dashboard never sets this
        flag — it is used by the regression suite and by
        lightweight diagnostic clients that want a one-shot read
        without holding an open connection.
        """

        last_seq = 0
        last_keepalive = asyncio.get_event_loop().time()
        yield b": connected\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    return
                state = self._state_supplier()
                with state.lock:
                    # ``events`` is appended-left so the newest record
                    # is at index 0; iterate oldest-first so consumers
                    # see the natural chronological order.
                    snapshot = [
                        dict(record)
                        for record in reversed(state.events)
                        if isinstance(record, Mapping)
                        and isinstance(record.get("seq"), int)
                        and record["seq"] > last_seq
                    ]
                for record in snapshot:
                    last_seq = max(last_seq, int(record.get("seq", last_seq)))
                    stream_event = {
                        "channel": sse_channel_for(record),
                        "ts_iso": sse_ts_iso_for(record),
                        "payload": record,
                    }
                    yield sse_format(stream_event).encode("utf-8")
                if backfill_only:
                    return
                now_loop = asyncio.get_event_loop().time()
                if now_loop - last_keepalive >= keepalive_every_s:
                    yield b": keepalive\n\n"
                    last_keepalive = now_loop
                await asyncio.sleep(poll_interval_s)
        except asyncio.CancelledError:  # pragma: no cover - client disconnect path
            return


__all__ = (
    "HarnessBackgroundTaskManager",
    "SSE_CHANNEL_ALIASES",
    "sse_channel_for",
    "sse_format",
    "sse_ts_iso_for",
)
