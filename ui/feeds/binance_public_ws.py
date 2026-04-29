"""Read-only Binance public WebSocket adapter (SRC-MARKET-BINANCE-001).

Streams the public 24-hour rolling ticker for one or more spot symbols
(no API key, no auth, no order routing) and converts each frame into a
canonical :class:`core.contracts.market.MarketTick`.

Layered split:

* :func:`make_combined_stream_url` — pure URL builder.
* :func:`parse_24hr_ticker` — pure frame → ``MarketTick`` projection.
  Returns ``None`` for non-data frames (subscription acks, heartbeats,
  malformed payloads) so the pump loop can skip them without raising.
* :class:`BinancePublicWSPump` — thin async I/O wrapper with reconnect
  + exponential backoff. Takes a ``connect`` callable and a
  ``clock_ns`` callable so tests inject fakes (no real network, no
  ``time.time``) and the determinism boundary stays explicit.
* :class:`FeedStatus` — frozen telemetry snapshot exposed by the HTTP
  status endpoint.

INV-15: the parser is pure (caller-supplied ``ts_ns``); the pump is
non-deterministic by design (network), but every event it emits is
funneled into the harness via the same code path as ``POST /api/tick``,
which the engine ledger replays deterministically.

Binance public 24hrTicker payload reference:
https://binance-docs.github.io/apidocs/spot/en/#individual-symbol-ticker-streams
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Protocol

from core.contracts.market import MarketTick

LOG = logging.getLogger(__name__)

#: Canonical public WS host (no auth required).
BINANCE_PUBLIC_WS_BASE = "wss://stream.binance.com:9443"

#: Default reconnect backoff floor + ceiling (seconds).
DEFAULT_RECONNECT_DELAY_S = 5.0
DEFAULT_RECONNECT_DELAY_MAX_S = 60.0

#: Default symbol set used when the operator hits ``start`` with no body.
#: Lower-case here because the URL builder lowercases anyway, but kept
#: explicit so the registry/SCVS audit reads cleanly.
DEFAULT_SYMBOLS: tuple[str, ...] = ("btcusdt", "ethusdt")

#: Venue tag stamped onto every emitted ``MarketTick`` so the ledger
#: distinguishes Binance ticks from manual ``POST /api/tick`` ticks
#: (``venue=""`` by default in the ``TickIn`` model).
VENUE_TAG = "BINANCE"


def make_combined_stream_url(
    symbols: Sequence[str], *, stream: str = "ticker"
) -> str:
    """Build the combined-stream URL for the given symbols.

    Returns ``"{BASE}/stream?streams=btcusdt@ticker/ethusdt@ticker"``
    for ``["BTCUSDT", "ETHUSDT"]`` (case-insensitive). Raises
    :class:`ValueError` if the symbol list is empty or any symbol is
    not alphanumeric (Binance only accepts ``[A-Za-z0-9]+``).
    """
    if not symbols:
        raise ValueError(
            "make_combined_stream_url: at least one symbol required"
        )
    cleaned: list[str] = []
    for raw in symbols:
        s = raw.lower().strip()
        if not s or not s.isalnum():
            raise ValueError(
                f"make_combined_stream_url: invalid symbol {raw!r}"
            )
        cleaned.append(s)
    parts = "/".join(f"{s}@{stream}" for s in cleaned)
    return f"{BINANCE_PUBLIC_WS_BASE}/stream?streams={parts}"


def parse_24hr_ticker(
    payload: Mapping[str, Any] | Any,
    *,
    ts_ns: int,
    venue: str = VENUE_TAG,
) -> MarketTick | None:
    """Project one Binance ``24hrTicker`` frame into a :class:`MarketTick`.

    Returns ``None`` (never raises) if ``payload`` is not a recognisable
    24hrTicker frame — this lets the pump silently skip subscription
    acks, control messages, and malformed JSON without aborting the
    connection.

    INV-15 (pure projection): ``ts_ns`` is supplied by the caller, never
    derived from ``payload`` or a system clock, so two replays with the
    same input produce byte-identical output.
    """
    if not isinstance(payload, Mapping):
        return None
    # Combined-stream wrapper: ``{"stream": "...", "data": {...}}``
    inner = payload.get("data")
    data: Mapping[str, Any]
    if isinstance(inner, Mapping):
        data = inner
    else:
        data = payload
    if data.get("e") != "24hrTicker":
        return None
    try:
        symbol = str(data["s"])
        bid = float(data["b"])
        ask = float(data["a"])
        last = float(data["c"])
    except (KeyError, ValueError, TypeError):
        return None
    raw_volume = data.get("v", 0.0)
    try:
        volume = float(raw_volume)
    except (ValueError, TypeError):
        volume = 0.0
    if not symbol:
        return None
    if bid <= 0 or ask <= 0 or last <= 0:
        return None
    if volume < 0:
        return None
    return MarketTick(
        ts_ns=ts_ns,
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        volume=volume,
        venue=venue,
    )


class _WSConnection(Protocol):
    """Minimal subset of ``websockets.WebSocketClientProtocol`` we use.

    Declared as a :class:`typing.Protocol` so tests inject a fake
    connection without importing ``websockets``. The pump only needs
    async iteration (yielding text frames) and ``close()``.
    """

    def __aiter__(self) -> AsyncIterable[str]:  # pragma: no cover - protocol
        ...

    async def close(self) -> None:  # pragma: no cover - protocol
        ...


WSConnect = Callable[[str], Awaitable[_WSConnection]]
"""Async URL → connection callable. Production uses ``websockets.connect``."""


async def _default_connect(url: str) -> _WSConnection:
    """Default connector — imported lazily so the parser is usable
    without ``websockets`` installed (tests, lint, etc.)."""

    import websockets  # local import; heavy dependency
    return await websockets.connect(url)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class FeedStatus:
    """Snapshot of pump health — exposed by ``GET /api/feeds/binance/status``."""

    running: bool
    symbols: tuple[str, ...]
    url: str
    last_tick_ts_ns: int | None
    ticks_received: int
    errors: int


class BinancePublicWSPump:
    """Async pump streaming Binance public 24hrTicker frames into a sink.

    The sink callable runs synchronously inside the asyncio loop; for
    cross-thread state mutation use :class:`ui.feeds.runner.FeedRunner`
    which wraps the pump and bridges to the FastAPI sync world.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        sink: Callable[[MarketTick], None],
        *,
        clock_ns: Callable[[], int],
        connect: WSConnect | None = None,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        reconnect_delay_max_s: float = DEFAULT_RECONNECT_DELAY_MAX_S,
        venue: str = VENUE_TAG,
    ) -> None:
        if not symbols:
            raise ValueError(
                "BinancePublicWSPump: at least one symbol required"
            )
        if reconnect_delay_s <= 0:
            raise ValueError(
                "BinancePublicWSPump: reconnect_delay_s must be positive"
            )
        if reconnect_delay_max_s < reconnect_delay_s:
            raise ValueError(
                "BinancePublicWSPump: reconnect_delay_max_s must be >= "
                "reconnect_delay_s"
            )
        self._symbols: tuple[str, ...] = tuple(s.upper() for s in symbols)
        self._sink = sink
        self._clock_ns = clock_ns
        self._connect: WSConnect = connect if connect is not None else _default_connect
        self._reconnect_delay_s = reconnect_delay_s
        self._reconnect_delay_max_s = reconnect_delay_max_s
        self._venue = venue
        self._url = make_combined_stream_url(symbols)
        self._stop_event = asyncio.Event()
        self._ticks_received = 0
        self._errors = 0
        self._last_tick_ts_ns: int | None = None
        self._running = False

    @property
    def url(self) -> str:
        return self._url

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def status(self) -> FeedStatus:
        return FeedStatus(
            running=self._running,
            symbols=self._symbols,
            url=self._url,
            last_tick_ts_ns=self._last_tick_ts_ns,
            ticks_received=self._ticks_received,
            errors=self._errors,
        )

    def stop(self) -> None:
        """Signal the run loop to exit on its next iteration.

        Safe to call from any thread that holds a reference to this
        pump's event loop (use ``loop.call_soon_threadsafe(pump.stop)``
        from outside the loop).
        """
        self._stop_event.set()

    async def run(self) -> None:
        """Connect → consume → reconnect on error until ``stop()``.

        Backs off exponentially from ``reconnect_delay_s`` up to
        ``reconnect_delay_max_s`` between reconnect attempts. The first
        successful frame resets the delay floor.
        """
        self._running = True
        delay = self._reconnect_delay_s
        try:
            while not self._stop_event.is_set():
                conn: _WSConnection | None = None
                try:
                    conn = await self._connect(self._url)
                    LOG.info(
                        "binance_public_ws: connected url=%s symbols=%s",
                        self._url,
                        ",".join(self._symbols),
                    )
                    delay = self._reconnect_delay_s
                    async for raw in conn:  # type: ignore[union-attr]
                        if self._stop_event.is_set():
                            break
                        self._handle_frame(raw)
                except Exception:  # noqa: BLE001 - log + reconnect
                    self._errors += 1
                    LOG.exception(
                        "binance_public_ws: connection failure; "
                        "reconnecting in %.1fs",
                        delay,
                    )
                finally:
                    if conn is not None:
                        try:
                            await conn.close()
                        except Exception:  # noqa: BLE001
                            pass
                if self._stop_event.is_set():
                    break
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=delay
                    )
                except TimeoutError:
                    pass
                delay = min(delay * 2, self._reconnect_delay_max_s)
        finally:
            self._running = False

    def _handle_frame(self, raw: str | bytes) -> None:
        """Decode + parse + dispatch a single WS frame.

        Bad JSON or unknown frame types increment ``errors`` (for the
        JSON case) or are silently ignored (for non-data frames like
        subscription acks). Sink failures are logged and counted but
        never abort the connection.
        """
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                self._errors += 1
                return
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            self._errors += 1
            return
        tick = parse_24hr_ticker(
            payload, ts_ns=self._clock_ns(), venue=self._venue
        )
        if tick is None:
            return
        self._ticks_received += 1
        self._last_tick_ts_ns = tick.ts_ns
        try:
            self._sink(tick)
        except Exception:  # noqa: BLE001
            self._errors += 1
            LOG.exception("binance_public_ws: sink failed")
