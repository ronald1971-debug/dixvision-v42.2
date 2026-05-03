"""Read-only Pump.fun launch firehose adapter (SRC-LAUNCH-PUMPFUN-001).

Pump.fun is the dominant Solana memecoin-launch venue. Its public
PumpPortal WebSocket gateway exposes new-token mint events without
requiring an API key:

* ``wss://pumpportal.fun/api/data`` accepts a JSON subscription frame
  ``{"method": "subscribeNewToken"}`` and then streams one JSON object
  per new mint.

This module follows the same layered split as
:mod:`ui.feeds.binance_public_ws`:

* :func:`parse_new_token` — pure frame → :class:`LaunchEvent`
  projection. Returns ``None`` for non-data frames (subscription
  acks, keepalives, malformed payloads) so the pump loop can skip
  them without raising.
* :class:`PumpFunLaunchPump` — thin async I/O wrapper with reconnect
  + exponential backoff. Takes a ``connect`` callable and a
  ``clock_ns`` callable so tests inject fakes (no real network, no
  raw clock calls) and the determinism boundary stays explicit
  (INV-15).

Refs:
  https://pumpportal.fun/data-api/real-time/
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
)
from dataclasses import dataclass
from typing import Any, Protocol

from core.contracts.launches import LaunchEvent

LOG = logging.getLogger(__name__)

#: Canonical PumpPortal public WS endpoint.
PUMPPORTAL_WS_URL = "wss://pumpportal.fun/api/data"

#: Subscription frame the gateway expects to flip on new-token streaming.
SUBSCRIBE_NEW_TOKEN_FRAME: str = json.dumps(
    {"method": "subscribeNewToken"}, separators=(",", ":")
)

#: Default reconnect backoff floor + ceiling (seconds).
DEFAULT_RECONNECT_DELAY_S = 5.0
DEFAULT_RECONNECT_DELAY_MAX_S = 60.0

#: Venue tag stamped onto every emitted ``LaunchEvent``.
VENUE_TAG = "PUMPFUN"

#: Chain tag.
CHAIN_TAG = "solana"


def parse_new_token(
    payload: Mapping[str, Any] | Any,
    *,
    ts_ns: int,
    venue: str = VENUE_TAG,
    chain: str = CHAIN_TAG,
) -> LaunchEvent | None:
    """Project one PumpPortal ``newToken`` frame into a :class:`LaunchEvent`.

    Returns ``None`` (never raises) if ``payload`` does not look like a
    new-token announcement — this lets the pump silently skip
    subscription acks (``{"message": "Subscribed"}``), keepalives, and
    malformed JSON without aborting the connection.

    INV-15 (pure projection): ``ts_ns`` is supplied by the caller, never
    derived from ``payload`` or a system clock, so two replays with the
    same input produce byte-identical output.
    """
    if not isinstance(payload, Mapping):
        return None
    # PumpPortal flags new-token announcements with ``txType == "create"``.
    if payload.get("txType") not in ("create", "newToken", "new_token"):
        return None
    mint = payload.get("mint") or payload.get("address") or ""
    if not isinstance(mint, str) or not mint:
        return None
    symbol = str(payload.get("symbol") or payload.get("ticker") or "")
    name = str(payload.get("name") or "")
    creator = str(
        payload.get("traderPublicKey")
        or payload.get("creator")
        or payload.get("dev")
        or ""
    )
    market_cap_usd = _to_float(
        payload.get("marketCapSol")
        or payload.get("marketCapUsd")
        or payload.get("usdMarketCap")
    )
    liquidity_usd = _to_float(
        payload.get("vSolInBondingCurve") or payload.get("liquidityUsd")
    )
    return LaunchEvent(
        ts_ns=ts_ns,
        chain=chain,
        venue=venue,
        mint=mint,
        symbol=symbol,
        name=name,
        creator=creator,
        market_cap_usd=market_cap_usd,
        liquidity_usd=liquidity_usd,
    )


def _to_float(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


class _WSConnection(Protocol):
    """Minimal subset of ``websockets.WebSocketClientProtocol`` we use.

    Declared as a :class:`typing.Protocol` so tests inject a fake
    connection without importing ``websockets``. The pump needs:
    async iteration (yielding text frames), ``send()`` (subscription
    frame), and ``close()``.
    """

    async def send(self, message: str) -> None:  # pragma: no cover - protocol
        ...

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
class PumpFunStatus:
    """Snapshot of pump health — exposed by ``GET /api/feeds/pumpfun/status``."""

    running: bool
    url: str
    last_launch_ts_ns: int | None
    launches_received: int
    errors: int


class PumpFunLaunchPump:
    """Async pump streaming PumpPortal new-token frames into a sink.

    The sink callable runs synchronously inside the asyncio loop; for
    cross-thread state mutation use :class:`ui.feeds.runner.FeedRunner`-
    style wrappers (see ``ui/server.py`` startup).
    """

    def __init__(
        self,
        sink: Callable[[LaunchEvent], None],
        *,
        clock_ns: Callable[[], int],
        connect: WSConnect | None = None,
        url: str = PUMPPORTAL_WS_URL,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        reconnect_delay_max_s: float = DEFAULT_RECONNECT_DELAY_MAX_S,
        venue: str = VENUE_TAG,
        chain: str = CHAIN_TAG,
    ) -> None:
        if not url:
            raise ValueError("PumpFunLaunchPump: url required")
        if reconnect_delay_s <= 0:
            raise ValueError(
                "PumpFunLaunchPump: reconnect_delay_s must be positive"
            )
        if reconnect_delay_max_s < reconnect_delay_s:
            raise ValueError(
                "PumpFunLaunchPump: reconnect_delay_max_s must be >= "
                "reconnect_delay_s"
            )
        self._sink = sink
        self._clock_ns = clock_ns
        self._connect: WSConnect = (
            connect if connect is not None else _default_connect
        )
        self._url = url
        self._reconnect_delay_s = reconnect_delay_s
        self._reconnect_delay_max_s = reconnect_delay_max_s
        self._venue = venue
        self._chain = chain
        self._stop_event = asyncio.Event()
        self._launches_received = 0
        self._errors = 0
        self._last_launch_ts_ns: int | None = None
        self._running = False

    @property
    def url(self) -> str:
        return self._url

    def status(self) -> PumpFunStatus:
        return PumpFunStatus(
            running=self._running,
            url=self._url,
            last_launch_ts_ns=self._last_launch_ts_ns,
            launches_received=self._launches_received,
            errors=self._errors,
        )

    def stop(self) -> None:
        """Signal the run loop to exit on its next iteration."""
        self._stop_event.set()

    async def run(self) -> None:
        """Connect → subscribe → consume → reconnect on error until ``stop()``.

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
                    await conn.send(SUBSCRIBE_NEW_TOKEN_FRAME)
                    LOG.info(
                        "pumpfun_ws: connected url=%s venue=%s",
                        self._url,
                        self._venue,
                    )
                    delay = self._reconnect_delay_s
                    async for raw in conn:  # type: ignore[union-attr]
                        if self._stop_event.is_set():
                            break
                        self._handle_frame(raw)
                except Exception:  # noqa: BLE001 - log + reconnect
                    self._errors += 1
                    LOG.exception(
                        "pumpfun_ws: connection failure; "
                        "reconnect in %.1fs",
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
                delay = min(delay * 2.0, self._reconnect_delay_max_s)
        finally:
            self._running = False

    def _handle_frame(self, raw: str | bytes) -> None:
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._errors += 1
            return
        ts_ns = self._clock_ns()
        event = parse_new_token(
            payload, ts_ns=ts_ns, venue=self._venue, chain=self._chain
        )
        if event is None:
            return
        try:
            self._sink(event)
        except Exception:  # noqa: BLE001 - sink must not kill the pump
            self._errors += 1
            LOG.exception("pumpfun_ws: sink raised on event=%r", event)
            return
        self._launches_received += 1
        self._last_launch_ts_ns = ts_ns


__all__ = [
    "CHAIN_TAG",
    "DEFAULT_RECONNECT_DELAY_MAX_S",
    "DEFAULT_RECONNECT_DELAY_S",
    "PUMPPORTAL_WS_URL",
    "PumpFunLaunchPump",
    "PumpFunStatus",
    "SUBSCRIBE_NEW_TOKEN_FRAME",
    "VENUE_TAG",
    "WSConnect",
    "parse_new_token",
]
