"""Read-only Raydium AMM pool snapshot adapter (SRC-POOL-RAYDIUM-001).

Raydium publishes a free, key-less REST endpoint listing every pool on
its AMM:

* ``GET https://api.raydium.io/v2/main/pairs`` → JSON array, one entry
  per pool with TVL, 24h volume, base/quote symbols, mints, etc.

This module follows the same layered split as
:mod:`ui.feeds.binance_public_ws`:

* :func:`parse_pair` — pure JSON-row → :class:`PoolSnapshot`
  projection. Returns ``None`` for malformed entries so the poll loop
  can silently skip them without raising.
* :class:`RaydiumPoolPoller` — thin async I/O wrapper that calls
  ``httpx.AsyncClient`` on a configurable interval. Takes a
  ``client_factory`` so tests inject ``httpx.MockTransport`` (no real
  network) and a ``clock_ns`` so the determinism boundary stays
  explicit (INV-15).
* :class:`RaydiumPoolStatus` — frozen telemetry exposed by the HTTP
  status endpoint.

Refs:
  https://docs.raydium.io/raydium/protocol/developers/apis-and-sdks
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from core.contracts.launches import PoolSnapshot

LOG = logging.getLogger(__name__)

#: Canonical Raydium pairs endpoint.
RAYDIUM_PAIRS_URL = "https://api.raydium.io/v2/main/pairs"

#: Default poll interval (seconds). Raydium's index updates roughly
#: every 30s; 60s keeps us well under any rate limit.
DEFAULT_POLL_INTERVAL_S = 60.0

#: Default per-request timeout.
DEFAULT_TIMEOUT_S = 10.0

#: Default reconnect / retry backoff.
DEFAULT_RETRY_DELAY_S = 5.0
DEFAULT_RETRY_DELAY_MAX_S = 60.0

#: Venue tag stamped onto every emitted ``PoolSnapshot``.
VENUE_TAG = "RAYDIUM"

#: Chain tag.
CHAIN_TAG = "solana"


def parse_pair(
    row: Mapping[str, Any] | Any,
    *,
    ts_ns: int,
    venue: str = VENUE_TAG,
    chain: str = CHAIN_TAG,
) -> PoolSnapshot | None:
    """Project one Raydium pair JSON row into a :class:`PoolSnapshot`.

    Returns ``None`` if the row is missing a pool ID — every other
    field is optional and falls back to ``""`` / ``0.0``.

    INV-15 (pure projection): ``ts_ns`` is supplied by the caller, so
    two replays with the same input produce byte-identical output.
    """
    if not isinstance(row, Mapping):
        return None
    pool_id = row.get("ammId") or row.get("pool_id") or row.get("id") or ""
    if not isinstance(pool_id, str) or not pool_id:
        return None
    base_mint = str(row.get("baseMint") or row.get("base_mint") or "")
    quote_mint = str(row.get("quoteMint") or row.get("quote_mint") or "")
    name = str(row.get("name") or "")
    base_symbol = ""
    quote_symbol = ""
    if "/" in name:
        parts = name.split("/", 1)
        base_symbol, quote_symbol = parts[0].strip(), parts[1].strip()
    base_symbol = str(row.get("baseSymbol") or base_symbol)
    quote_symbol = str(row.get("quoteSymbol") or quote_symbol)
    price = _to_float(row.get("price"))
    liquidity_usd = _to_float(
        _first_present(row, "liquidity", "tvl", "liquidityUsd")
    )
    volume_24h_usd = _to_float(
        _first_present(
            row, "volume24h", "volume_24h", "volumeUsd24h"
        )
    )
    return PoolSnapshot(
        ts_ns=ts_ns,
        chain=chain,
        venue=venue,
        pool_id=pool_id,
        base_mint=base_mint,
        quote_mint=quote_mint,
        base_symbol=base_symbol,
        quote_symbol=quote_symbol,
        price=price,
        liquidity_usd=liquidity_usd,
        volume_24h_usd=volume_24h_usd,
    )


def _first_present(
    payload: Mapping[str, Any], *keys: str
) -> Any:
    """Return the value of the first key whose value is not ``None``.

    Unlike an ``or`` chain, this preserves legitimate zero values on
    the *preferred* key (e.g. a brand-new pool with ``volume24h == 0``)
    instead of silently falling through to a fallback key.
    """
    for k in keys:
        v = payload.get(k)
        if v is not None:
            return v
    return None


def _to_float(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def parse_pairs(
    rows: Iterable[Any],
    *,
    ts_ns: int,
    venue: str = VENUE_TAG,
    chain: str = CHAIN_TAG,
) -> list[PoolSnapshot]:
    """Project a JSON array of Raydium pairs into snapshots."""
    out: list[PoolSnapshot] = []
    for row in rows:
        snap = parse_pair(row, ts_ns=ts_ns, venue=venue, chain=chain)
        if snap is not None:
            out.append(snap)
    return out


ClientFactory = Callable[[], httpx.AsyncClient]


def _default_client_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S)


@dataclass(frozen=True, slots=True)
class RaydiumPoolStatus:
    """Snapshot of poller health — exposed by ``GET /api/feeds/raydium/status``."""

    running: bool
    url: str
    last_poll_ts_ns: int | None
    snapshots_emitted: int
    errors: int


class RaydiumPoolPoller:
    """Async poller that scrapes Raydium's REST pair list on an interval.

    Each successful poll emits one :class:`PoolSnapshot` per pair in
    the response into the supplied sink. Errors increment ``errors``
    and trigger an exponential backoff up to ``retry_delay_max_s``;
    the next successful poll resets the floor.
    """

    def __init__(
        self,
        sink: Callable[[PoolSnapshot], None],
        *,
        clock_ns: Callable[[], int],
        client_factory: ClientFactory | None = None,
        url: str = RAYDIUM_PAIRS_URL,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
        retry_delay_max_s: float = DEFAULT_RETRY_DELAY_MAX_S,
        venue: str = VENUE_TAG,
        chain: str = CHAIN_TAG,
    ) -> None:
        if not url:
            raise ValueError("RaydiumPoolPoller: url required")
        if poll_interval_s <= 0:
            raise ValueError(
                "RaydiumPoolPoller: poll_interval_s must be positive"
            )
        if retry_delay_s <= 0:
            raise ValueError(
                "RaydiumPoolPoller: retry_delay_s must be positive"
            )
        if retry_delay_max_s < retry_delay_s:
            raise ValueError(
                "RaydiumPoolPoller: retry_delay_max_s must be >= "
                "retry_delay_s"
            )
        self._sink = sink
        self._clock_ns = clock_ns
        self._client_factory: ClientFactory = (
            client_factory
            if client_factory is not None
            else _default_client_factory
        )
        self._url = url
        self._poll_interval_s = poll_interval_s
        self._retry_delay_s = retry_delay_s
        self._retry_delay_max_s = retry_delay_max_s
        self._venue = venue
        self._chain = chain
        self._stop_event = asyncio.Event()
        self._snapshots_emitted = 0
        self._errors = 0
        self._last_poll_ts_ns: int | None = None
        self._running = False

    @property
    def url(self) -> str:
        return self._url

    def status(self) -> RaydiumPoolStatus:
        return RaydiumPoolStatus(
            running=self._running,
            url=self._url,
            last_poll_ts_ns=self._last_poll_ts_ns,
            snapshots_emitted=self._snapshots_emitted,
            errors=self._errors,
        )

    def stop(self) -> None:
        """Signal the poll loop to exit on its next iteration."""
        self._stop_event.set()

    async def run(self) -> None:
        """Poll the Raydium pairs endpoint forever (until ``stop()``)."""
        self._running = True
        delay = self._retry_delay_s
        try:
            while not self._stop_event.is_set():
                client = self._client_factory()
                try:
                    ok = await self._poll_once(client)
                finally:
                    try:
                        await client.aclose()
                    except Exception:  # noqa: BLE001
                        pass
                if ok:
                    delay = self._retry_delay_s
                    sleep_for = self._poll_interval_s
                else:
                    sleep_for = delay
                    delay = min(delay * 2.0, self._retry_delay_max_s)
                if self._stop_event.is_set():
                    break
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=sleep_for
                    )
                except TimeoutError:
                    pass
        finally:
            self._running = False

    async def _poll_once(self, client: httpx.AsyncClient) -> bool:
        try:
            resp = await client.get(self._url)
        except httpx.HTTPError:
            self._errors += 1
            LOG.exception("raydium_pools: GET %s failed", self._url)
            return False
        if resp.status_code != 200:
            self._errors += 1
            LOG.warning(
                "raydium_pools: GET %s -> %d", self._url, resp.status_code
            )
            return False
        try:
            data = resp.json()
        except ValueError:
            self._errors += 1
            LOG.warning(
                "raydium_pools: non-json body from %s", self._url
            )
            return False
        if not isinstance(data, list):
            self._errors += 1
            LOG.warning(
                "raydium_pools: unexpected payload shape from %s", self._url
            )
            return False
        ts_ns = self._clock_ns()
        snaps = parse_pairs(
            data, ts_ns=ts_ns, venue=self._venue, chain=self._chain
        )
        for snap in snaps:
            try:
                self._sink(snap)
            except Exception:  # noqa: BLE001 - sink must not kill the loop
                self._errors += 1
                LOG.exception(
                    "raydium_pools: sink raised on snap=%r", snap
                )
                continue
            self._snapshots_emitted += 1
        self._last_poll_ts_ns = ts_ns
        return True


async def aiter_status(
    poller: RaydiumPoolPoller, *, interval_s: float = 1.0
) -> AsyncIterator[RaydiumPoolStatus]:  # pragma: no cover - test helper
    while True:
        yield poller.status()
        await asyncio.sleep(interval_s)


__all__ = [
    "CHAIN_TAG",
    "ClientFactory",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_RETRY_DELAY_MAX_S",
    "DEFAULT_RETRY_DELAY_S",
    "DEFAULT_TIMEOUT_S",
    "RAYDIUM_PAIRS_URL",
    "RaydiumPoolPoller",
    "RaydiumPoolStatus",
    "VENUE_TAG",
    "aiter_status",
    "parse_pair",
    "parse_pairs",
]
