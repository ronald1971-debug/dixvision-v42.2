"""Read-only FRED HTTP macro adapter (SRC-MACRO-FRED-001).

Polls the public FRED ``/fred/series/observations`` JSON endpoint on a
fixed interval per series and converts each observation into a canonical
:class:`core.contracts.macro.MacroObservation`.

Layered split (mirrors :mod:`ui.feeds.coindesk_rss`):

* :func:`make_fred_observations_url` — pure URL builder. Takes the API
  key out-of-band (never inlined in tests, never logged in normal
  operation).
* :func:`parse_observations_payload` — pure ``bytes | str`` →
  ``tuple[MacroObservation, ...]`` projection. Skips malformed
  observations rather than raising. Caller supplies ``ts_ns`` so the
  parser stays INV-15-pure.
* :class:`FredHTTPPump` — thin async I/O wrapper with poll interval +
  reconnect backoff. Takes a ``fetch`` callable and a ``clock_ns``
  callable so tests inject fakes (no real network, no ``time.time``)
  and the determinism boundary stays explicit.
* :class:`MacroFeedStatus` — frozen telemetry snapshot, exposed by the
  HTTP status endpoint.

INV-15: the parser is pure (caller-supplied ``ts_ns``); the pump is
non-deterministic by design (network), but every observation it emits
is funneled into the harness via the same code path the engine ledger
replays deterministically.

FRED API reference:
https://fred.stlouisfed.org/docs/api/fred/series_observations.html
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from core.contracts.macro import MacroObservation

LOG = logging.getLogger(__name__)

#: Public FRED API base (HTTPS, JSON).
FRED_API_BASE = "https://api.stlouisfed.org/fred"

#: Default poll cadence (seconds). FRED series update at most daily,
#: so 5 minutes is conservative and safely below the rate limit.
DEFAULT_POLL_INTERVAL_S = 300.0

#: Default reconnect backoff floor + ceiling (seconds).
DEFAULT_RECONNECT_DELAY_S = 5.0
DEFAULT_RECONNECT_DELAY_MAX_S = 600.0

#: Source tag stamped onto every emitted observation so the ledger
#: distinguishes FRED rows from manually injected ones.
SOURCE_TAG = "FRED"

#: FRED's sentinel for "no data" (holidays, unreleased prints, …).
_FRED_MISSING_VALUE = "."


def make_fred_observations_url(
    series_id: str,
    api_key: str,
    *,
    base: str = FRED_API_BASE,
    file_type: str = "json",
    limit: int | None = None,
    sort_order: str = "asc",
) -> str:
    """Return the canonical FRED ``/series/observations`` URL.

    The API key is appended via :mod:`urllib.parse` so values that
    contain reserved characters (``+``, ``/``, ``&``) are escaped
    rather than silently truncating the query.
    """
    if not series_id:
        raise ValueError("make_fred_observations_url: series_id must be non-empty")
    if not api_key:
        raise ValueError("make_fred_observations_url: api_key must be non-empty")
    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": file_type,
        "sort_order": sort_order,
    }
    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "make_fred_observations_url: limit must be positive"
            )
        params["limit"] = str(limit)
    return f"{base.rstrip('/')}/series/observations?" + urllib.parse.urlencode(
        params
    )


def _parse_observation_date(raw: str | None) -> int | None:
    """Parse an ISO ``YYYY-MM-DD`` date into a UTC-midnight ns timestamp.

    Returns ``None`` for empty / unparseable input — never raises, so
    one bad row never poisons the whole payload.
    """
    if not raw:
        return None
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None
    try:
        ts_ns = int(dt.timestamp() * 1_000_000_000)
    except (OverflowError, ValueError):
        return None
    # MacroObservation.observed_ts_ns must be positive or None per its
    # contract. Pre-1970 dates are returned as None so the rest of the
    # row (series_id, value, units) can still be emitted.
    return ts_ns if ts_ns > 0 else None


def _parse_value(raw: object) -> float | None:
    """Convert a FRED ``"value"`` field to ``float | None``.

    FRED encodes "no data" as ``"."``; anything else parseable as a
    float is returned numerically. Strings that don't parse return
    ``None`` (caller drops the row only if it can't construct the
    contract — a missing value is still a valid observation).
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        # FRED never emits NaN/Inf; defend against future API changes.
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text == _FRED_MISSING_VALUE:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def parse_observations_payload(
    payload: bytes | str,
    *,
    ts_ns: int,
    series_id: str,
    source: str = SOURCE_TAG,
    units: str = "",
    title: str = "",
) -> tuple[MacroObservation, ...]:
    """Project a FRED observations JSON document into MacroObservation tuple.

    Skips (never raises on) malformed observations; an entry must have
    a non-empty ``"date"`` field to be emitted, otherwise it is dropped
    silently and the caller's pump increments the per-call ``errors``
    counter.

    INV-15 (pure projection): ``ts_ns``, ``series_id``, ``units`` and
    ``title`` are supplied by the caller, never derived from
    ``payload`` or a system clock, so two replays with the same input
    produce byte-identical output.

    Args:
        payload: Raw JSON document body from the FRED endpoint.
        ts_ns: Caller-supplied ingestion timestamp (TimeAuthority).
        series_id: Series tag stamped onto every observation. Required
            because the FRED payload doesn't echo it back.
        source: SCVS source tag. Defaults to ``"FRED"``.
        units: Optional unit hint propagated from the registry.
        title: Optional human-readable series title.
    """
    if not series_id:
        raise ValueError("parse_observations_payload: series_id must be non-empty")
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return ()
    else:
        text = payload
    if not text:
        return ()
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return ()
    if not isinstance(doc, dict):
        return ()
    raw_obs = doc.get("observations")
    if not isinstance(raw_obs, list):
        return ()

    out: list[MacroObservation] = []
    for entry in raw_obs:
        if not isinstance(entry, dict):
            continue
        date = entry.get("date")
        if not isinstance(date, str) or not date:
            continue
        value = _parse_value(entry.get("value"))
        observed_ts_ns = _parse_observation_date(date)
        try:
            out.append(
                MacroObservation(
                    ts_ns=ts_ns,
                    source=source,
                    series_id=series_id,
                    observation_date=date,
                    value=value,
                    units=units,
                    title=title,
                    observed_ts_ns=observed_ts_ns,
                )
            )
        except ValueError:
            # Defensive — MacroObservation.__post_init__ already covers
            # the empty-string cases we filter above; future stricter
            # validation would land here.
            continue
    return tuple(out)


HTTPFetch = Callable[[str], Awaitable[bytes]]
"""Async URL → bytes callable. Production uses :mod:`urllib.request`."""


async def _default_fetch(url: str) -> bytes:
    """Default fetcher — imported lazily so the parser is usable without
    an HTTP client installed (tests, lint, etc.). Uses the stdlib so we
    don't add a dependency for the canonical macro pump.
    """
    import urllib.request  # local import; stdlib

    def _blocking_get() -> bytes:
        req = urllib.request.Request(  # noqa: S310 - fixed https URL
            url,
            headers={"User-Agent": "dixvision-v42.2/fred-http"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.read()

    return await asyncio.to_thread(_blocking_get)


@dataclass(frozen=True, slots=True)
class MacroFeedStatus:
    """Snapshot of pump health — exposed by ``GET /api/feeds/fred/status``.

    The URL deliberately omits the API key; the pump strips it from
    the recorded URL via :func:`_redact_api_key` before publishing.
    """

    running: bool
    source: str
    series_ids: tuple[str, ...]
    last_poll_ts_ns: int | None
    last_observation_ts_ns: int | None
    observations_received: int
    polls: int
    errors: int


def _redact_api_key(url: str) -> str:
    """Strip the ``api_key`` query parameter from a URL before logging.

    FRED requires an API key as a query parameter, so the URL we use
    internally has the secret embedded. Status endpoints and logs must
    never echo it. This helper rewrites the query so a future ops
    reader can still see what series was polled, without leaking the
    credential.
    """
    parsed = urllib.parse.urlsplit(url)
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    sanitized = [(k, "REDACTED" if k == "api_key" else v) for k, v in pairs]
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(sanitized),
            parsed.fragment,
        )
    )


@dataclass(frozen=True, slots=True)
class FredSeriesSpec:
    """One series the pump should poll on each tick.

    Attributes:
        series_id: FRED series id (e.g. ``"DGS10"``).
        units: Optional unit hint stamped onto every emitted
            ``MacroObservation``.
        title: Optional human-readable series title.
        limit: Optional cap on the number of trailing observations
            requested per poll. ``None`` returns the default FRED
            window (the full series).
    """

    series_id: str
    units: str = ""
    title: str = ""
    limit: int | None = None

    def __post_init__(self) -> None:
        if not self.series_id:
            raise ValueError("FredSeriesSpec.series_id must be non-empty")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("FredSeriesSpec.limit must be positive")


class _StopHandle(Protocol):
    """Minimal subset of :class:`asyncio.Event` we use, for typing."""

    def set(self) -> None: ...  # pragma: no cover - protocol
    def is_set(self) -> bool: ...  # pragma: no cover - protocol

    async def wait(self) -> bool: ...  # pragma: no cover - protocol


class FredHTTPPump:
    """Async pump polling the FRED observations endpoint into a sink.

    The sink callable runs synchronously inside the asyncio loop; for
    cross-thread state mutation use a :class:`ui.feeds.runner.FeedRunner`
    -style wrapper.

    The pump treats each :class:`FredSeriesSpec` independently: per
    poll cycle it iterates them in order, fetches each one, and emits
    every observation to ``sink``. A failure on one series advances
    the per-series ``errors`` counter and continues to the next; the
    whole tick fails (and triggers reconnect backoff) only if every
    series in the cycle raises.
    """

    def __init__(
        self,
        sink: Callable[[MacroObservation], None],
        *,
        api_key: str,
        series: Sequence[FredSeriesSpec],
        clock_ns: Callable[[], int],
        fetch: HTTPFetch | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        reconnect_delay_max_s: float = DEFAULT_RECONNECT_DELAY_MAX_S,
        base: str = FRED_API_BASE,
        source: str = SOURCE_TAG,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("FredHTTPPump: poll_interval_s must be positive")
        if reconnect_delay_s <= 0:
            raise ValueError("FredHTTPPump: reconnect_delay_s must be positive")
        if reconnect_delay_max_s < reconnect_delay_s:
            raise ValueError(
                "FredHTTPPump: reconnect_delay_max_s must be >= reconnect_delay_s"
            )
        if not source:
            raise ValueError("FredHTTPPump: source must be non-empty")
        if not api_key:
            raise ValueError("FredHTTPPump: api_key must be non-empty")
        if not series:
            raise ValueError("FredHTTPPump: at least one series spec required")
        # Defensive copy + dedup-by-id (preserves first-seen order).
        seen: set[str] = set()
        ordered: list[FredSeriesSpec] = []
        for spec in series:
            if spec.series_id in seen:
                continue
            seen.add(spec.series_id)
            ordered.append(spec)
        self._sink = sink
        self._api_key = api_key
        self._series: tuple[FredSeriesSpec, ...] = tuple(ordered)
        self._clock_ns = clock_ns
        self._fetch: HTTPFetch = fetch if fetch is not None else _default_fetch
        self._poll_interval_s = poll_interval_s
        self._reconnect_delay_s = reconnect_delay_s
        self._reconnect_delay_max_s = reconnect_delay_max_s
        self._base = base
        self._source = source

        self._running: bool = False
        self._last_poll_ts_ns: int | None = None
        self._last_observation_ts_ns: int | None = None
        self._observations_received: int = 0
        self._polls: int = 0
        self._errors: int = 0

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(spec.series_id for spec in self._series)

    def status(self) -> MacroFeedStatus:
        """Return a frozen snapshot of pump telemetry."""
        return MacroFeedStatus(
            running=self._running,
            source=self._source,
            series_ids=self.series_ids,
            last_poll_ts_ns=self._last_poll_ts_ns,
            last_observation_ts_ns=self._last_observation_ts_ns,
            observations_received=self._observations_received,
            polls=self._polls,
            errors=self._errors,
        )

    async def run(self, stop: _StopHandle) -> None:
        """Poll the FRED endpoint until ``stop`` is set.

        The loop sleeps in two places:

        * the regular cadence between successful polls
          (``poll_interval_s``), and
        * an exponential backoff on errors (capped at
          ``reconnect_delay_max_s``).

        Both sleeps are interruptible — calling ``stop.set()`` causes
        the pending wait to short-circuit so shutdown is prompt.
        """
        self._running = True
        backoff = self._reconnect_delay_s
        try:
            while not stop.is_set():
                try:
                    await self._poll_once()
                except Exception:  # noqa: BLE001 — defensive boundary
                    self._errors += 1
                    LOG.exception(
                        "fred_http: poll cycle failed, backing off %.1fs",
                        backoff,
                    )
                    if await _interruptible_sleep(stop, backoff):
                        return
                    backoff = min(backoff * 2.0, self._reconnect_delay_max_s)
                    continue
                backoff = self._reconnect_delay_s
                if await _interruptible_sleep(stop, self._poll_interval_s):
                    return
        finally:
            self._running = False

    async def _poll_once(self) -> None:
        """Fetch every configured series once and dispatch observations.

        A per-series exception increments ``errors`` and is *not*
        propagated; only an across-the-board failure (every series
        raising) re-raises so the outer loop can apply backoff.
        """
        any_success = False
        any_failure = False
        last_failure: BaseException | None = None
        self._polls += 1
        self._last_poll_ts_ns = self._clock_ns()
        for spec in self._series:
            url = make_fred_observations_url(
                spec.series_id,
                self._api_key,
                base=self._base,
                limit=spec.limit,
            )
            try:
                payload = await self._fetch(url)
            except Exception as exc:  # noqa: BLE001 — defensive boundary
                self._errors += 1
                last_failure = exc
                any_failure = True
                LOG.warning(
                    "fred_http: fetch failed for series %s (%s)",
                    spec.series_id,
                    _redact_api_key(url),
                )
                continue
            ts_ns = self._clock_ns()
            observations = parse_observations_payload(
                payload,
                ts_ns=ts_ns,
                series_id=spec.series_id,
                source=self._source,
                units=spec.units,
                title=spec.title,
            )
            for obs in observations:
                try:
                    self._sink(obs)
                except Exception:  # noqa: BLE001 — sink boundary
                    self._errors += 1
                    LOG.exception(
                        "fred_http: sink raised for %s/%s",
                        spec.series_id,
                        obs.observation_date,
                    )
                    continue
                self._observations_received += 1
                if obs.observed_ts_ns is not None and (
                    self._last_observation_ts_ns is None
                    or obs.observed_ts_ns > self._last_observation_ts_ns
                ):
                    self._last_observation_ts_ns = obs.observed_ts_ns
            any_success = True
        if not any_success and any_failure and last_failure is not None:
            raise last_failure


async def _interruptible_sleep(stop: _StopHandle, seconds: float) -> bool:
    """Sleep for ``seconds`` unless ``stop`` is set first.

    Returns ``True`` if the sleep was cut short by ``stop`` (caller
    should exit the outer loop), ``False`` otherwise.
    """
    if seconds <= 0:
        return stop.is_set()
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True


__all__ = [
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_RECONNECT_DELAY_MAX_S",
    "DEFAULT_RECONNECT_DELAY_S",
    "FRED_API_BASE",
    "FredHTTPPump",
    "FredSeriesSpec",
    "HTTPFetch",
    "MacroFeedStatus",
    "SOURCE_TAG",
    "make_fred_observations_url",
    "parse_observations_payload",
]
