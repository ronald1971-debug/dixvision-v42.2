"""Read-only BLS HTTP macro adapter (SRC-MACRO-BLS-001).

Polls the U.S. Bureau of Labor Statistics public Data API v2
(``/publicAPI/v2/timeseries/data/``) on a fixed interval, batching every
configured series into a single POST per cycle, and converts each
observation into a canonical :class:`core.contracts.macro.MacroObservation`.

Layered split (mirrors :mod:`ui.feeds.fred_http`):

* :func:`make_bls_request_body` — pure JSON-body builder. Takes the
  registration key out-of-band; the BLS API accepts the key inside the
  POST body (``"registrationkey"``), not the URL, so redaction is
  body-side rather than URL-side.
* :func:`parse_bls_payload` — pure ``bytes | str`` →
  ``tuple[MacroObservation, ...]`` projection. Skips malformed
  observations rather than raising. Caller supplies ``ts_ns`` so the
  parser stays INV-15-pure. The parser is series-agnostic — it walks
  every ``Results.series[*]`` row in the payload, which is how the
  pump batches multiple series into one round-trip.
* :class:`BLSHTTPPump` — thin async I/O wrapper with poll interval +
  reconnect backoff. Takes a ``fetch`` callable and a ``clock_ns``
  callable so tests inject fakes (no real network, no ``time.time``)
  and the determinism boundary stays explicit.
* :class:`MacroFeedStatus` — frozen telemetry snapshot, exposed by the
  HTTP status endpoint.

INV-15: the parser is pure (caller-supplied ``ts_ns``); the pump is
non-deterministic by design (network), but every observation it emits
is funneled into the harness via the same code path the engine ledger
replays deterministically.

Why a separate module instead of generalising :mod:`ui.feeds.fred_http`:

* BLS uses **POST + JSON body**, FRED uses **GET + query string** — the
  HTTP shape is genuinely different at the request boundary.
* BLS supports **multi-series batching** in one round-trip (up to 50 with
  a registration key); FRED requires one request per series. The pump
  control flow differs (one outer try/except per cycle vs per-series).
* BLS encodes **value as a string with footnotes**, FRED encodes value
  as either string or number with a sentinel. The parsers diverge.
* The two registries (FRED, BLS) issue different keys with different
  rate limits — keeping the modules separate keeps the per-source
  redaction + status surfaces cleanly partitioned.

BLS API reference:
https://www.bls.gov/developers/api_signature_v2.htm
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol

from core.contracts.macro import MacroObservation

LOG = logging.getLogger(__name__)

#: Public BLS Data API base (HTTPS, JSON).
BLS_API_BASE = "https://api.bls.gov/publicAPI/v2"

#: Default poll cadence (seconds). BLS publishes most series monthly
#: with weekly intervention windows; 5 minutes is conservative and
#: comfortably below the v2 rate limit (500 req/day with key).
DEFAULT_POLL_INTERVAL_S = 300.0

#: Default reconnect backoff floor + ceiling (seconds).
DEFAULT_RECONNECT_DELAY_S = 5.0
DEFAULT_RECONNECT_DELAY_MAX_S = 600.0

#: Source tag stamped onto every emitted observation so the ledger
#: distinguishes BLS rows from FRED / manually injected ones.
SOURCE_TAG = "BLS"

#: BLS sentinels for "no data".
_BLS_MISSING_VALUES: Final[frozenset[str]] = frozenset({"", "-", "(NA)", "(R)"})

#: Period prefixes the parser understands. ``M`` = monthly,
#: ``Q`` = quarterly, ``S`` = semi-annual, ``A`` = annual.
_MONTHLY_PERIOD_PREFIX: Final[str] = "M"
_QUARTERLY_PERIOD_PREFIX: Final[str] = "Q"
_SEMIANNUAL_PERIOD_PREFIX: Final[str] = "S"
_ANNUAL_PERIOD_PREFIX: Final[str] = "A"

#: Per-quarter end month (1-indexed) used to normalise quarterly periods
#: to a calendar date (month-end is conventional for macro analysis).
_QUARTER_END_MONTH: Final[dict[str, int]] = {
    "Q01": 3,
    "Q02": 6,
    "Q03": 9,
    "Q04": 12,
    # Q05 is the BLS sentinel for an annual aggregate that happens
    # to live alongside quarterly rows in the same series; we map it
    # to December so it sorts last.
    "Q05": 12,
}

#: Per-semester end month (1-indexed).
_SEMIANNUAL_END_MONTH: Final[dict[str, int]] = {
    "S01": 6,
    "S02": 12,
    "S03": 12,
}


def make_bls_request_body(
    series_ids: Sequence[str],
    *,
    registration_key: str,
    start_year: int | None = None,
    end_year: int | None = None,
    catalog: bool = False,
    calculations: bool = False,
    annual_average: bool = False,
) -> str:
    """Return the canonical BLS v2 ``POST`` body as a JSON string.

    The registration key is included verbatim in the body; the pump's
    redaction layer strips it before logging or surfacing through
    :func:`MacroFeedStatus`. Callers MUST treat the returned string as
    secret-bearing.

    Args:
        series_ids: One or more BLS series ids (e.g. ``("CPIAUCSL",
            "UNRATE")``). Empty sequence is rejected. Up to 50 series
            may be batched per request when a key is supplied.
        registration_key: BLS-issued v2 API key. Empty string rejected.
        start_year: Optional 4-digit year filter (``catalog=False``
            still requires both bounds when one is set).
        end_year: Optional 4-digit year filter.
        catalog: If True, request series catalog metadata. We default
            to False because catalog responses include unredacted
            third-party publishers and balloon the payload.
        calculations: If True, request period-over-period calculations
            (1m, 3m, 6m, 12m). Defaults to False — projection logic
            performs its own deltas.
        annual_average: If True, request the annual-average period
            alongside the monthly rows (BLS exposes this as period
            ``M13``). Defaults to False to keep the period→date
            normaliser simpler.
    """
    if not registration_key:
        raise ValueError(
            "make_bls_request_body: registration_key must be non-empty"
        )
    # Defensive copy + dedup-by-id (preserves first-seen order).
    seen: set[str] = set()
    ordered: list[str] = []
    for sid in series_ids:
        if not sid:
            continue
        if sid in seen:
            continue
        seen.add(sid)
        ordered.append(sid)
    if not ordered:
        raise ValueError(
            "make_bls_request_body: at least one non-empty series id required"
        )
    if (start_year is None) ^ (end_year is None):
        raise ValueError(
            "make_bls_request_body: start_year and end_year must be set together"
        )
    if start_year is not None and end_year is not None:
        if start_year < 1900 or end_year < 1900:
            raise ValueError("make_bls_request_body: years must be >= 1900")
        if end_year < start_year:
            raise ValueError(
                "make_bls_request_body: end_year must be >= start_year"
            )
    body: dict[str, object] = {
        "seriesid": ordered,
        "registrationkey": registration_key,
        "catalog": catalog,
        "calculations": calculations,
        "annualaverage": annual_average,
    }
    if start_year is not None and end_year is not None:
        body["startyear"] = str(start_year)
        body["endyear"] = str(end_year)
    return json.dumps(body, separators=(",", ":"), sort_keys=True)


def _redact_request_body(body: str) -> str:
    """Strip the ``registrationkey`` field from a request body string.

    Used by status / log paths so the v2 API key never leaks. Falls
    back to a generic placeholder if the body is unparseable JSON.
    """
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return "<redacted-body>"
    if not isinstance(parsed, dict):
        return "<redacted-body>"
    if "registrationkey" in parsed:
        parsed["registrationkey"] = "REDACTED"
    return json.dumps(parsed, separators=(",", ":"), sort_keys=True)


def _period_to_observation_date(year: str, period: str) -> str | None:
    """Convert a BLS ``(year, period)`` pair to ISO ``YYYY-MM-DD``.

    Returns ``None`` for unparseable / out-of-range inputs. The parser
    drops rows where this returns ``None`` because
    :class:`MacroObservation.observation_date` must be non-empty.
    """
    if not year or not period or len(year) != 4 or not year.isdigit():
        return None
    yr = int(year)
    if yr < 1900:
        return None
    if period.startswith(_MONTHLY_PERIOD_PREFIX):
        # M01..M12 -> first-of-month (BLS convention is mid-month
        # publication, but the as-of date is the publishing month;
        # using day=01 keeps replay deterministic).
        suffix = period[len(_MONTHLY_PERIOD_PREFIX):]
        if not suffix.isdigit():
            return None
        month = int(suffix)
        if not 1 <= month <= 12:
            return None
        return f"{yr:04d}-{month:02d}-01"
    if period in _QUARTER_END_MONTH:
        month = _QUARTER_END_MONTH[period]
        return f"{yr:04d}-{month:02d}-01"
    if period in _SEMIANNUAL_END_MONTH:
        month = _SEMIANNUAL_END_MONTH[period]
        return f"{yr:04d}-{month:02d}-01"
    if period.startswith(_ANNUAL_PERIOD_PREFIX):
        return f"{yr:04d}-12-01"
    return None


def _observation_date_to_ns(date_str: str) -> int | None:
    """Convert an ISO ``YYYY-MM-DD`` to a UTC-midnight ns timestamp."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None
    try:
        ts_ns = int(dt.timestamp() * 1_000_000_000)
    except (OverflowError, ValueError):
        return None
    return ts_ns if ts_ns > 0 else None


def _parse_value(raw: object) -> float | None:
    """Convert a BLS ``value`` field to ``float | None``.

    BLS encodes "no data" as one of :data:`_BLS_MISSING_VALUES`; values
    that parse as a finite float are returned numerically.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text in _BLS_MISSING_VALUES:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def parse_bls_payload(
    payload: bytes | str,
    *,
    ts_ns: int,
    source: str = SOURCE_TAG,
    units_overrides: dict[str, str] | None = None,
    title_overrides: dict[str, str] | None = None,
) -> tuple[MacroObservation, ...]:
    """Project a BLS v2 response document into a MacroObservation tuple.

    The BLS payload returns ``Results.series`` as a list — one entry
    per series id requested — each carrying its own ``data`` array of
    period observations. The parser walks every ``data`` row in every
    series, in the order BLS returned them, so multi-series batching
    is transparent to the caller.

    Skips (never raises on) malformed observations; an entry must
    project to a usable ``observation_date`` and a non-empty
    ``series_id`` to be emitted, otherwise it is dropped silently and
    the caller's pump increments the per-call ``errors`` counter (or
    not — soft drop).

    INV-15 (pure projection): ``ts_ns`` is supplied by the caller,
    never derived from ``payload`` or a system clock, so two replays
    with the same input produce byte-identical output.

    Args:
        payload: Raw JSON document body from the BLS endpoint.
        ts_ns: Caller-supplied ingestion timestamp (TimeAuthority).
        source: SCVS source tag. Defaults to ``"BLS"``.
        units_overrides: Optional ``series_id -> units`` map propagated
            from the caller's registry — BLS doesn't echo the unit on
            every row, only in the catalog block (which we skip by
            default). The override lets the registry stamp e.g.
            ``"Index 1982-1984=100"`` onto every CPIAUCSL observation.
        title_overrides: Optional ``series_id -> title`` map for the
            human-readable series title.
    """
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

    # BLS surfaces request-level errors via ``status="REQUEST_NOT_PROCESSED"``
    # plus a ``message`` array. We treat those as a parser-level no-op
    # rather than raising — the pump's ``errors`` counter is the right
    # place for this signal.
    status = doc.get("status")
    if isinstance(status, str) and status != "REQUEST_SUCCEEDED":
        return ()

    results = doc.get("Results")
    if not isinstance(results, dict):
        return ()
    series_list = results.get("series")
    if not isinstance(series_list, list):
        return ()

    units_map = dict(units_overrides) if units_overrides else {}
    title_map = dict(title_overrides) if title_overrides else {}

    out: list[MacroObservation] = []
    for series_entry in series_list:
        if not isinstance(series_entry, dict):
            continue
        series_id = series_entry.get("seriesID")
        if not isinstance(series_id, str) or not series_id:
            continue
        units = units_map.get(series_id, "")
        title = title_map.get(series_id, "")
        data_rows = series_entry.get("data")
        if not isinstance(data_rows, list):
            continue
        for entry in data_rows:
            if not isinstance(entry, dict):
                continue
            year = entry.get("year")
            period = entry.get("period")
            if not isinstance(year, str) or not isinstance(period, str):
                continue
            obs_date = _period_to_observation_date(year, period)
            if obs_date is None:
                continue
            value = _parse_value(entry.get("value"))
            observed_ts_ns = _observation_date_to_ns(obs_date)
            try:
                out.append(
                    MacroObservation(
                        ts_ns=ts_ns,
                        source=source,
                        series_id=series_id,
                        observation_date=obs_date,
                        value=value,
                        units=units,
                        title=title,
                        observed_ts_ns=observed_ts_ns,
                    )
                )
            except ValueError:
                # Defensive — MacroObservation.__post_init__ rejects
                # empty required fields, all of which we filter above.
                continue
    return tuple(out)


HTTPPost = Callable[[str, str], Awaitable[bytes]]
"""Async ``(url, body) -> bytes`` callable. Production uses urllib."""


async def _default_post(url: str, body: str) -> bytes:
    """Default POSTer — imported lazily so the parser is usable
    without an HTTP client installed (tests, lint, etc.). Uses the
    stdlib so we don't add a dependency for the canonical macro pump.
    """
    import urllib.request  # local import; stdlib

    def _blocking_post() -> bytes:
        req = urllib.request.Request(  # noqa: S310 - fixed https URL
            url,
            data=body.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "dixvision-v42.2/bls-http",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.read()

    return await asyncio.to_thread(_blocking_post)


@dataclass(frozen=True, slots=True)
class MacroFeedStatus:
    """Snapshot of pump health — exposed by ``GET /api/feeds/bls/status``.

    Mirrors :class:`ui.feeds.fred_http.MacroFeedStatus` so the
    dashboard JSON surface is uniform across macro adapters.
    """

    running: bool
    source: str
    series_ids: tuple[str, ...]
    last_poll_ts_ns: int | None
    last_observation_ts_ns: int | None
    observations_received: int
    polls: int
    errors: int


@dataclass(frozen=True, slots=True)
class BLSSeriesSpec:
    """One series the pump should poll on each tick.

    Attributes:
        series_id: BLS series id (e.g. ``"CPIAUCSL"``).
        units: Optional unit hint stamped onto every emitted
            ``MacroObservation``.
        title: Optional human-readable series title.
    """

    series_id: str
    units: str = ""
    title: str = ""

    def __post_init__(self) -> None:
        if not self.series_id:
            raise ValueError("BLSSeriesSpec.series_id must be non-empty")


class _StopHandle(Protocol):
    """Minimal subset of :class:`asyncio.Event` we use, for typing."""

    def set(self) -> None: ...  # pragma: no cover - protocol
    def is_set(self) -> bool: ...  # pragma: no cover - protocol

    async def wait(self) -> bool: ...  # pragma: no cover - protocol


class BLSHTTPPump:
    """Async pump batching multiple BLS series into one POST per cycle.

    The sink callable runs synchronously inside the asyncio loop; for
    cross-thread state mutation use a :class:`ui.feeds.runner.FeedRunner`
    -style wrapper.

    Unlike FRED, BLS supports batching all configured series into a
    single round-trip — so per cycle the pump issues exactly one POST,
    parses the multi-series response, and emits every observation to
    ``sink``. Per-series error counters are not tracked at this layer
    (BLS partial responses surface as ``status="REQUEST_NOT_PROCESSED"``
    with a ``message`` array describing which series failed); the
    pump increments a single ``errors`` counter on transport / parse
    failure of the whole cycle.
    """

    def __init__(
        self,
        sink: Callable[[MacroObservation], None],
        *,
        registration_key: str,
        series: Sequence[BLSSeriesSpec],
        clock_ns: Callable[[], int],
        post: HTTPPost | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        reconnect_delay_max_s: float = DEFAULT_RECONNECT_DELAY_MAX_S,
        base: str = BLS_API_BASE,
        source: str = SOURCE_TAG,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("BLSHTTPPump: poll_interval_s must be positive")
        if reconnect_delay_s <= 0:
            raise ValueError("BLSHTTPPump: reconnect_delay_s must be positive")
        if reconnect_delay_max_s < reconnect_delay_s:
            raise ValueError(
                "BLSHTTPPump: reconnect_delay_max_s must be >= reconnect_delay_s"
            )
        if not source:
            raise ValueError("BLSHTTPPump: source must be non-empty")
        if not registration_key:
            raise ValueError("BLSHTTPPump: registration_key must be non-empty")
        if not series:
            raise ValueError("BLSHTTPPump: at least one series spec required")
        # Defensive copy + dedup-by-id (preserves first-seen order).
        seen: set[str] = set()
        ordered: list[BLSSeriesSpec] = []
        for spec in series:
            if spec.series_id in seen:
                continue
            seen.add(spec.series_id)
            ordered.append(spec)
        self._sink = sink
        self._registration_key = registration_key
        self._series: tuple[BLSSeriesSpec, ...] = tuple(ordered)
        self._clock_ns = clock_ns
        self._post: HTTPPost = post if post is not None else _default_post
        self._poll_interval_s = poll_interval_s
        self._reconnect_delay_s = reconnect_delay_s
        self._reconnect_delay_max_s = reconnect_delay_max_s
        self._base = base
        self._source = source
        self._start_year = start_year
        self._end_year = end_year

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
        """Poll the BLS endpoint until ``stop`` is set."""
        self._running = True
        backoff = self._reconnect_delay_s
        try:
            while not stop.is_set():
                try:
                    await self._poll_once()
                except Exception:  # noqa: BLE001 — defensive boundary
                    self._errors += 1
                    LOG.exception(
                        "bls_http: poll cycle failed, backing off %.1fs",
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
        """POST every configured series in one batch and dispatch
        observations.

        The whole cycle re-raises on transport / parse failure so the
        outer ``run`` loop applies backoff. Sink-side exceptions are
        absorbed (per-observation) and counted in ``errors``.
        """
        self._polls += 1
        self._last_poll_ts_ns = self._clock_ns()
        url = f"{self._base.rstrip('/')}/timeseries/data/"
        body = make_bls_request_body(
            tuple(spec.series_id for spec in self._series),
            registration_key=self._registration_key,
            start_year=self._start_year,
            end_year=self._end_year,
        )
        payload = await self._post(url, body)
        ts_ns = self._clock_ns()
        units_overrides = {
            spec.series_id: spec.units for spec in self._series if spec.units
        }
        title_overrides = {
            spec.series_id: spec.title for spec in self._series if spec.title
        }
        observations = parse_bls_payload(
            payload,
            ts_ns=ts_ns,
            source=self._source,
            units_overrides=units_overrides or None,
            title_overrides=title_overrides or None,
        )
        for obs in observations:
            try:
                self._sink(obs)
            except Exception:  # noqa: BLE001 — sink boundary
                self._errors += 1
                LOG.exception(
                    "bls_http: sink raised for %s/%s",
                    obs.series_id,
                    obs.observation_date,
                )
                continue
            self._observations_received += 1
            if obs.observed_ts_ns is not None and (
                self._last_observation_ts_ns is None
                or obs.observed_ts_ns > self._last_observation_ts_ns
            ):
                self._last_observation_ts_ns = obs.observed_ts_ns


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
    "BLS_API_BASE",
    "BLSHTTPPump",
    "BLSSeriesSpec",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_RECONNECT_DELAY_MAX_S",
    "DEFAULT_RECONNECT_DELAY_S",
    "HTTPPost",
    "MacroFeedStatus",
    "SOURCE_TAG",
    "make_bls_request_body",
    "parse_bls_payload",
]
