"""Read-only Dune Analytics query adapter (SRC-ONCHAIN-DUNE-001).

Drives the Dune Analytics asynchronous-execution API
(``POST /query/{id}/execute`` → poll ``/execution/{id}/status`` →
``GET /execution/{id}/results``) and projects each returned row into a
canonical :class:`sensory.onchain.contracts.OnChainMetric`.

Layered split (mirrors :mod:`ui.feeds.fred_http`):

* URL builders (:func:`make_execute_url`, :func:`make_status_url`,
  :func:`make_results_url`, :func:`make_verify_url`) — pure, IO-free.
  The API key is supplied out-of-band via the ``X-DUNE-API-KEY``
  request header, **never** inlined in URL query strings (Dune
  rejects key-in-URL anyway, and our authority-lint pins the
  separation).
* Pure parsers (:func:`parse_execution_id`,
  :func:`parse_execution_status`, :func:`parse_results_payload`) —
  ``bytes | str`` → typed value. Skip malformed rows rather than
  raising. Caller supplies ``ts_ns`` so the parser stays INV-15-pure.
* :class:`DuneQuerySpec` — frozen + slotted spec for one query the
  adapter polls (query_id + metric tag + optional asset / unit /
  value column).
* :class:`DuneAnalyticsClient` — thin async I/O wrapper with an
  injectable ``fetch`` callable + ``clock_ns`` callable so tests
  inject fakes (no real network, no ``time.time``) and the
  determinism boundary stays explicit. Per-query state is *not*
  retained between runs — every call to :meth:`run_query_once`
  carries its own short-lived execution id.
* :class:`DuneFeedStatus` — frozen telemetry snapshot, exposed by
  any future ``GET /api/feeds/dune/status`` route.

INV-15: every parser is pure (caller-supplied ``ts_ns``); the
client itself is non-deterministic by design (network), but every
``OnChainMetric`` it produces flows into the harness via the same
code path the engine ledger replays deterministically.

B1 isolation: this module never imports
:mod:`intelligence_engine` / :mod:`execution_engine` /
:mod:`governance_engine` / :mod:`evolution_engine` — it is a
read-only sensory adapter on the upstream side of the typed-event
authority boundary.

B27 / B28 / INV-71: this module never constructs ``SignalEvent`` /
``HazardEvent`` / ``ExecutionEvent`` / ``SystemEvent`` /
``PatchProposal``. ``OnChainMetric`` is the advisory value type;
projection to typed bus events happens downstream of the authority
boundary.

Dune API reference:
https://docs.dune.com/api-reference/executions/execute-query
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from sensory.onchain.contracts import OnChainMetric

LOG = logging.getLogger(__name__)

#: Public Dune API base (HTTPS, JSON). All routes documented at
#: https://docs.dune.com/api-reference/.
DUNE_API_BASE = "https://api.dune.com/api/v1"

#: Required header carrying the API key. Dune does **not** accept an
#: ``Authorization: Bearer`` header — every request must use this
#: header instead. Pinned by :func:`_build_request`.
DUNE_API_KEY_HEADER = "X-DUNE-API-KEY"

#: Source tag stamped onto every emitted :class:`OnChainMetric`.
#: Matches the SCVS row ``SRC-ONCHAIN-DUNE-001`` and the verifier
#: spec in :mod:`system_engine.credentials.verifiers`.
SOURCE_TAG = "DUNE"

#: Default poll cadence for the execution-status loop (seconds).
#: Dune executions for typical analytic queries complete in 5-30 s;
#: 2 s is short enough to feel responsive without spamming the API.
DEFAULT_POLL_INTERVAL_S = 2.0

#: Default ceiling on how long one query may take before the client
#: gives up and increments the ``errors`` counter (seconds).
DEFAULT_EXECUTION_TIMEOUT_S = 300.0

#: Canonical Dune execution states.  These are surfaced verbatim by
#: the ``/execution/{id}/status`` endpoint; treating them as string
#: constants (rather than an :class:`enum.Enum`) keeps the parser
#: pure-string and avoids unnecessary churn if Dune adds a new
#: state.
EXECUTION_STATE_PENDING = "QUERY_STATE_PENDING"
EXECUTION_STATE_EXECUTING = "QUERY_STATE_EXECUTING"
EXECUTION_STATE_COMPLETED = "QUERY_STATE_COMPLETED"
EXECUTION_STATE_FAILED = "QUERY_STATE_FAILED"
EXECUTION_STATE_CANCELLED = "QUERY_STATE_CANCELLED"
EXECUTION_STATE_EXPIRED = "QUERY_STATE_EXPIRED"

#: States we consider terminal — once the client sees any of these
#: it stops polling.  ``COMPLETED`` is the only success state; the
#: rest mean "no rows will ever come".
TERMINAL_STATES: frozenset[str] = frozenset(
    {
        EXECUTION_STATE_COMPLETED,
        EXECUTION_STATE_FAILED,
        EXECUTION_STATE_CANCELLED,
        EXECUTION_STATE_EXPIRED,
    }
)


# ---------------------------------------------------------------------------
# URL builders — pure, no IO.
# ---------------------------------------------------------------------------


def _validate_query_id(query_id: int) -> None:
    if not isinstance(query_id, int) or isinstance(query_id, bool):
        raise TypeError("query_id must be an int")
    if query_id <= 0:
        raise ValueError("query_id must be positive")


def _validate_execution_id(execution_id: str) -> None:
    if not isinstance(execution_id, str):
        raise TypeError("execution_id must be a str")
    if not execution_id:
        raise ValueError("execution_id must be non-empty")


def make_execute_url(query_id: int, *, base: str = DUNE_API_BASE) -> str:
    """Return the canonical Dune ``POST /query/{id}/execute`` URL.

    The query id is path-segment-validated (positive int); the base
    is stripped of trailing slashes for byte-stability across the
    test suite.
    """
    _validate_query_id(query_id)
    return f"{base.rstrip('/')}/query/{query_id}/execute"


def make_status_url(execution_id: str, *, base: str = DUNE_API_BASE) -> str:
    """Return the canonical Dune ``GET /execution/{id}/status`` URL."""
    _validate_execution_id(execution_id)
    safe = urllib.parse.quote(execution_id, safe="")
    return f"{base.rstrip('/')}/execution/{safe}/status"


def make_results_url(execution_id: str, *, base: str = DUNE_API_BASE) -> str:
    """Return the canonical Dune ``GET /execution/{id}/results`` URL."""
    _validate_execution_id(execution_id)
    safe = urllib.parse.quote(execution_id, safe="")
    return f"{base.rstrip('/')}/execution/{safe}/results"


def make_verify_url(query_id: int = 1, *, base: str = DUNE_API_BASE) -> str:
    """Return the canonical Dune ``GET /query/{id}`` URL.

    Used by :mod:`system_engine.credentials.verifiers` as a
    non-consumptive auth-ping: ``GET /query/1`` returns 200 with
    valid key, 401 without — and does **not** spend execution
    credits the way ``POST /execute`` would.
    """
    _validate_query_id(query_id)
    return f"{base.rstrip('/')}/query/{query_id}"


# ---------------------------------------------------------------------------
# Pure parsers — bytes/str → typed values. INV-15 safe.
# ---------------------------------------------------------------------------


def _decode_payload(payload: bytes | str) -> object | None:
    """Decode a JSON body to a Python object, or ``None`` on any
    decode error.  Never raises.
    """
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        text = payload
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_execution_id(payload: bytes | str) -> str | None:
    """Project a ``POST /query/{id}/execute`` body into the execution id.

    The Dune contract documents this body as::

        {"execution_id": "01HF1...", "state": "QUERY_STATE_PENDING"}

    Returns the ``execution_id`` string if present and non-empty,
    otherwise ``None``.  Never raises.
    """
    doc = _decode_payload(payload)
    if not isinstance(doc, dict):
        return None
    raw = doc.get("execution_id")
    if isinstance(raw, str) and raw:
        return raw
    return None


def parse_execution_status(payload: bytes | str) -> str | None:
    """Project a ``GET /execution/{id}/status`` body into the state string.

    Returns the canonical ``QUERY_STATE_*`` string if present, else
    ``None``.  Never raises.
    """
    doc = _decode_payload(payload)
    if not isinstance(doc, dict):
        return None
    raw = doc.get("state")
    if isinstance(raw, str) and raw:
        return raw
    return None


def _parse_value(raw: object) -> float | None:
    """Convert one Dune result-cell to ``float | None``.

    Dune emits numeric columns as bare JSON numbers, but some legacy
    queries stringify them — accept both.  Non-finite values are
    coerced to ``None`` so an outlier never poisons replay.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        # JSON booleans are not numeric here — protect against
        # ``json.loads`` returning True/False for unexpected columns.
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return None


def _parse_observed_ts_ns(raw: object) -> int | None:
    """Parse a Dune result-row timestamp into nanoseconds, or None.

    Dune typically returns either an integer epoch (s, ms, or us)
    or an ISO-8601 string in the result rows.  We accept all
    three numeric scales by inspecting magnitude, and parse ISO
    via the stdlib :meth:`datetime.fromisoformat`.

    ``None`` is returned for unparseable / missing / non-positive
    values — the caller propagates that to ``OnChainMetric``,
    which permits ``observed_ts_ns=None``.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and (raw != raw or raw in (float("inf"), float("-inf"))):
            return None
        scaled = _scale_epoch_to_ns(float(raw))
        if scaled is None or scaled <= 0:
            return None
        return scaled
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        # Try ISO-8601 first.
        from datetime import UTC, datetime  # local import — INV-15

        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            try:
                ts_ns = int(dt.timestamp() * 1_000_000_000)
            except (OverflowError, ValueError):
                return None
            return ts_ns if ts_ns > 0 else None
        # Fall back to numeric-string epoch.
        try:
            scaled = _scale_epoch_to_ns(float(text))
        except ValueError:
            return None
        if scaled is None or scaled <= 0:
            return None
        return scaled
    return None


def _scale_epoch_to_ns(value: float) -> int | None:
    """Heuristically scale a numeric epoch to nanoseconds.

    Dune queries return timestamps at multiple resolutions
    depending on the function used in SQL.  The magnitude buckets
    are stable: seconds < 1e12, milliseconds < 1e15, microseconds
    < 1e18, nanoseconds ≥ 1e18.
    """
    if value <= 0:
        return None
    if value < 1e12:
        return int(value * 1_000_000_000)
    if value < 1e15:
        return int(value * 1_000_000)
    if value < 1e18:
        return int(value * 1_000)
    return int(value)


def parse_results_payload(
    payload: bytes | str,
    *,
    ts_ns: int,
    metric: str,
    value_field: str = "value",
    asset: str = "",
    unit: str = "",
    observed_ts_field: str | None = None,
    asset_field: str | None = None,
    source: str = SOURCE_TAG,
) -> tuple[OnChainMetric, ...]:
    """Project a Dune results JSON body into an ``OnChainMetric`` tuple.

    INV-15 (pure projection): every parameter is supplied by the
    caller — never derived from the payload itself, never from a
    system clock — so two replays with the same input produce
    byte-identical output.

    Args:
        payload: Raw JSON document from
            ``GET /execution/{id}/results``.
        ts_ns: Caller-supplied ingestion timestamp from
            :class:`system.time_source.TimeAuthority`.
        metric: Stable metric identifier stamped onto every row
            (e.g. ``"daily_active_addresses"``). The Dune row
            itself doesn't carry a metric name, so the spec
            supplies one.
        value_field: Column name in the Dune row that holds the
            numeric value (default ``"value"``).
        asset: Default asset tag stamped onto every row when
            ``asset_field`` does not resolve to a non-empty string.
        unit: Free-form unit hint (e.g. ``"USD"``, ``"count"``).
        observed_ts_field: Optional column name carrying a
            per-row provider timestamp.  When ``None`` (or when
            the column is missing / unparseable for a given row),
            ``observed_ts_ns`` is left as ``None`` on that row.
        asset_field: Optional column name carrying a per-row
            asset tag.  When the column resolves to a non-empty
            string, it overrides the ``asset`` parameter for that
            row.
        source: SCVS source tag.  Defaults to ``"DUNE"``.

    Rows missing the configured ``value_field`` are skipped
    silently (the parser never raises) — the caller's client
    increments its ``errors`` counter via the empty-tuple path.

    Returns:
        Tuple of ``OnChainMetric`` in the row order Dune
        delivered them.  Empty tuple on any parse failure.
    """
    if not metric:
        raise ValueError("parse_results_payload: metric must be non-empty")
    if not value_field:
        raise ValueError("parse_results_payload: value_field must be non-empty")

    doc = _decode_payload(payload)
    if not isinstance(doc, dict):
        return ()
    result = doc.get("result")
    if not isinstance(result, dict):
        return ()
    rows = result.get("rows")
    if not isinstance(rows, list):
        return ()

    out: list[OnChainMetric] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        value = _parse_value(entry.get(value_field))
        if value is None:
            # OnChainMetric requires a float — drop rows that don't
            # carry one rather than fabricating zero.
            continue
        row_asset = asset
        if asset_field is not None:
            raw_asset = entry.get(asset_field)
            if isinstance(raw_asset, str) and raw_asset:
                row_asset = raw_asset
        observed_ts_ns: int | None = None
        if observed_ts_field is not None:
            observed_ts_ns = _parse_observed_ts_ns(entry.get(observed_ts_field))
        try:
            out.append(
                OnChainMetric(
                    ts_ns=ts_ns,
                    source=source,
                    metric=metric,
                    value=value,
                    asset=row_asset,
                    unit=unit,
                    observed_ts_ns=observed_ts_ns,
                )
            )
        except ValueError:
            # OnChainMetric.__post_init__ would only raise on empty
            # source / metric — both are statically enforced above,
            # so this branch is defensive only.
            continue
    return tuple(out)


# ---------------------------------------------------------------------------
# Spec + status value types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DuneQuerySpec:
    """One Dune query the client should execute on each tick.

    Attributes:
        query_id: Numeric Dune query id (e.g. ``1234567``).
        metric: Stable metric tag stamped onto every emitted
            :class:`OnChainMetric`.
        value_field: Column name in the Dune row that holds the
            numeric value.  Defaults to ``"value"``.
        asset: Default asset tag.  Overridden per row by
            ``asset_field`` when set.
        unit: Free-form unit hint.
        observed_ts_field: Optional column carrying a per-row
            timestamp.  When set, the parser projects it onto
            ``OnChainMetric.observed_ts_ns``.
        asset_field: Optional column carrying a per-row asset
            override.
        parameters: Query-parameter substitutions passed verbatim
            to ``POST /query/{id}/execute`` as the JSON body's
            ``"query_parameters"`` map.  Ordered by sorted key so
            two specs with the same logical contents serialise
            byte-identically (INV-15).
        execution_timeout_s: Per-execution ceiling.  Defaults to
            :data:`DEFAULT_EXECUTION_TIMEOUT_S`.
        poll_interval_s: Status-poll cadence in seconds.
    """

    query_id: int
    metric: str
    value_field: str = "value"
    asset: str = ""
    unit: str = ""
    observed_ts_field: str | None = None
    asset_field: str | None = None
    parameters: Mapping[str, str] = ()  # type: ignore[assignment]
    execution_timeout_s: float = DEFAULT_EXECUTION_TIMEOUT_S
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S

    def __post_init__(self) -> None:
        _validate_query_id(self.query_id)
        if not self.metric:
            raise ValueError("DuneQuerySpec.metric must be non-empty")
        if not self.value_field:
            raise ValueError("DuneQuerySpec.value_field must be non-empty")
        if self.execution_timeout_s <= 0:
            raise ValueError("DuneQuerySpec.execution_timeout_s must be positive")
        if self.poll_interval_s <= 0:
            raise ValueError("DuneQuerySpec.poll_interval_s must be positive")
        if self.observed_ts_field is not None and not isinstance(self.observed_ts_field, str):
            raise TypeError("DuneQuerySpec.observed_ts_field must be a str or None")
        if self.asset_field is not None and not isinstance(self.asset_field, str):
            raise TypeError("DuneQuerySpec.asset_field must be a str or None")
        # Freeze parameters to a sorted-tuple-backed mapping so two
        # specs whose dicts differ only in iteration order serialise
        # to byte-identical request bodies.
        if isinstance(self.parameters, Mapping):
            items = tuple(sorted(self.parameters.items()))
        elif self.parameters == ():
            items = ()
        else:
            raise TypeError("DuneQuerySpec.parameters must be a mapping (or empty)")
        # Allowed because frozen=True still permits __post_init__ to
        # rewrite immutable fields via object.__setattr__.
        object.__setattr__(self, "parameters", dict(items))


def serialize_execute_body(spec: DuneQuerySpec) -> bytes:
    """Render the JSON body for ``POST /query/{id}/execute``.

    Byte-stable: keys are sorted, separators are canonical so two
    specs with the same logical parameter map serialise identically
    regardless of dict insertion order (INV-15).
    """
    if spec.parameters:
        body = {"query_parameters": dict(sorted(spec.parameters.items()))}
    else:
        body = {}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class DuneFeedStatus:
    """Snapshot of client health — exposed by status endpoints.

    Mirrors :class:`ui.feeds.fred_http.MacroFeedStatus` so the
    operator dashboard can render Dune health with the same
    widget shape.
    """

    running: bool
    source: str
    query_ids: tuple[int, ...]
    last_poll_ts_ns: int | None
    last_observation_ts_ns: int | None
    observations_received: int
    executions_started: int
    executions_completed: int
    polls: int
    errors: int


# ---------------------------------------------------------------------------
# Async client.
# ---------------------------------------------------------------------------


HTTPRequest = tuple[str, str, Mapping[str, str], bytes | None]
"""(method, url, headers, body_bytes_or_None) — what the client hands
to the injectable fetcher.  Body is ``None`` for GET calls.
"""


HTTPFetch = Callable[[HTTPRequest], Awaitable[bytes]]
"""Async (method, url, headers, body) → bytes callable.

Production wires :func:`_default_fetch`, which uses :mod:`urllib`
behind :func:`asyncio.to_thread`.  Tests inject a deterministic
fake that returns canned bytes.
"""


async def _default_fetch(request: HTTPRequest) -> bytes:
    """Default fetcher — stdlib-only.

    Lazy-imports :mod:`urllib.request` so the parser is usable
    without an HTTP client installed (lint, tests).  Runs the
    blocking call on a worker thread so the asyncio loop is not
    starved.
    """
    import urllib.request  # local import; stdlib

    method, url, headers, body = request

    def _blocking() -> bytes:
        req = urllib.request.Request(  # noqa: S310 - fixed https URL
            url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.read()

    return await asyncio.to_thread(_blocking)


class _SleepHandle(Protocol):
    """Minimal subset of :func:`asyncio.sleep` we depend on, for typing."""

    def __call__(self, seconds: float) -> Awaitable[None]: ...  # pragma: no cover


class DuneAnalyticsClient:
    """Run one Dune query end-to-end (execute → poll → fetch results).

    The client is **stateless across calls**: every invocation of
    :meth:`run_query_once` starts a fresh execution, polls it to a
    terminal state, fetches its results, and returns them as a
    tuple of :class:`OnChainMetric`.  Long-lived state — counters,
    last-poll timestamps — lives only on the client instance and
    is exposed via :meth:`status` for operator visibility.

    The constructor validates everything that can be validated
    statically (positive intervals, non-empty api key, dedup-by-id
    spec list) so misconfiguration fails at boot, not under load.
    """

    def __init__(
        self,
        sink: Callable[[OnChainMetric], None],
        *,
        api_key: str,
        specs: Sequence[DuneQuerySpec],
        clock_ns: Callable[[], int],
        fetch: HTTPFetch | None = None,
        sleep: _SleepHandle | None = None,
        base: str = DUNE_API_BASE,
        source: str = SOURCE_TAG,
    ) -> None:
        if not api_key:
            raise ValueError("DuneAnalyticsClient: api_key must be non-empty")
        if not source:
            raise ValueError("DuneAnalyticsClient: source must be non-empty")
        if not specs:
            raise ValueError("DuneAnalyticsClient: at least one DuneQuerySpec required")

        seen: set[int] = set()
        ordered: list[DuneQuerySpec] = []
        for spec in specs:
            if spec.query_id in seen:
                continue
            seen.add(spec.query_id)
            ordered.append(spec)

        self._sink = sink
        self._api_key = api_key
        self._specs: tuple[DuneQuerySpec, ...] = tuple(ordered)
        self._clock_ns = clock_ns
        self._fetch: HTTPFetch = fetch if fetch is not None else _default_fetch
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._base = base
        self._source = source

        self._running: bool = False
        self._last_poll_ts_ns: int | None = None
        self._last_observation_ts_ns: int | None = None
        self._observations_received: int = 0
        self._executions_started: int = 0
        self._executions_completed: int = 0
        self._polls: int = 0
        self._errors: int = 0

    @property
    def query_ids(self) -> tuple[int, ...]:
        return tuple(spec.query_id for spec in self._specs)

    @property
    def specs(self) -> tuple[DuneQuerySpec, ...]:
        return self._specs

    def status(self) -> DuneFeedStatus:
        """Return a frozen snapshot of client telemetry."""
        return DuneFeedStatus(
            running=self._running,
            source=self._source,
            query_ids=self.query_ids,
            last_poll_ts_ns=self._last_poll_ts_ns,
            last_observation_ts_ns=self._last_observation_ts_ns,
            observations_received=self._observations_received,
            executions_started=self._executions_started,
            executions_completed=self._executions_completed,
            polls=self._polls,
            errors=self._errors,
        )

    def _headers(self) -> dict[str, str]:
        return {
            DUNE_API_KEY_HEADER: self._api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "dixvision-v42.2/dune-adapter",
        }

    async def _start_execution(self, spec: DuneQuerySpec) -> str | None:
        url = make_execute_url(spec.query_id, base=self._base)
        body = serialize_execute_body(spec)
        try:
            raw = await self._fetch(("POST", url, self._headers(), body))
        except Exception as exc:  # noqa: BLE001 - log + telemetry only
            LOG.warning(
                "dune: execute failed query_id=%d err=%s",
                spec.query_id,
                exc.__class__.__name__,
            )
            self._errors += 1
            return None
        self._executions_started += 1
        execution_id = parse_execution_id(raw)
        if execution_id is None:
            LOG.warning(
                "dune: execute returned no execution_id query_id=%d",
                spec.query_id,
            )
            self._errors += 1
            return None
        return execution_id

    async def _poll_status(self, execution_id: str, spec: DuneQuerySpec) -> str | None:
        """Poll status until terminal or timeout.  Returns the final
        state, or ``None`` if the poll loop errored out.
        """
        url = make_status_url(execution_id, base=self._base)
        # Convert ns to seconds — the spec carries seconds and the
        # status counter is ns from the injected clock.
        deadline_ns = self._clock_ns() + int(spec.execution_timeout_s * 1_000_000_000)
        while True:
            self._polls += 1
            try:
                raw = await self._fetch(("GET", url, self._headers(), None))
            except Exception as exc:  # noqa: BLE001 - telemetry
                LOG.warning(
                    "dune: status failed execution_id=%s err=%s",
                    execution_id,
                    exc.__class__.__name__,
                )
                self._errors += 1
                return None
            state = parse_execution_status(raw)
            if state is None:
                self._errors += 1
                return None
            if state in TERMINAL_STATES:
                return state
            if self._clock_ns() >= deadline_ns:
                LOG.warning(
                    "dune: execution timed out execution_id=%s state=%s",
                    execution_id,
                    state,
                )
                self._errors += 1
                return None
            await self._sleep(spec.poll_interval_s)

    async def _fetch_results(
        self, execution_id: str, spec: DuneQuerySpec
    ) -> tuple[OnChainMetric, ...]:
        url = make_results_url(execution_id, base=self._base)
        try:
            raw = await self._fetch(("GET", url, self._headers(), None))
        except Exception as exc:  # noqa: BLE001 - telemetry
            LOG.warning(
                "dune: results failed execution_id=%s err=%s",
                execution_id,
                exc.__class__.__name__,
            )
            self._errors += 1
            return ()
        ts_ns = self._clock_ns()
        return parse_results_payload(
            raw,
            ts_ns=ts_ns,
            metric=spec.metric,
            value_field=spec.value_field,
            asset=spec.asset,
            unit=spec.unit,
            observed_ts_field=spec.observed_ts_field,
            asset_field=spec.asset_field,
            source=self._source,
        )

    async def run_query_once(self, spec: DuneQuerySpec) -> tuple[OnChainMetric, ...]:
        """Execute one Dune query end-to-end.

        Stages:

        1. ``POST /query/{id}/execute`` — kicks off execution.
           Failure → empty tuple, ``errors`` incremented.
        2. ``GET /execution/{id}/status`` — polled at the spec's
           ``poll_interval_s`` until the state is in
           :data:`TERMINAL_STATES` or the execution timeout
           expires.  Non-success terminal states → empty tuple.
        3. ``GET /execution/{id}/results`` — body projected via
           :func:`parse_results_payload`.

        Successful rows are fed into the sink in row order before
        the tuple is returned, so callers that subscribe via the
        sink see them at the same point in the call as callers
        that subscribe via the return value.

        ``last_poll_ts_ns`` and ``last_observation_ts_ns`` are
        updated from the injected clock — the caller's
        :class:`~system.time_source.TimeAuthority` is the only
        time source the client ever consults.
        """
        execution_id = await self._start_execution(spec)
        if execution_id is None:
            return ()
        state = await self._poll_status(execution_id, spec)
        if state != EXECUTION_STATE_COMPLETED:
            return ()
        self._executions_completed += 1
        rows = await self._fetch_results(execution_id, spec)
        self._last_poll_ts_ns = self._clock_ns()
        if rows:
            self._observations_received += len(rows)
            self._last_observation_ts_ns = self._last_poll_ts_ns
            for row in rows:
                self._sink(row)
        return rows

    async def run_all_specs_once(self) -> tuple[OnChainMetric, ...]:
        """Convenience: run every spec serially in registration order.

        Returns the flat concatenation of all per-spec results.
        Per-spec failures degrade gracefully — one bad query never
        blocks the rest.
        """
        self._running = True
        try:
            out: list[OnChainMetric] = []
            for spec in self._specs:
                out.extend(await self.run_query_once(spec))
            return tuple(out)
        finally:
            self._running = False
