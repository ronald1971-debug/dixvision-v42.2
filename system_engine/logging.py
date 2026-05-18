# ADAPTED FROM: https://github.com/hynek/structlog  (Apache-2.0 / MIT)
#
# Canonical structured-logging facade — OFFLINE_ONLY / RUNTIME_SAFE tier.
#
# NEW_PIP_DEPENDENCIES = ("structlog",)
#
# Authority constraints (pinned by tests/test_structured_logging.py):
#
#   * B1   — no imports from any runtime engine tier (intelligence_engine,
#            execution_engine, governance_engine, evolution_engine,
#            learning_engine).
#   * INV-15 — every log line is a pure function of its caller-supplied
#              ``ts_ns`` / ``engine_id`` / ``tier`` / ``event_kind`` / fields.
#              The renderer never calls ``time.time()``, ``time.monotonic()``,
#              ``datetime.now()`` or any wall-clock source.
#   * B27 / B28 / INV-71 — this module never constructs typed events; it is
#              a passive renderer.
#   * No top-level imports of :mod:`structlog`, :mod:`time`, :mod:`datetime`,
#     :mod:`random`, :mod:`asyncio`, :mod:`numpy`, :mod:`torch`,
#     :mod:`polars`, :mod:`requests`.
#
# Required line fields (canonical doc): ``ts_ns`` / ``engine_id`` / ``tier`` /
# ``event_kind``.  Forbidden keys (auto-scrubbed): credentials, api_key,
# secret, private_key, password, position_size, notional, balance.
"""Canonical structured-logging facade (I-04)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from system_engine.codec.json_codec import CodecError, canonical_dumps

LOGGER_VERSION: Final[str] = "v1.0-I04"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("structlog",)


class LogLevel(str, Enum):  # noqa: UP042 - str subclass needed for byte-stable JSON output
    """Canonical log levels (string-valued for byte-stable JSON output)."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


_LEVEL_RANK: Final[dict[LogLevel, int]] = {
    LogLevel.DEBUG: 10,
    LogLevel.INFO: 20,
    LogLevel.WARNING: 30,
    LogLevel.ERROR: 40,
    LogLevel.CRITICAL: 50,
}


# Canonical tier vocabulary — pinned by registry/engines.yaml.  Any value
# outside this set is rejected at line build time so misclassified tiers
# cannot leak into the ledger.
_ALLOWED_TIERS: Final[frozenset[str]] = frozenset(
    {
        "system",
        "governance",
        "execution",
        "intelligence",
        "evolution",
        "learning",
        "sensory",
        "state",
        "ui",
        "tools",
        "tests",
    }
)


# Keys that must never appear in a structured-log payload.  Even if a caller
# accidentally passes one, the scrubber redacts the value with the literal
# string ``"<redacted>"`` so the line shape stays stable.
_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "credentials",
        "credential",
        "api_key",
        "apikey",
        "secret",
        "private_key",
        "privatekey",
        "password",
        "position_size",
        "positionsize",
        "notional",
        "balance",
        "wallet_balance",
    }
)

_REDACTED_SENTINEL: Final[str] = "<redacted>"


class LoggingError(ValueError):
    """Raised when a log line is malformed (missing required field, bad tier)."""


@dataclass(frozen=True, slots=True)
class LogLine:
    """An immutable structured log line.

    Field order in the JSON output is fixed by ``canonical_dumps`` (alphabetic
    key sort), so two callers that pass the same fields produce byte-identical
    bytes — required by INV-15 replay.
    """

    ts_ns: int
    engine_id: str
    tier: str
    event_kind: str
    level: LogLevel
    fields: Mapping[str, Any]

    def to_bytes(self) -> bytes:
        """Render the line as a single UTF-8 JSON object (no trailing newline)."""

        payload: dict[str, Any] = {
            "ts_ns": self.ts_ns,
            "engine_id": self.engine_id,
            "tier": self.tier,
            "event_kind": self.event_kind,
            "level": self.level.value,
            "fields": dict(self.fields),
        }
        try:
            return canonical_dumps(payload)
        except CodecError as exc:
            raise LoggingError(f"non-canonicalisable log payload: {exc}") from exc


def _scrub(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Replace forbidden values with ``_REDACTED_SENTINEL`` (case-insensitive)."""

    out: dict[str, Any] = {}
    for key, value in fields.items():
        if not isinstance(key, str):
            raise LoggingError(f"log field keys must be str (got {type(key).__name__!r})")
        lowered = key.lower()
        if lowered in _FORBIDDEN_KEYS:
            out[key] = _REDACTED_SENTINEL
        elif isinstance(value, Mapping):
            out[key] = _scrub(value)
        else:
            out[key] = value
    return out


def build_log_line(
    *,
    ts_ns: int,
    engine_id: str,
    tier: str,
    event_kind: str,
    level: LogLevel | str = LogLevel.INFO,
    fields: Mapping[str, Any] | None = None,
) -> LogLine:
    """Validate inputs and return a frozen :class:`LogLine`.

    Required-field rules (canonical doc I-04):

    * ``ts_ns`` must be a non-negative ``int`` — callers pass the monotone
      timestamp from :class:`system.time_source.TimeAuthority`.  This module
      never reads wall-clock time itself.
    * ``engine_id`` must be a non-empty string identifying the producing
      engine (``"governance"`` / ``"execution"`` / ...).
    * ``tier`` must be a member of :data:`_ALLOWED_TIERS`.
    * ``event_kind`` must be a non-empty string identifying the structured
      event ("decision_signed" / "policy_drift_detected" / ...).
    """

    if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
        raise LoggingError(f"ts_ns must be int (got {type(ts_ns).__name__!r})")
    if ts_ns < 0:
        raise LoggingError(f"ts_ns must be non-negative (got {ts_ns})")
    if not isinstance(engine_id, str) or not engine_id:
        raise LoggingError("engine_id must be a non-empty string")
    if not isinstance(tier, str) or tier not in _ALLOWED_TIERS:
        raise LoggingError(f"tier must be one of {sorted(_ALLOWED_TIERS)} (got {tier!r})")
    if not isinstance(event_kind, str) or not event_kind:
        raise LoggingError("event_kind must be a non-empty string")

    if isinstance(level, str) and not isinstance(level, LogLevel):
        try:
            level_value = LogLevel(level)
        except ValueError as exc:
            raise LoggingError(f"unknown log level: {level!r}") from exc
    else:
        level_value = level

    scrubbed = _scrub(fields or {})
    return LogLine(
        ts_ns=ts_ns,
        engine_id=engine_id,
        tier=tier,
        event_kind=event_kind,
        level=level_value,
        fields=scrubbed,
    )


@dataclass(frozen=True, slots=True)
class StructuredLogger:
    """Sync, deterministic structured logger.

    ``sink`` is a caller-supplied callable that receives the rendered bytes —
    typically a wrapper around ``sys.stdout.buffer.write`` or a ledger writer
    seam.  The logger itself never opens files, never touches the network,
    and never reads wall-clock time.
    """

    engine_id: str
    tier: str
    sink: Callable[[bytes], None]
    min_level: LogLevel = LogLevel.INFO

    def __post_init__(self) -> None:
        if not isinstance(self.engine_id, str) or not self.engine_id:
            raise LoggingError("engine_id must be a non-empty string")
        if self.tier not in _ALLOWED_TIERS:
            raise LoggingError(f"tier must be one of {sorted(_ALLOWED_TIERS)}")
        if not callable(self.sink):
            raise LoggingError("sink must be callable")
        if self.min_level not in _LEVEL_RANK:
            raise LoggingError(f"min_level must be a LogLevel (got {self.min_level!r})")

    def log(
        self,
        *,
        ts_ns: int,
        event_kind: str,
        level: LogLevel | str = LogLevel.INFO,
        fields: Mapping[str, Any] | None = None,
    ) -> LogLine | None:
        """Build a :class:`LogLine`, write it to ``sink``, and return it.

        Returns ``None`` if ``level`` is below ``min_level`` (line is
        suppressed and ``sink`` is *not* invoked — INV-15 safe because the
        suppression decision is a pure function of ``min_level``).
        """

        line = build_log_line(
            ts_ns=ts_ns,
            engine_id=self.engine_id,
            tier=self.tier,
            event_kind=event_kind,
            level=level,
            fields=fields,
        )
        if _LEVEL_RANK[line.level] < _LEVEL_RANK[self.min_level]:
            return None
        self.sink(line.to_bytes())
        return line


def stdlib_logger_factory(
    *,
    engine_id: str,
    tier: str,
    sink: Callable[[bytes], None],
    min_level: LogLevel | str = LogLevel.INFO,
) -> StructuredLogger:
    """Return the always-available pure-stdlib logger."""

    if isinstance(min_level, str) and not isinstance(min_level, LogLevel):
        level = LogLevel(min_level)
    else:
        level = min_level
    return StructuredLogger(
        engine_id=engine_id,
        tier=tier,
        sink=sink,
        min_level=level,
    )


def enable_structlog_factory(
    *,
    engine_id: str,
    tier: str,
    min_level: LogLevel | str = LogLevel.INFO,
) -> Any:
    """Build a real :mod:`structlog`-backed logger.

    The :mod:`structlog` import happens *inside* this function body (never at
    module level) and is only triggered when the operator opts in.  The
    returned object is the bound structlog logger configured with a JSON
    renderer pipeline byte-identical to :func:`canonical_dumps`:

      * no wall-clock processors (no :class:`structlog.processors.TimeStamper`)
      * :func:`structlog.processors.add_log_level`
      * :func:`structlog.processors.JSONRenderer` with ``sort_keys=True``

    The structlog backend is operator-driven and stays cold until research
    acceptance + shadow-equivalence — matching every other lazy-seam adapter.
    """

    import structlog  # noqa: PLC0415 - intentional lazy seam

    if tier not in _ALLOWED_TIERS:
        raise LoggingError(f"tier must be one of {sorted(_ALLOWED_TIERS)}")
    if isinstance(min_level, str) and not isinstance(min_level, LogLevel):
        min_level = LogLevel(min_level)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_LEVEL_RANK[min_level]),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger().bind(engine_id=engine_id, tier=tier)


__all__: tuple[str, ...] = (
    "LOGGER_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "LogLevel",
    "LogLine",
    "LoggingError",
    "StructuredLogger",
    "build_log_line",
    "enable_structlog_factory",
    "stdlib_logger_factory",
)
