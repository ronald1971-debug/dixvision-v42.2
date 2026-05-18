# ADAPTED FROM: https://github.com/Delgan/loguru  (MIT)
#
# Canonical loguru-shape dev-only text logger — OFFLINE_ONLY / RUNTIME_SAFE tier.
#
# NEW_PIP_DEPENDENCIES = ("loguru",)
#
# Authority constraints (pinned by tests/test_dev_logger.py):
#
#   * B1   — no imports from any runtime engine tier (intelligence_engine,
#            execution_engine, governance_engine, evolution_engine,
#            learning_engine).
#   * INV-15 — every formatted line is a pure function of its caller-supplied
#              :class:`system_engine.logging.LogLine` plus a frozen
#              :class:`DevFormatConfig`.  The formatter never calls
#              ``time.time()``, ``time.monotonic()``, ``datetime.now()``
#              or any wall-clock source.
#   * B27 / B28 / INV-71 — this module never constructs typed events; it
#              is a passive renderer.
#   * No top-level imports of :mod:`loguru`, :mod:`time`, :mod:`datetime`,
#     :mod:`random`, :mod:`asyncio`, :mod:`os`, :mod:`numpy`, :mod:`torch`,
#     :mod:`polars`, :mod:`requests`.
#
# Production canonical logger is :mod:`system_engine.logging` (I-04).  This
# module is the dev-friendly companion that renders the SAME ``LogLine`` as
# a single-line colorised text string suitable for stdout / stderr / file
# tails.  ``loguru`` is wired as a lazy seam (function-local import only)
# in :func:`enable_loguru_dev_logger_factory`; the stdlib backend is the
# always-available production default.
"""Canonical dev-friendly text logger (I-05)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from system_engine.logging import (
    _ALLOWED_TIERS,
    LoggingError,
    LogLevel,
    LogLine,
    build_log_line,
)
from system_engine.logging import (
    LOGGER_VERSION as _STRUCT_LOGGER_VERSION,
)

DEV_LOGGER_VERSION: Final[str] = "v1.0-I05"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("loguru",)

# We keep the I-04 version reachable so callers that want to assert
# "dev-logger matches my structured-logger build" have a single anchor.
STRUCT_LOGGER_VERSION: Final[str] = _STRUCT_LOGGER_VERSION


_LEVEL_RANK: Final[dict[LogLevel, int]] = {
    LogLevel.DEBUG: 10,
    LogLevel.INFO: 20,
    LogLevel.WARNING: 30,
    LogLevel.ERROR: 40,
    LogLevel.CRITICAL: 50,
}


# ANSI SGR codes — minimal palette mirroring loguru's default level styling.
# Bytes are explicit so the formatter remains hermetic (no terminal
# auto-detection, no ``colorama`` dependency).
_ANSI_RESET: Final[str] = "\x1b[0m"
_LEVEL_ANSI: Final[dict[LogLevel, str]] = {
    LogLevel.DEBUG: "\x1b[2m",  # dim
    LogLevel.INFO: "\x1b[32m",  # green
    LogLevel.WARNING: "\x1b[33m",  # yellow
    LogLevel.ERROR: "\x1b[31m",  # red
    LogLevel.CRITICAL: "\x1b[1;31m",  # bright red
}


@dataclass(frozen=True, slots=True)
class DevFormatConfig:
    """Immutable formatter configuration.

    ``use_color`` toggles ANSI SGR escapes; off by default so file sinks
    and CI logs stay clean.  ``include_fields`` lets callers strip the
    structured tail when piping to a human-only tail.  ``key_sort`` mirrors
    the canonical JSON encoder — fields are sorted alphabetically so two
    runs produce byte-identical output (INV-15).
    """

    use_color: bool = False
    include_engine: bool = True
    include_tier: bool = True
    include_fields: bool = True
    key_sort: bool = True

    def __post_init__(self) -> None:
        for name in (
            "use_color",
            "include_engine",
            "include_tier",
            "include_fields",
            "key_sort",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise LoggingError(
                    f"DevFormatConfig.{name} must be bool (got {type(value).__name__})"
                )


def _format_value(value: Any) -> str:
    """Render a single field value as a stable string.

    Strings are wrapped in double quotes (with backslash-escaped quote and
    backslash).  Booleans / None render as their canonical JSON spellings.
    Numbers render via :func:`repr` so integers stay integer-shaped and
    floats keep their full precision.  Mappings and sequences render via
    :func:`repr` to keep the formatter pure — callers who want deep
    structured output should reach for the I-04 JSON renderer instead.
    """

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda kv: kv[0])
        rendered = ",".join(f"{k}={_format_value(v)}" for k, v in items)
        return "{" + rendered + "}"
    if isinstance(value, (list, tuple)):
        rendered = ",".join(_format_value(v) for v in value)
        return "[" + rendered + "]"
    return repr(value)


def format_dev_line(line: LogLine, config: DevFormatConfig) -> str:
    """Render a :class:`LogLine` as a single-line text string.

    Output shape (no trailing newline):

        ``ts_ns=<int> [<TIER>/<ENGINE>] <LEVEL> <event_kind> | k=v k=v ...``

    The leading ``ts_ns=`` prefix preserves the canonical doc rule that
    every log line surfaces the monotone timestamp first — replay tooling
    sorts by this column.  All optional sections are caller-controlled via
    :class:`DevFormatConfig`.
    """

    if not isinstance(line, LogLine):
        raise LoggingError(f"format_dev_line requires LogLine (got {type(line).__name__})")
    if not isinstance(config, DevFormatConfig):
        raise LoggingError(
            f"format_dev_line requires DevFormatConfig (got {type(config).__name__})"
        )

    parts: list[str] = [f"ts_ns={line.ts_ns}"]

    bracket_bits: list[str] = []
    if config.include_tier:
        bracket_bits.append(line.tier)
    if config.include_engine:
        bracket_bits.append(line.engine_id)
    if bracket_bits:
        parts.append("[" + "/".join(bracket_bits) + "]")

    if config.use_color:
        ansi = _LEVEL_ANSI[line.level]
        parts.append(f"{ansi}{line.level.value}{_ANSI_RESET}")
    else:
        parts.append(line.level.value)

    parts.append(line.event_kind)

    if config.include_fields and line.fields:
        items = list(line.fields.items())
        if config.key_sort:
            items.sort(key=lambda kv: kv[0])
        rendered = " ".join(f"{k}={_format_value(v)}" for k, v in items)
        return " ".join(parts) + " | " + rendered

    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class DevLogger:
    """Sync, deterministic dev-text logger.

    Mirrors :class:`system_engine.logging.StructuredLogger` but emits
    formatted *strings* to ``text_sink`` instead of canonical JSON bytes.
    The line is built via the I-04 :func:`build_log_line` validator so the
    redaction matrix + tier allowlist + required-field rules apply
    unchanged.
    """

    engine_id: str
    tier: str
    text_sink: Callable[[str], None]
    min_level: LogLevel = LogLevel.INFO
    config: DevFormatConfig = DevFormatConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.engine_id, str) or not self.engine_id:
            raise LoggingError("engine_id must be a non-empty string")
        if self.tier not in _ALLOWED_TIERS:
            raise LoggingError(f"tier must be one of {sorted(_ALLOWED_TIERS)}")
        if not callable(self.text_sink):
            raise LoggingError("text_sink must be callable")
        if self.min_level not in _LEVEL_RANK:
            raise LoggingError(f"min_level must be a LogLevel (got {self.min_level!r})")
        if not isinstance(self.config, DevFormatConfig):
            raise LoggingError("config must be a DevFormatConfig")

    def log(
        self,
        *,
        ts_ns: int,
        event_kind: str,
        level: LogLevel | str = LogLevel.INFO,
        fields: Mapping[str, Any] | None = None,
    ) -> LogLine | None:
        """Build, suppress-by-level, format and write.

        Returns the :class:`LogLine` that was actually written, or
        ``None`` if the line was suppressed by ``min_level``.
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
        self.text_sink(format_dev_line(line, self.config))
        return line


def stdlib_dev_logger_factory(
    *,
    engine_id: str,
    tier: str,
    text_sink: Callable[[str], None],
    min_level: LogLevel | str = LogLevel.INFO,
    config: DevFormatConfig | None = None,
) -> DevLogger:
    """Return the always-available pure-stdlib dev logger."""

    if isinstance(min_level, str) and not isinstance(min_level, LogLevel):
        level = LogLevel(min_level)
    else:
        level = min_level
    return DevLogger(
        engine_id=engine_id,
        tier=tier,
        text_sink=text_sink,
        min_level=level,
        config=config if config is not None else DevFormatConfig(),
    )


def enable_loguru_dev_logger_factory(
    *,
    engine_id: str,
    tier: str,
    min_level: LogLevel | str = LogLevel.INFO,
) -> Any:
    """Build a real :mod:`loguru`-backed dev logger.

    The :mod:`loguru` import happens *inside* this function body (never at
    module level) and is only triggered when the operator opts in.  The
    returned object is a thin shim that exposes a ``log(...)`` method
    matching :class:`DevLogger`'s shape, so production callers can swap
    backends behind the research-acceptance gate without API churn.

    Returns ``None`` when ``loguru`` is not installed — callers should
    fall back to :func:`stdlib_dev_logger_factory` in that case.
    """

    try:
        from loguru import logger as _loguru_logger  # noqa: F401
    except Exception:
        return None

    # Lazy import: re-import inside the body so the shim is closed over
    # the actual loguru module rather than the stdlib stub.
    from loguru import logger  # noqa: F811

    if isinstance(min_level, str) and not isinstance(min_level, LogLevel):
        level = LogLevel(min_level)
    else:
        level = min_level

    class _LoguruShim:
        """Adapter that routes :class:`DevLogger`-shape calls into loguru."""

        __slots__ = ("_logger", "_engine_id", "_tier", "_min_level")

        def __init__(self) -> None:
            self._logger = logger
            self._engine_id = engine_id
            self._tier = tier
            self._min_level = level

        def log(
            self,
            *,
            ts_ns: int,
            event_kind: str,
            level: LogLevel | str = LogLevel.INFO,
            fields: Mapping[str, Any] | None = None,
        ) -> LogLine | None:
            line = build_log_line(
                ts_ns=ts_ns,
                engine_id=self._engine_id,
                tier=self._tier,
                event_kind=event_kind,
                level=level,
                fields=fields,
            )
            if _LEVEL_RANK[line.level] < _LEVEL_RANK[self._min_level]:
                return None
            text = format_dev_line(line, DevFormatConfig())
            self._logger.log(line.level.value, text)
            return line

    return _LoguruShim()


__all__ = [
    "DEV_LOGGER_VERSION",
    "DevFormatConfig",
    "DevLogger",
    "NEW_PIP_DEPENDENCIES",
    "STRUCT_LOGGER_VERSION",
    "enable_loguru_dev_logger_factory",
    "format_dev_line",
    "stdlib_dev_logger_factory",
]
