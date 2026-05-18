"""B-10 — sentry-sdk → error telemetry.

# ADAPTED FROM: getsentry/sentry-python
#   - sentry_sdk/__init__.py — init(), capture_exception(), set_tag(), set_context()
#   - sentry_sdk/scope.py — Scope.set_tag, Scope.set_context, before_send hook
#   - sentry_sdk/integrations/fastapi.py — FastApiIntegration
#
# License: MIT.

Canonical adaptation of Sentry's error-telemetry primitives to the DIX
RUNTIME_SAFE tier with **strict data scrubbing**.  The official
``sentry_sdk`` package is lazy-imported *only* inside
:func:`sentry_telemetry_factory` — top-level imports remain empty so the
module is importable in replay / test environments where the dependency is
absent.

Design choices (vs. the upstream library):

* Event timestamps are **always caller-supplied**.  The module never reads
  a clock so INV-15 byte-identical replay holds across machines.
* ``event_id`` is derived from a caller-supplied
  ``(seed, ts_ns, exception_type, traceback_digest)`` tuple via splitmix64
  + BLAKE2b-8 — no randomness.
* Every outbound event is run through :func:`scrub_event` which strips
  financial data (positions, P&L, balances, fills, intents, prices,
  quantities, sizes, notionals, leverage, fees), credentials (API keys,
  tokens, passwords, secrets, signatures, mnemonics, seed phrases,
  private keys, addresses, bearer headers, cookies), and free-form
  bodies / contexts — only the exception class name, the cleaned
  traceback (file:line:func + class only, no locals), the DIX version
  tag, and the operator_id tag are emitted.
* :class:`ErrorTelemetry` is a runtime-checkable Protocol; both the pure
  :class:`InProcessErrorTelemetry` and the lazy Sentry wrapper satisfy it.
* The module **does not** construct typed bus events (``SignalEvent`` /
  ``ExecutionIntent`` / ``HazardEvent`` / ``GovernanceDecision`` /
  ``PatchProposal``) — B27 / B28 / INV-71 authority symmetry holds.

The module is RUNTIME_SAFE: it never blocks the hot path, never imports
``time``/``datetime``/``random``/``os``/``asyncio``/``socket`` at top
level, and never reaches into ``governance_engine`` / ``execution_engine``
/ ``evolution_engine`` (B1).
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("sentry-sdk",)
ERROR_TELEMETRY_ADAPTER_VERSION: str = "1"

MAX_TAG_VALUE_LEN: int = 200
MAX_TAG_COUNT: int = 32
MAX_FRAME_COUNT: int = 64
MAX_EVENT_BUFFER: int = 4_096
MAX_BREADCRUMB_BUFFER: int = 256
MAX_BREADCRUMB_MESSAGE_LEN: int = 200

DEFAULT_SAMPLE_RATIO: float = 1.0

_TAG_FALLBACK: str = "_"

# Forbidden key prefixes / substrings — case-insensitive substring match
# on event tag, context, or breadcrumb keys removes the field entirely.
SCRUB_KEY_FRAGMENTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "token",
    "bearer",
    "authorization",
    "auth",
    "cookie",
    "session",
    "signature",
    "private_key",
    "privatekey",
    "mnemonic",
    "seed_phrase",
    "seedphrase",
    "wallet",
    "address",
    "balance",
    "fund",
    "equity",
    "pnl",
    "profit",
    "loss",
    "position",
    "qty",
    "quantity",
    "size",
    "notional",
    "leverage",
    "margin",
    "price",
    "fill",
    "intent",
    "order",
    "fee",
    "commission",
    "slippage",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ErrorTelemetryError(RuntimeError):
    """Raised when the telemetry layer rejects an input or transport fails."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_SPLITMIX_MASK = (1 << 64) - 1


def _splitmix64(seed: int) -> int:
    z = (seed + 0x9E3779B97F4A7C15) & _SPLITMIX_MASK
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _SPLITMIX_MASK
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _SPLITMIX_MASK
    return z ^ (z >> 31)


def _is_scrubbable_key(key: str) -> bool:
    """Return True when *key* contains a forbidden fragment (case-insensitive)."""
    if not isinstance(key, str):
        return True
    if not key:
        return True
    lowered = key.lower()
    for fragment in SCRUB_KEY_FRAGMENTS:
        if fragment in lowered:
            return True
    return False


def _sanitize_tag_value(value: Any) -> str:
    """Coerce tag value to a Sentry-safe primitive string."""
    if value is None:
        return _TAG_FALLBACK
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if not text:
        return _TAG_FALLBACK
    cleaned_chars: list[str] = []
    for ch in text:
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            cleaned_chars.append("_")
        else:
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    if len(cleaned) > MAX_TAG_VALUE_LEN:
        cleaned = cleaned[:MAX_TAG_VALUE_LEN]
    return cleaned


def _canonicalise_tags(
    tags: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if tags is None:
        return ()
    if not isinstance(tags, Mapping):
        raise TypeError("tags must be a mapping")
    if len(tags) > MAX_TAG_COUNT:
        raise ValueError(f"tags must be ≤ {MAX_TAG_COUNT}, got {len(tags)}")
    pairs: list[tuple[str, str]] = []
    for key in sorted(tags):
        if not isinstance(key, str):
            raise TypeError("tag keys must be str")
        if not key:
            raise ValueError("tag keys must be non-empty")
        if _is_scrubbable_key(key):
            continue
        pairs.append((key, _sanitize_tag_value(tags[key])))
    return tuple(pairs)


# ---------------------------------------------------------------------------
# Frame / Traceback value objects
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, slots=True)
class Frame:
    """One scrubbed stack frame.

    Contains only filename, line number, function name, and (optional)
    qualified class name.  Local variables and frame bodies are never
    captured.
    """

    filename: str
    lineno: int
    function: str
    qualname: str

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename:
            raise ValueError("Frame.filename must be a non-empty str")
        if not isinstance(self.lineno, int) or self.lineno < 0:
            raise ValueError("Frame.lineno must be a non-negative int")
        if not isinstance(self.function, str) or not self.function:
            raise ValueError("Frame.function must be a non-empty str")
        if not isinstance(self.qualname, str):
            raise TypeError("Frame.qualname must be a str")


@dataclasses.dataclass(frozen=True, slots=True)
class ScrubbedTraceback:
    """Caller-supplied, scrubbed traceback representation."""

    exception_type: str
    exception_message_digest: str
    frames: tuple[Frame, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.exception_type, str) or not self.exception_type:
            raise ValueError("ScrubbedTraceback.exception_type must be non-empty")
        if not isinstance(self.exception_message_digest, str):
            raise TypeError("exception_message_digest must be a str")
        if len(self.exception_message_digest) != 16:
            raise ValueError("exception_message_digest must be a 16-hex BLAKE2b-8 digest")
        try:
            int(self.exception_message_digest, 16)
        except ValueError as exc:
            raise ValueError("exception_message_digest must be valid hex") from exc
        if not isinstance(self.frames, tuple):
            raise TypeError("frames must be a tuple")
        if len(self.frames) > MAX_FRAME_COUNT:
            raise ValueError(f"frames must be ≤ {MAX_FRAME_COUNT}, got {len(self.frames)}")
        for f in self.frames:
            if not isinstance(f, Frame):
                raise TypeError("frames must be Frame instances")


def exception_message_digest(message: str) -> str:
    """BLAKE2b-8 hex digest of *message* — caller projection for scrubbing.

    Sentry would normally upload the raw exception message which can leak
    financial PII.  DIX uploads only a 16-hex digest of the message so
    different sites of the same exception type still cluster but no
    plaintext leaks.
    """
    if not isinstance(message, str):
        raise TypeError("message must be a str")
    return hashlib.blake2b(message.encode("utf-8"), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# ErrorEvent value object
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, slots=True)
class ErrorEvent:
    """An immutable, scrubbed error event ready for transport."""

    event_id: str
    ts_ns: int
    operator_id: str
    dix_version: str
    environment: str
    exception_type: str
    exception_message_digest: str
    frames: tuple[Frame, ...]
    tags: tuple[tuple[str, str], ...]
    breadcrumb_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or len(self.event_id) != 16:
            raise ValueError("event_id must be a 16-hex BLAKE2b-8 digest")
        try:
            int(self.event_id, 16)
        except ValueError as exc:
            raise ValueError("event_id must be valid hex") from exc
        if not isinstance(self.ts_ns, int) or self.ts_ns < 0:
            raise ValueError("ts_ns must be a non-negative int")
        if not isinstance(self.operator_id, str) or not self.operator_id:
            raise ValueError("operator_id must be non-empty")
        if not isinstance(self.dix_version, str) or not self.dix_version:
            raise ValueError("dix_version must be non-empty")
        if not isinstance(self.environment, str) or not self.environment:
            raise ValueError("environment must be non-empty")
        if not isinstance(self.exception_type, str) or not self.exception_type:
            raise ValueError("exception_type must be non-empty")
        if not isinstance(self.exception_message_digest, str):
            raise TypeError("exception_message_digest must be a str")
        if len(self.exception_message_digest) != 16:
            raise ValueError("exception_message_digest must be a 16-hex BLAKE2b-8 digest")
        try:
            int(self.exception_message_digest, 16)
        except ValueError as exc:
            raise ValueError("exception_message_digest must be valid hex") from exc
        if not isinstance(self.frames, tuple):
            raise TypeError("frames must be a tuple")
        if len(self.frames) > MAX_FRAME_COUNT:
            raise ValueError(f"frames must be ≤ {MAX_FRAME_COUNT}, got {len(self.frames)}")
        for f in self.frames:
            if not isinstance(f, Frame):
                raise TypeError("frames must be Frame instances")
        if not isinstance(self.tags, tuple):
            raise TypeError("tags must be a tuple")
        for entry in self.tags:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
            ):
                raise TypeError("tags must be tuple[(str, str), ...]")
        if not isinstance(self.breadcrumb_digests, tuple):
            raise TypeError("breadcrumb_digests must be a tuple")
        for d in self.breadcrumb_digests:
            if not isinstance(d, str) or len(d) != 16:
                raise ValueError("breadcrumb_digests must be 16-hex BLAKE2b-8 digests")


@dataclasses.dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Deterministic snapshot of all captured events."""

    events: tuple[ErrorEvent, ...]
    captured: int
    dropped_by_sample: int
    dropped_by_buffer: int


# ---------------------------------------------------------------------------
# Breadcrumb value object
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, slots=True)
class Breadcrumb:
    """A scrubbed breadcrumb (operator action trail).

    The message is digested by the caller via
    :func:`exception_message_digest` so no plaintext is retained.
    """

    ts_ns: int
    category: str
    level: str
    message_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or self.ts_ns < 0:
            raise ValueError("ts_ns must be a non-negative int")
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("category must be non-empty")
        if _is_scrubbable_key(self.category):
            raise ValueError(f"category contains forbidden fragment: {self.category!r}")
        if not isinstance(self.level, str) or self.level not in {
            "debug",
            "info",
            "warning",
            "error",
            "fatal",
        }:
            raise ValueError("level must be one of debug/info/warning/error/fatal")
        if not isinstance(self.message_digest, str):
            raise TypeError("message_digest must be a str")
        if len(self.message_digest) != 16:
            raise ValueError("message_digest must be a 16-hex BLAKE2b-8 digest")
        try:
            int(self.message_digest, 16)
        except ValueError as exc:
            raise ValueError("message_digest must be valid hex") from exc


# ---------------------------------------------------------------------------
# Event scrubbing — before_send hook
# ---------------------------------------------------------------------------
def scrub_event(
    *,
    seed: int,
    ts_ns: int,
    operator_id: str,
    dix_version: str,
    environment: str,
    traceback: ScrubbedTraceback,
    tags: Mapping[str, Any] | None = None,
    breadcrumbs: Sequence[Breadcrumb] | None = None,
) -> ErrorEvent:
    """Project caller inputs into a fully scrubbed :class:`ErrorEvent`.

    This is the **single emission seam** — production code must always
    flow through this function before any sink receives an event.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if not isinstance(ts_ns, int) or ts_ns < 0:
        raise ValueError("ts_ns must be a non-negative int")
    if not isinstance(operator_id, str) or not operator_id:
        raise ValueError("operator_id must be non-empty")
    if not isinstance(dix_version, str) or not dix_version:
        raise ValueError("dix_version must be non-empty")
    if not isinstance(environment, str) or not environment:
        raise ValueError("environment must be non-empty")
    if not isinstance(traceback, ScrubbedTraceback):
        raise TypeError("traceback must be a ScrubbedTraceback")

    scrubbed_tags = _canonicalise_tags(tags)
    # Always pin DIX-managed tags
    pinned: dict[str, str] = {
        "dix_version": _sanitize_tag_value(dix_version),
        "operator_id": _sanitize_tag_value(operator_id),
        "environment": _sanitize_tag_value(environment),
    }
    merged: dict[str, str] = {}
    for k, v in scrubbed_tags:
        merged[k] = v
    for k, v in pinned.items():
        merged[k] = v
    sorted_tags = tuple(sorted(merged.items()))

    if breadcrumbs is None:
        crumb_digests: tuple[str, ...] = ()
    else:
        if not isinstance(breadcrumbs, Sequence):
            raise TypeError("breadcrumbs must be a Sequence")
        if len(breadcrumbs) > MAX_BREADCRUMB_BUFFER:
            raise ValueError(f"breadcrumbs must be ≤ {MAX_BREADCRUMB_BUFFER}")
        digests: list[str] = []
        for crumb in breadcrumbs:
            if not isinstance(crumb, Breadcrumb):
                raise TypeError("breadcrumbs must be Breadcrumb instances")
            digests.append(
                hashlib.blake2b(
                    b"|".join(
                        (
                            str(crumb.ts_ns).encode("ascii"),
                            crumb.category.encode("utf-8"),
                            crumb.level.encode("ascii"),
                            crumb.message_digest.encode("ascii"),
                        )
                    ),
                    digest_size=8,
                ).hexdigest()
            )
        crumb_digests = tuple(digests)

    # Derive event_id deterministically: splitmix(seed) ^ ts_ns hashed with
    # exception_type + traceback.frames + message digest + breadcrumbs
    mixed = _splitmix64(seed) ^ ts_ns
    h = hashlib.blake2b(mixed.to_bytes(8, "big", signed=False), digest_size=8)
    h.update(traceback.exception_type.encode("utf-8"))
    h.update(traceback.exception_message_digest.encode("ascii"))
    for f in traceback.frames:
        h.update(f.filename.encode("utf-8"))
        h.update(str(f.lineno).encode("ascii"))
        h.update(f.function.encode("utf-8"))
        h.update(f.qualname.encode("utf-8"))
    for cd in crumb_digests:
        h.update(cd.encode("ascii"))
    event_id = h.hexdigest()

    return ErrorEvent(
        event_id=event_id,
        ts_ns=ts_ns,
        operator_id=_sanitize_tag_value(operator_id),
        dix_version=_sanitize_tag_value(dix_version),
        environment=_sanitize_tag_value(environment),
        exception_type=traceback.exception_type,
        exception_message_digest=traceback.exception_message_digest,
        frames=traceback.frames,
        tags=sorted_tags,
        breadcrumb_digests=crumb_digests,
    )


# ---------------------------------------------------------------------------
# ErrorTelemetry Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class ErrorTelemetry(Protocol):
    """Caller-driven, clock-free, fully scrubbed error-telemetry sink."""

    def capture(self, event: ErrorEvent) -> None: ...

    def add_breadcrumb(self, crumb: Breadcrumb) -> None: ...

    def snapshot(self) -> TelemetrySnapshot: ...


def _mask_to_unit_interval(value: int) -> float:
    return (value & _SPLITMIX_MASK) / float(1 << 64)


# ---------------------------------------------------------------------------
# Pure-Python in-process telemetry (default sink, replay-safe)
# ---------------------------------------------------------------------------
class InProcessErrorTelemetry:
    """Append-only in-process telemetry sink.

    * Thread-safe via a single ``threading.Lock``.
    * Events are kept only when the BLAKE2b-8 sampling oracle over
      ``event_id`` falls below ``sample_ratio`` (deterministic per event).
    * Buffer is bounded at :data:`MAX_EVENT_BUFFER`; once full, additional
      events are dropped and counted.
    * Breadcrumbs are kept in a bounded ring buffer
      (:data:`MAX_BREADCRUMB_BUFFER`).
    """

    def __init__(
        self,
        *,
        sample_ratio: float = DEFAULT_SAMPLE_RATIO,
        event_buffer_size: int = MAX_EVENT_BUFFER,
        breadcrumb_buffer_size: int = MAX_BREADCRUMB_BUFFER,
    ) -> None:
        if not isinstance(sample_ratio, (int, float)):
            raise TypeError("sample_ratio must be numeric")
        sr = float(sample_ratio)
        if not (0.0 <= sr <= 1.0):
            raise ValueError("sample_ratio must be in [0.0, 1.0]")
        if not isinstance(event_buffer_size, int) or event_buffer_size <= 0:
            raise ValueError("event_buffer_size must be a positive int")
        if not isinstance(breadcrumb_buffer_size, int) or breadcrumb_buffer_size <= 0:
            raise ValueError("breadcrumb_buffer_size must be a positive int")
        self._sample_ratio = sr
        self._event_buffer_size = event_buffer_size
        self._breadcrumb_buffer_size = breadcrumb_buffer_size
        self._lock = threading.Lock()
        self._events: list[ErrorEvent] = []
        self._crumbs: list[Breadcrumb] = []
        self._captured: int = 0
        self._dropped_by_sample: int = 0
        self._dropped_by_buffer: int = 0

    @property
    def sample_ratio(self) -> float:
        return self._sample_ratio

    @property
    def event_buffer_size(self) -> int:
        return self._event_buffer_size

    @property
    def breadcrumb_buffer_size(self) -> int:
        return self._breadcrumb_buffer_size

    def _is_sampled(self, event_id: str) -> bool:
        if self._sample_ratio >= 1.0:
            return True
        if self._sample_ratio <= 0.0:
            return False
        digest = hashlib.blake2b(event_id.encode("ascii"), digest_size=8).digest()
        oracle = int.from_bytes(digest, "big") / float(1 << 64)
        return oracle < self._sample_ratio

    def capture(self, event: ErrorEvent) -> None:
        if not isinstance(event, ErrorEvent):
            raise TypeError("event must be an ErrorEvent")
        with self._lock:
            if not self._is_sampled(event.event_id):
                self._dropped_by_sample += 1
                return
            if len(self._events) >= self._event_buffer_size:
                self._dropped_by_buffer += 1
                return
            self._events.append(event)
            self._captured += 1

    def add_breadcrumb(self, crumb: Breadcrumb) -> None:
        if not isinstance(crumb, Breadcrumb):
            raise TypeError("crumb must be a Breadcrumb")
        with self._lock:
            self._crumbs.append(crumb)
            if len(self._crumbs) > self._breadcrumb_buffer_size:
                # Drop oldest crumb to keep buffer bounded.
                del self._crumbs[0]

    def breadcrumbs(self) -> tuple[Breadcrumb, ...]:
        with self._lock:
            return tuple(self._crumbs)

    def snapshot(self) -> TelemetrySnapshot:
        with self._lock:
            sorted_events = tuple(
                sorted(
                    self._events,
                    key=lambda e: (e.ts_ns, e.event_id),
                )
            )
            return TelemetrySnapshot(
                events=sorted_events,
                captured=self._captured,
                dropped_by_sample=self._dropped_by_sample,
                dropped_by_buffer=self._dropped_by_buffer,
            )


# ---------------------------------------------------------------------------
# Lazy Sentry SDK factory
# ---------------------------------------------------------------------------
def sentry_telemetry_factory(
    *,
    dsn: str,
    environment: str,
    dix_version: str,
    operator_id: str,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
) -> ErrorTelemetry:
    """Construct a Sentry-backed :class:`ErrorTelemetry` (lazy import).

    The Sentry SDK is **only** imported inside this factory body so the
    module remains importable in replay / offline environments where
    ``sentry-sdk`` is absent.  The returned object wraps the SDK behind
    the :class:`ErrorTelemetry` Protocol so callers never see the raw
    Sentry types.
    """
    if not isinstance(dsn, str) or not dsn:
        raise ValueError("dsn must be non-empty")
    if not isinstance(environment, str) or not environment:
        raise ValueError("environment must be non-empty")
    if not isinstance(dix_version, str) or not dix_version:
        raise ValueError("dix_version must be non-empty")
    if not isinstance(operator_id, str) or not operator_id:
        raise ValueError("operator_id must be non-empty")
    if not isinstance(sample_ratio, (int, float)):
        raise TypeError("sample_ratio must be numeric")
    sr = float(sample_ratio)
    if not (0.0 <= sr <= 1.0):
        raise ValueError("sample_ratio must be in [0.0, 1.0]")

    try:
        import sentry_sdk  # noqa: PLC0415
    except ImportError as exc:
        raise ErrorTelemetryError(
            "sentry-sdk not installed; install sentry-sdk to use sentry_telemetry_factory"
        ) from exc

    def _before_send(
        event: Mapping[str, Any], _hint: Mapping[str, Any] | None
    ) -> Mapping[str, Any] | None:
        """Final scrub — strips any field whose key matches a forbidden fragment.

        Sentry SDK calls this hook before transport.  We defensively scrub
        again here even though all DIX events already flowed through
        :func:`scrub_event`, because the SDK may have added contexts of
        its own (request body, breadcrumbs, locals).
        """
        cleaned: dict[str, Any] = {}
        for key, value in dict(event).items():
            if _is_scrubbable_key(key):
                continue
            if isinstance(value, Mapping):
                inner = {ik: iv for ik, iv in value.items() if not _is_scrubbable_key(str(ik))}
                cleaned[key] = inner
            else:
                cleaned[key] = value
        # Strip any free-form bodies — only structured fields survive.
        for forbidden in (
            "request",
            "response",
            "extra",
            "user",
            "breadcrumbs",
            "modules",
            "server_name",
        ):
            cleaned.pop(forbidden, None)
        return cleaned

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=dix_version,
        sample_rate=sr,
        traces_sample_rate=0.0,
        send_default_pii=False,
        attach_stacktrace=False,
        before_send=_before_send,
    )

    return _SentryAdapter(
        operator_id=operator_id,
        dix_version=dix_version,
        environment=environment,
        sentry_module=sentry_sdk,
    )


@dataclasses.dataclass(slots=True)
class _SentryAdapter:
    """Internal wrapper that bridges :class:`ErrorTelemetry` to ``sentry_sdk``.

    Stays inside the module so callers never depend on ``sentry_sdk``
    types directly.
    """

    operator_id: str
    dix_version: str
    environment: str
    sentry_module: Any

    def capture(self, event: ErrorEvent) -> None:
        if not isinstance(event, ErrorEvent):
            raise TypeError("event must be an ErrorEvent")
        scope_fn = self.sentry_module.push_scope
        with scope_fn() as scope:
            for tag_key, tag_value in event.tags:
                scope.set_tag(tag_key, tag_value)
            scope.set_tag("dix_event_id", event.event_id)
            scope.set_tag("dix_exception_type", event.exception_type)
            scope.set_tag(
                "dix_exception_message_digest",
                event.exception_message_digest,
            )
            self.sentry_module.capture_message(
                f"{event.exception_type}#{event.exception_message_digest}",
                level="error",
            )

    def add_breadcrumb(self, crumb: Breadcrumb) -> None:
        if not isinstance(crumb, Breadcrumb):
            raise TypeError("crumb must be a Breadcrumb")
        self.sentry_module.add_breadcrumb(
            category=crumb.category,
            level=crumb.level,
            message=f"digest:{crumb.message_digest}",
        )

    def snapshot(self) -> TelemetrySnapshot:
        # The Sentry SDK does not expose a local buffer — return empty
        # snapshot so the Protocol is still satisfied.
        return TelemetrySnapshot(
            events=(),
            captured=0,
            dropped_by_sample=0,
            dropped_by_buffer=0,
        )


# ---------------------------------------------------------------------------
# Frame projection helper
# ---------------------------------------------------------------------------
def project_frames(
    raw_frames: Iterable[Mapping[str, Any]],
) -> tuple[Frame, ...]:
    """Project caller-supplied raw frame dicts into scrubbed :class:`Frame`.

    Each input mapping must provide:
    ``filename`` (str), ``lineno`` (int), ``function`` (str),
    ``qualname`` (str, may be empty).  No other keys are read — this
    prevents accidental local-variable leakage.
    """
    if not isinstance(raw_frames, Iterable):
        raise TypeError("raw_frames must be an iterable of mappings")
    projected: list[Frame] = []
    for raw in raw_frames:
        if not isinstance(raw, Mapping):
            raise TypeError("each frame must be a mapping")
        filename = raw.get("filename")
        lineno = raw.get("lineno")
        function = raw.get("function")
        qualname = raw.get("qualname", "")
        if not isinstance(filename, str):
            raise TypeError("filename must be a str")
        if not isinstance(lineno, int):
            raise TypeError("lineno must be an int")
        if not isinstance(function, str):
            raise TypeError("function must be a str")
        if not isinstance(qualname, str):
            raise TypeError("qualname must be a str")
        projected.append(
            Frame(
                filename=filename,
                lineno=lineno,
                function=function,
                qualname=qualname,
            )
        )
        if len(projected) > MAX_FRAME_COUNT:
            raise ValueError(f"frames must be ≤ {MAX_FRAME_COUNT}, got > {len(projected)}")
    return tuple(projected)


__all__ = [
    "Breadcrumb",
    "DEFAULT_SAMPLE_RATIO",
    "ERROR_TELEMETRY_ADAPTER_VERSION",
    "ErrorEvent",
    "ErrorTelemetry",
    "ErrorTelemetryError",
    "Frame",
    "InProcessErrorTelemetry",
    "MAX_BREADCRUMB_BUFFER",
    "MAX_BREADCRUMB_MESSAGE_LEN",
    "MAX_EVENT_BUFFER",
    "MAX_FRAME_COUNT",
    "MAX_TAG_COUNT",
    "MAX_TAG_VALUE_LEN",
    "NEW_PIP_DEPENDENCIES",
    "SCRUB_KEY_FRAGMENTS",
    "ScrubbedTraceback",
    "TelemetrySnapshot",
    "exception_message_digest",
    "project_frames",
    "scrub_event",
    "sentry_telemetry_factory",
]
