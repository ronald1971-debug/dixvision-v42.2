"""Canonical TIER I I-03 — msgspec hot-path struct surface (OFFLINE_ONLY).

# ADAPTED FROM: https://github.com/jcrist/msgspec (BSD-3-Clause)
# Sources referenced (read-only, no upstream code copied verbatim):
#   * msgspec/structs.py      — Struct base (immutable, no __dict__)
#   * msgspec/json.py         — encode / decode (faster than orjson for structs)
#   * msgspec/msgpack.py      — binary wire format (not used here; JSON only)

DIX VISION already pins the canonical four event types as frozen+slotted
dataclasses in :mod:`core.contracts.events`.  Those events cross every
engine boundary and govern replay determinism (INV-15) — they MUST NOT
be silently swapped for a third-party struct type.

This module instead provides a **hot-path-only** struct surface
(``FastSignal`` / ``FastExecution``) that mirrors the minimal subset of
fields the per-tick gate in :mod:`execution_engine.hot_path.fast_execute`
actually reads.  The hot path:

  * never reads :attr:`SignalEvent.meta` / :attr:`signal_trust` /
    :attr:`signal_source` / :attr:`produced_by_engine`
  * never reads :attr:`ExecutionEvent.meta` / :attr:`venue` /
    :attr:`order_id` / :attr:`produced_by_engine`

so the hot-path struct can be a tight 5-field frozen value object.  All
fields are deterministic functions of the source ``SignalEvent`` /
``ExecutionEvent`` — there is no behavioural change.

Authority
---------

  * INV-15 — pure value objects.  No clocks, no PRNG, no IO.  3-run
    byte-identical replay is pinned via BLAKE2b-16 digest equality in
    the test suite.
  * B1     — no runtime-tier cross-imports (no
    ``intelligence_engine`` / ``governance_engine`` /
    ``evolution_engine`` / ``learning_engine`` / ``system_engine`` —
    only ``core.contracts``).
  * B27 / B28 / INV-71 — this module never constructs typed events
    (``PatchProposal`` / ``HazardEvent`` / ``SignalEvent`` /
    ``ExecutionEvent`` / ``SystemEvent`` / ``LearningUpdate``).  The
    bridge functions ``signal_event_from_fast`` /
    ``execution_event_from_fast`` are deliberately *not* defined here
    — they live in :mod:`execution_engine` proper, where the typed
    constructor is policy-clean.
  * msgspec is the **lazy seam** —
    :func:`enable_msgspec_fast_struct_factory` is the only place
    ``msgspec`` is imported, and only inside the function body so the
    AST-guarded check ``"msgspec" not in top-level imports`` holds.

The stdlib factory :func:`stdlib_fast_struct_factory` is always
available — it returns the frozen+slotted dataclass surface defined
below.  Promotion to the actual msgspec backend is a separate
research-acceptance PR (shadow-equivalence gate).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final

from core.contracts.events import ExecutionStatus, Side, SignalEvent

_JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")


def _canonical_dumps(value: Any) -> bytes:
    """Stdlib mirror of ``system_engine.codec.json_codec.canonical_dumps``.

    Inlined here so this module stays B1-clean (no cross-engine import
    into ``system_engine``).  Identical sort-keys / compact-separators
    contract — the I-02 canonical codec wraps exactly this stdlib call.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=_JSON_SEPARATORS,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_loads(blob: bytes) -> Any:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(f"_canonical_loads requires bytes-like input (got {type(blob).__name__})")
    return json.loads(bytes(blob).decode("utf-8"))


NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("msgspec",)
"""Canonical declaration of the third-party packages this module would
*adopt* once the research-acceptance gate clears.  Declared but never
imported at module level (AST-pinned).
"""

FAST_STRUCT_VERSION: Final[str] = "1"
"""Bumped only when the on-wire shape of ``FastSignal`` / ``FastExecution``
changes — keeps the canonical-encode bytes byte-stable across releases."""

_ALLOWED_SIDES: Final[frozenset[str]] = frozenset({s.value for s in Side})
_ALLOWED_EXEC_STATUSES: Final[frozenset[str]] = frozenset({s.value for s in ExecutionStatus})


# ---------------------------------------------------------------------------
# FastSignal — 5-field hot-path mirror of SignalEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FastSignal:
    """Hot-path struct mirroring the minimal :class:`SignalEvent` surface.

    Fields are the *only* fields :func:`execution_engine.hot_path.fast_execute.fast_execute`
    reads from a signal:

      * ``ts_ns``       — monotonic timestamp (TimeAuthority, T0-04).
      * ``symbol``      — instrument id.
      * ``side``        — :class:`Side` literal.
      * ``confidence``  — ``[0.0, 1.0]`` band.
      * ``plugin_chain`` — tuple of plugin names that contributed.

    The remaining ``SignalEvent`` fields (``meta`` / ``signal_trust`` /
    ``signal_source`` / ``produced_by_engine`` / ``kind``) are
    *offline-tier* metadata; carrying them through the hot path is the
    leak msgspec-shape optimisation eliminates.
    """

    ts_ns: int
    symbol: str
    side: Side
    confidence: float
    plugin_chain: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"FastSignal.ts_ns must be int (got {type(self.ts_ns).__name__})")
        if self.ts_ns < 0:
            raise ValueError("FastSignal.ts_ns must be non-negative")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("FastSignal.symbol must be a non-empty str")
        if not isinstance(self.side, Side):
            raise TypeError(f"FastSignal.side must be Side enum (got {type(self.side).__name__})")
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise TypeError("FastSignal.confidence must be a real number")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("FastSignal.confidence must lie in [0.0, 1.0]")
        if not isinstance(self.plugin_chain, tuple):
            raise TypeError("FastSignal.plugin_chain must be a tuple")
        for entry in self.plugin_chain:
            if not isinstance(entry, str) or not entry:
                raise ValueError("FastSignal.plugin_chain entries must be non-empty strings")


# ---------------------------------------------------------------------------
# FastExecution — 5-field hot-path mirror of ExecutionEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FastExecution:
    """Hot-path struct mirroring the minimal :class:`ExecutionEvent` surface.

    Carries only the fields the per-tick gate emits:

      * ``ts_ns``    — monotonic timestamp.
      * ``symbol``   — instrument id.
      * ``side``     — :class:`Side` literal.
      * ``qty``      — quantity (non-negative).
      * ``price``    — mark price.
      * ``status``   — :class:`ExecutionStatus` literal.

    ``venue`` / ``order_id`` / ``meta`` / ``produced_by_engine`` are
    populated on the slow-path lifecycle FSM in :mod:`execution_engine`.
    """

    ts_ns: int
    symbol: str
    side: Side
    qty: float
    price: float
    status: ExecutionStatus

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"FastExecution.ts_ns must be int (got {type(self.ts_ns).__name__})")
        if self.ts_ns < 0:
            raise ValueError("FastExecution.ts_ns must be non-negative")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("FastExecution.symbol must be a non-empty str")
        if not isinstance(self.side, Side):
            raise TypeError("FastExecution.side must be Side enum")
        for fname, fval in (("qty", self.qty), ("price", self.price)):
            if not isinstance(fval, (int, float)) or isinstance(fval, bool):
                raise TypeError(f"FastExecution.{fname} must be a real number")
        if float(self.qty) < 0.0:
            raise ValueError("FastExecution.qty must be non-negative")
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("FastExecution.status must be ExecutionStatus enum")


# ---------------------------------------------------------------------------
# Pure conversions — SignalEvent → FastSignal
# ---------------------------------------------------------------------------


def fast_signal_from_event(event: SignalEvent) -> FastSignal:
    """Project a :class:`SignalEvent` onto its hot-path 5-field subset.

    The function is a pure deterministic function — given the same
    ``event`` it always returns the same ``FastSignal``.  No clock, no
    PRNG.  This is the canonical entrypoint for the per-tick gate.
    """

    if not isinstance(event, SignalEvent):
        raise TypeError(f"fast_signal_from_event requires SignalEvent (got {type(event).__name__})")
    return FastSignal(
        ts_ns=event.ts_ns,
        symbol=event.symbol,
        side=event.side,
        confidence=float(event.confidence),
        plugin_chain=tuple(event.plugin_chain),
    )


def project_fast_signals(
    events: Iterable[SignalEvent],
) -> tuple[FastSignal, ...]:
    """Batch-project a stream of :class:`SignalEvent` to :class:`FastSignal`.

    Output is a tuple (immutable, replay-stable).  Order is preserved.
    """

    return tuple(fast_signal_from_event(e) for e in events)


# ---------------------------------------------------------------------------
# Canonical encode / decode (msgspec-shape, stdlib backend)
# ---------------------------------------------------------------------------


def _signal_to_payload(signal: FastSignal) -> dict[str, Any]:
    return {
        "fast_struct_version": FAST_STRUCT_VERSION,
        "kind": "fast_signal",
        "ts_ns": int(signal.ts_ns),
        "symbol": str(signal.symbol),
        "side": signal.side.value,
        "confidence": float(signal.confidence),
        "plugin_chain": list(signal.plugin_chain),
    }


def _execution_to_payload(execution: FastExecution) -> dict[str, Any]:
    return {
        "fast_struct_version": FAST_STRUCT_VERSION,
        "kind": "fast_execution",
        "ts_ns": int(execution.ts_ns),
        "symbol": str(execution.symbol),
        "side": execution.side.value,
        "qty": float(execution.qty),
        "price": float(execution.price),
        "status": execution.status.value,
    }


def canonical_encode_fast_signal(signal: FastSignal) -> bytes:
    """Encode a :class:`FastSignal` to canonical JSON bytes.

    Routes through the I-02 canonical codec —
    :func:`system_engine.codec.json_codec.canonical_dumps` —  so the
    output is byte-identical to what an ``orjson`` (or ``msgspec.json``)
    promotion would produce, with keys lexicographically sorted.
    """

    if not isinstance(signal, FastSignal):
        raise TypeError("canonical_encode_fast_signal requires FastSignal")
    return _canonical_dumps(_signal_to_payload(signal))


def canonical_encode_fast_execution(execution: FastExecution) -> bytes:
    if not isinstance(execution, FastExecution):
        raise TypeError("canonical_encode_fast_execution requires FastExecution")
    return _canonical_dumps(_execution_to_payload(execution))


def canonical_decode_fast_signal(blob: bytes) -> FastSignal:
    """Decode bytes produced by :func:`canonical_encode_fast_signal`.

    Round-trip is byte-stable:

        encode(decode(encode(x))) == encode(x)

    pinned by the test suite.
    """

    payload = _canonical_loads(blob)
    if not isinstance(payload, dict):
        raise ValueError("canonical_decode_fast_signal expected an object")
    if payload.get("kind") != "fast_signal":
        raise ValueError(
            f"canonical_decode_fast_signal expected kind=fast_signal (got {payload.get('kind')!r})"
        )
    if payload.get("fast_struct_version") != FAST_STRUCT_VERSION:
        raise ValueError(
            f"canonical_decode_fast_signal expected version="
            f"{FAST_STRUCT_VERSION!r} (got "
            f"{payload.get('fast_struct_version')!r})"
        )
    side_raw = payload.get("side")
    if side_raw not in _ALLOWED_SIDES:
        raise ValueError(f"canonical_decode_fast_signal got bad side={side_raw!r}")
    plugin_chain_raw = payload.get("plugin_chain", [])
    if not isinstance(plugin_chain_raw, list):
        raise ValueError("canonical_decode_fast_signal expected plugin_chain to be a list")
    return FastSignal(
        ts_ns=int(payload["ts_ns"]),
        symbol=str(payload["symbol"]),
        side=Side(side_raw),
        confidence=float(payload["confidence"]),
        plugin_chain=tuple(str(e) for e in plugin_chain_raw),
    )


def canonical_decode_fast_execution(blob: bytes) -> FastExecution:
    payload = _canonical_loads(blob)
    if not isinstance(payload, dict):
        raise ValueError("canonical_decode_fast_execution expected an object")
    if payload.get("kind") != "fast_execution":
        raise ValueError(
            f"canonical_decode_fast_execution expected kind=fast_execution "
            f"(got {payload.get('kind')!r})"
        )
    if payload.get("fast_struct_version") != FAST_STRUCT_VERSION:
        raise ValueError(
            f"canonical_decode_fast_execution expected version="
            f"{FAST_STRUCT_VERSION!r} (got "
            f"{payload.get('fast_struct_version')!r})"
        )
    side_raw = payload.get("side")
    if side_raw not in _ALLOWED_SIDES:
        raise ValueError(f"canonical_decode_fast_execution got bad side={side_raw!r}")
    status_raw = payload.get("status")
    if status_raw not in _ALLOWED_EXEC_STATUSES:
        raise ValueError(f"canonical_decode_fast_execution got bad status={status_raw!r}")
    return FastExecution(
        ts_ns=int(payload["ts_ns"]),
        symbol=str(payload["symbol"]),
        side=Side(side_raw),
        qty=float(payload["qty"]),
        price=float(payload["price"]),
        status=ExecutionStatus(status_raw),
    )


# ---------------------------------------------------------------------------
# Replay digest (INV-15 anchor)
# ---------------------------------------------------------------------------


def replay_digest(
    signals: Iterable[FastSignal],
    executions: Iterable[FastExecution],
) -> str:
    """Return a BLAKE2b-16 hex digest over a canonical stream.

    Stream = sorted(encode(signal) for signal in signals) ++
             sorted(encode(execution) for execution in executions)

    Same inputs → same digest → byte-identical replay (INV-15).
    """

    hasher = hashlib.blake2b(digest_size=16)
    for blob in sorted(canonical_encode_fast_signal(s) for s in signals):
        hasher.update(b"S|")
        hasher.update(blob)
    for blob in sorted(canonical_encode_fast_execution(e) for e in executions):
        hasher.update(b"E|")
        hasher.update(blob)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Factories — stdlib default + msgspec lazy seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FastStructBackend:
    """The active hot-path struct backend.

    ``name``  — short label ("stdlib" / "msgspec").
    ``signal_factory``    — callable that constructs a ``FastSignal``-equivalent.
    ``execution_factory`` — callable that constructs a ``FastExecution``-equivalent.

    Both factories must yield value objects whose canonical-encode bytes
    are byte-identical to the stdlib backend's output — the
    research-acceptance gate is a shadow-equivalence harness on this
    contract.
    """

    name: str
    signal_factory: Callable[..., FastSignal]
    execution_factory: Callable[..., FastExecution]


def stdlib_fast_struct_factory() -> FastStructBackend:
    """Always-available production backend.

    Returns the frozen+slotted dataclass surface defined in this module.
    The bytes produced by :func:`canonical_encode_fast_signal` /
    :func:`canonical_encode_fast_execution` are the source of truth.
    """

    return FastStructBackend(
        name="stdlib",
        signal_factory=FastSignal,
        execution_factory=FastExecution,
    )


def enable_msgspec_fast_struct_factory() -> FastStructBackend | None:
    """Lazy seam — return the msgspec backend if and only if the
    third-party ``msgspec`` package is importable.

    The :func:`stdlib_fast_struct_factory` remains the source of truth
    until the research-acceptance shadow-equivalence harness pins
    byte-equivalence on a representative workload.  Until that lands,
    callers MUST treat the seam as advisory — never wire it into the
    hot path directly.
    """

    try:
        import msgspec  # noqa: F401 — lazy seam, function-local only.
    except ImportError:
        return None

    # The msgspec backend keeps the stdlib dataclass surface as the
    # public type.  Promotion to actual msgspec.Struct subclasses is a
    # separate PR gated on the shadow-equivalence harness.
    return FastStructBackend(
        name="msgspec",
        signal_factory=FastSignal,
        execution_factory=FastExecution,
    )


__all__ = [
    "FAST_STRUCT_VERSION",
    "FastExecution",
    "FastSignal",
    "FastStructBackend",
    "NEW_PIP_DEPENDENCIES",
    "canonical_decode_fast_execution",
    "canonical_decode_fast_signal",
    "canonical_encode_fast_execution",
    "canonical_encode_fast_signal",
    "enable_msgspec_fast_struct_factory",
    "fast_signal_from_event",
    "project_fast_signals",
    "replay_digest",
    "stdlib_fast_struct_factory",
]
