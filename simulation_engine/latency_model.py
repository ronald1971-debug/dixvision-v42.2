# ADAPTED FROM: hftbacktest/hftbacktest/src/backtest/models/latencies.rs
# ADAPTED FROM: hftbacktest/py-hftbacktest/hftbacktest/binding.pyi
#   (ConstantLatency / IntpOrderLatency)
"""Latency models (S-02.2) — adapted from ``hftbacktest``.

The queue-time / round-trip-latency half of the S-02 ``hftbacktest``
adaptation pair. The price-impact half (S-02.1) lives in
:mod:`simulation_engine.slippage_model`.

What survives from upstream
---------------------------
* hftbacktest's two-leg latency model — ``entry_latency`` (submit → exchange
  acknowledge) + ``response_latency`` (exchange → local fill receipt) — is
  reproduced verbatim in :class:`LatencySample`. This is the canonical
  decomposition every hftbacktest backtest reports against and is the
  mental model adapter authors expect.
* The ``ConstantLatency`` shape from
  ``hftbacktest::backtest::models::latencies::ConstantLatency`` is the
  reference baseline (:class:`ConstantLatency`).
* The piecewise-linear ``IntpOrderLatency`` model from
  ``hftbacktest::backtest::models::latencies::IntpOrderLatency`` is
  reproduced as :class:`InterpolatedLatency`. It linearly interpolates
  between recorded ``(ts_ns, entry_latency_ns, response_latency_ns)``
  samples to match a historical-replay latency curve.

What is rewritten behind DIX contracts
--------------------------------------
* No ``hftbacktest`` pip dependency. We reproduce the algorithms in pure
  Python; ``NEW_PIP_DEPENDENCIES = ()``.
* No clock reads. Latency models *describe* delays; they don't measure
  the wall clock. Every method takes ``ts_ns`` as an explicit parameter
  (B-CLOCK / INV-15).
* No PRNG state. :class:`JitteredLatency` derives deterministic jitter
  from an explicit ``seed`` argument via a stateless
  splitmix64-style hash; replaying the same ``(ts_ns, seed)`` always
  yields the same sample to the bit.
* No global mutable state. Every concrete model is a frozen dataclass
  with NaN/inf-safe validators (PR #234 IEEE-754 ``not (x >= 0)``
  pattern); samples are frozen too.
* The :class:`LatencyModel` Protocol is the exclusive integration
  surface — every concrete model implements it; consumers depend on
  the Protocol, not the classes. New models can be added without
  touching callers.

Tier classification
-------------------
``simulation_engine/`` is **OFFLINE tier** per the master canonical
PART 1: it can use ML in future modules, must never be called from
``hot_path/`` directly, and only ever emits structured outputs the
meta-controller's scoring layer reads asynchronously.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

# This module reproduces hftbacktest's algorithms in plain Python, so
# we deliberately do not pull in the Rust-backed pip wheel.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Sample (the output type every model returns)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class LatencySample:
    """One frozen ``(entry, response)`` latency observation, in nanoseconds.

    Attributes:
        entry_latency_ns: Time from order-submit to exchange acknowledge
            (``>= 0``).
        response_latency_ns: Time from exchange acknowledge to local fill
            receipt (``>= 0``).

    Both fields are integers because hftbacktest itself reports latencies
    in integer nanoseconds, and using ``int`` everywhere on the latency
    side keeps replay byte-identical (no float-rounding drift across
    Python builds).
    """

    entry_latency_ns: int
    response_latency_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.entry_latency_ns, int):
            raise TypeError(
                "LatencySample.entry_latency_ns must be int, "
                f"got {type(self.entry_latency_ns).__name__}"
            )
        if not isinstance(self.response_latency_ns, int):
            raise TypeError(
                "LatencySample.response_latency_ns must be int, "
                f"got {type(self.response_latency_ns).__name__}"
            )
        if self.entry_latency_ns < 0:
            raise ValueError(
                f"LatencySample.entry_latency_ns must be >= 0, got {self.entry_latency_ns!r}"
            )
        if self.response_latency_ns < 0:
            raise ValueError(
                f"LatencySample.response_latency_ns must be >= 0, got {self.response_latency_ns!r}"
            )

    @property
    def round_trip_ns(self) -> int:
        """``entry + response`` — the full submit → fill round trip."""
        return self.entry_latency_ns + self.response_latency_ns


# ---------------------------------------------------------------------------
# Protocol (the exclusive integration surface)
# ---------------------------------------------------------------------------


@runtime_checkable
class LatencyModel(Protocol):
    """Latency Protocol — every concrete model implements this.

    Attributes:
        name: Short, stable identifier (logged into
            ``ExecutionEvent.meta["latency_model"]`` so audit replays
            can identify which model produced a given fill timing).

    The :meth:`sample` method is a *pure function* of its inputs plus
    the model's frozen config — no clock reads, no PRNG state, no IO.
    Replaying the same ``(ts_ns, seed)`` always returns the same
    :class:`LatencySample` to the bit (INV-15).
    """

    name: str

    def sample(self, ts_ns: int, seed: int = 0) -> LatencySample:
        """Return the latency sample for an order submitted at ``ts_ns``.

        ``seed`` is ignored by deterministic models (constant /
        interpolated) and consumed by jittered models. Callers should
        derive a stable, replayable seed from the order itself (e.g.
        ``hash((order_id, ts_ns)) & 0xFFFFFFFFFFFFFFFF``) so replays
        are byte-identical.
        """
        ...


# ---------------------------------------------------------------------------
# 1. Constant latency (the reference baseline; matches PaperBroker)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ConstantLatency:
    """Fixed ``(entry, response)`` latency — adapted from
    ``hftbacktest::backtest::models::latencies::ConstantLatency``.

    The reference baseline. Useful for unit tests, the smoke-test
    arena, and any backtest where a single measured latency
    approximates the venue. Output is independent of ``ts_ns`` and
    ``seed``.
    """

    entry_latency_ns: int = 0
    response_latency_ns: int = 0
    name: str = "constant_latency"

    def __post_init__(self) -> None:
        if self.entry_latency_ns < 0:
            raise ValueError(
                f"ConstantLatency.entry_latency_ns must be >= 0, got {self.entry_latency_ns!r}"
            )
        if self.response_latency_ns < 0:
            raise ValueError(
                "ConstantLatency.response_latency_ns must be >= 0, "
                f"got {self.response_latency_ns!r}"
            )

    def sample(self, ts_ns: int, seed: int = 0) -> LatencySample:
        del ts_ns, seed  # constant model — no inputs consumed
        return LatencySample(
            entry_latency_ns=self.entry_latency_ns,
            response_latency_ns=self.response_latency_ns,
        )


# ---------------------------------------------------------------------------
# 2. Interpolated latency (adapted from hftbacktest IntpOrderLatency)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class InterpolatedLatency:
    """Piecewise-linear interpolation of a recorded latency curve.

    Adapted from ``hftbacktest::backtest::models::latencies::IntpOrderLatency``.

    ``samples`` is a tuple of ``(ts_ns, entry_latency_ns,
    response_latency_ns)`` triplets, **strictly increasing in
    ``ts_ns``**. ``sample(ts_ns)`` linearly interpolates ``entry`` and
    ``response`` between the two surrounding recorded points; queries
    before the first sample or after the last clamp to the boundary
    values (matching hftbacktest's behaviour at the edges of the
    recorded window).

    Returned latencies are integers — interpolated floats are rounded
    half-to-even via Python's standard ``round`` so replays are
    byte-identical. Hand-validated to clamp at NaN-free, non-negative
    integer endpoints.
    """

    samples: tuple[tuple[int, int, int], ...]
    name: str = "interpolated_latency"

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("InterpolatedLatency.samples must contain at least one row")
        for i, row in enumerate(self.samples):
            if len(row) != 3:
                raise ValueError(
                    f"InterpolatedLatency.samples[{i}] must be a "
                    f"3-tuple (ts_ns, entry_ns, response_ns), "
                    f"got {row!r}"
                )
            ts_ns, entry, response = row
            if not (
                isinstance(ts_ns, int) and isinstance(entry, int) and isinstance(response, int)
            ):
                raise TypeError(f"InterpolatedLatency.samples[{i}] must be all int, got {row!r}")
            if entry < 0 or response < 0:
                raise ValueError(
                    f"InterpolatedLatency.samples[{i}] latencies must be >= 0, got {row!r}"
                )
            if i > 0 and ts_ns <= self.samples[i - 1][0]:
                raise ValueError(
                    "InterpolatedLatency.samples must be strictly "
                    "increasing in ts_ns; "
                    f"row {i} (ts_ns={ts_ns}) <= row {i - 1} "
                    f"(ts_ns={self.samples[i - 1][0]})"
                )

    def sample(self, ts_ns: int, seed: int = 0) -> LatencySample:
        del seed  # deterministic model — seed is not consumed
        rows = self.samples

        # Clamp left.
        if ts_ns <= rows[0][0]:
            return LatencySample(
                entry_latency_ns=rows[0][1],
                response_latency_ns=rows[0][2],
            )
        # Clamp right.
        if ts_ns >= rows[-1][0]:
            return LatencySample(
                entry_latency_ns=rows[-1][1],
                response_latency_ns=rows[-1][2],
            )

        # Locate the surrounding bracket via a linear scan
        # (hftbacktest does the same — sample tuples are typically
        # short and replay-deterministic order matters more than
        # asymptotic speed here; this is OFFLINE-tier work).
        for i in range(1, len(rows)):
            ts_hi = rows[i][0]
            if ts_ns < ts_hi:
                ts_lo = rows[i - 1][0]
                e_lo, r_lo = rows[i - 1][1], rows[i - 1][2]
                e_hi, r_hi = rows[i][1], rows[i][2]
                # Guarded by the strictly-increasing invariant.
                fraction = (ts_ns - ts_lo) / (ts_hi - ts_lo)
                entry = round(e_lo + (e_hi - e_lo) * fraction)
                response = round(r_lo + (r_hi - r_lo) * fraction)
                return LatencySample(
                    entry_latency_ns=int(entry),
                    response_latency_ns=int(response),
                )

        # Defensive — unreachable while the strictly-increasing
        # invariant holds.
        return LatencySample(
            entry_latency_ns=rows[-1][1],
            response_latency_ns=rows[-1][2],
        )


# ---------------------------------------------------------------------------
# 3. Jittered latency (deterministic per-call jitter, no PRNG state)
# ---------------------------------------------------------------------------

# splitmix64 constants from Vigna (2014). Public-domain finaliser used
# as a stateless deterministic hash so we don't carry PRNG state into
# OFFLINE-tier code.
_SPLITMIX64_GAMMA: int = 0x9E3779B97F4A7C15
_SPLITMIX64_M1: int = 0xBF58476D1CE4E5B9
_SPLITMIX64_M2: int = 0x94D049BB133111EB
_U64: int = 0xFFFFFFFFFFFFFFFF


def _splitmix64(x: int) -> int:
    """Stateless splitmix64 finaliser — deterministic, no global state."""
    x = (x + _SPLITMIX64_GAMMA) & _U64
    x = ((x ^ (x >> 30)) * _SPLITMIX64_M1) & _U64
    x = ((x ^ (x >> 27)) * _SPLITMIX64_M2) & _U64
    x ^= x >> 31
    return x & _U64


@dataclasses.dataclass(frozen=True, slots=True)
class JitteredLatency:
    """Constant base + deterministic uniform jitter.

    ``base.entry_latency_ns + uniform(0, max_jitter_entry_ns)`` (and
    similarly for response). The jitter is derived from a stateless
    splitmix64 mix of ``(ts_ns, seed)`` so replays are byte-identical
    — no PRNG instance, no global state, no clock.

    Useful for stress-testing: lets the strategy arena observe a
    realistic latency distribution while still being fully replayable
    (INV-15). Set the jitter caps to 0 to recover the exact
    :class:`ConstantLatency` baseline.
    """

    base: ConstantLatency
    max_jitter_entry_ns: int = 0
    max_jitter_response_ns: int = 0
    name: str = "jittered_latency"

    def __post_init__(self) -> None:
        if self.max_jitter_entry_ns < 0:
            raise ValueError(
                "JitteredLatency.max_jitter_entry_ns must be >= 0, "
                f"got {self.max_jitter_entry_ns!r}"
            )
        if self.max_jitter_response_ns < 0:
            raise ValueError(
                "JitteredLatency.max_jitter_response_ns must be >= 0, "
                f"got {self.max_jitter_response_ns!r}"
            )

    def sample(self, ts_ns: int, seed: int = 0) -> LatencySample:
        # Mix two independent draws so entry and response jitter aren't
        # perfectly correlated — matches hftbacktest's per-leg latency
        # sampling.
        h1 = _splitmix64((ts_ns ^ (seed << 1)) & _U64)
        h2 = _splitmix64((ts_ns ^ ((seed << 1) | 1)) & _U64)

        entry_jitter = h1 % (self.max_jitter_entry_ns + 1) if self.max_jitter_entry_ns > 0 else 0
        response_jitter = (
            h2 % (self.max_jitter_response_ns + 1) if self.max_jitter_response_ns > 0 else 0
        )
        return LatencySample(
            entry_latency_ns=self.base.entry_latency_ns + entry_jitter,
            response_latency_ns=(self.base.response_latency_ns + response_jitter),
        )


__all__ = [
    "ConstantLatency",
    "InterpolatedLatency",
    "JitteredLatency",
    "LatencyModel",
    "LatencySample",
    "NEW_PIP_DEPENDENCIES",
]
