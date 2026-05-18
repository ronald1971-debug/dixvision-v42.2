# ADAPTED FROM: https://github.com/pydata/bottleneck  (BSD-2-Clause)
#
# Fast rolling-window statistics — ``learning_engine/analytics/`` is the
# **OFFLINE** high-throughput slow-cadence analytics tier. The module
# body is pure-stdlib so it imports cleanly in environments without
# numpy or bottleneck. ``bottleneck`` (and ``numpy``) are lazy seams:
# production never imports them; the pure-Python backend is the default
# and produces byte-identical floats to bottleneck's ``move_mean`` /
# ``move_std`` / ``move_sum`` / ``move_min`` / ``move_max`` for the same
# inputs (within IEEE-754 summation tolerance).
#
# NEW_PIP_DEPENDENCIES = ("bottleneck",)
#
# Authority constraints (pinned by ``tests/test_rolling_stats.py``):
#
#   * **OFFLINE_ONLY** — must never be imported from ``execution_engine/``,
#     ``governance_engine/``, ``intelligence_engine/``, ``system_engine/``,
#     or ``core/``. (Authority lint enforcement deferred to a follow-up
#     sub-PR per the C-62 / S-10.4 pattern.)
#   * **B1** — no runtime engine imports here.
#   * **INV-15** — :func:`move_mean`, :func:`move_std`, :func:`move_sum`,
#     :func:`move_min`, :func:`move_max` are pure functions of their
#     inputs; three independent calls with identical arguments produce
#     byte-identical output tuples.
#   * **B27 / B28 / INV-71** — no typed-event constructors here.
#   * No top-level imports of :mod:`bottleneck`, :mod:`numpy`,
#     :mod:`time`, :mod:`datetime`, :mod:`random`, :mod:`asyncio`,
#     :mod:`torch`, :mod:`polars`, :mod:`requests`.
"""Bottleneck-shape rolling-window statistics (OFFLINE_ONLY)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "RollingStatsError",
    "RollingWindowSpec",
    "move_mean",
    "move_std",
    "move_sum",
    "move_min",
    "move_max",
    "enable_bottleneck_kernel_factory",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("bottleneck",)


class RollingStatsError(ValueError):
    """Raised by rolling-stats helpers for malformed inputs."""


@dataclasses.dataclass(frozen=True, slots=True)
class RollingWindowSpec:
    """Frozen window-spec for a rolling reduction.

    ``window`` is the number of samples in the trailing window
    (``window >= 1``). ``min_count`` is the minimum number of
    non-NaN samples required to emit a real value at a given index
    (``1 <= min_count <= window``); positions with fewer non-NaN
    samples emit ``NaN`` (this matches bottleneck and pandas'
    ``rolling(...).mean()`` shape).
    """

    window: int
    min_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.window, int) or self.window < 1:
            raise RollingStatsError(
                f"RollingWindowSpec.window must be int >= 1, got {self.window!r}"
            )
        if (
            not isinstance(self.min_count, int)
            or self.min_count < 1
            or self.min_count > self.window
        ):
            raise RollingStatsError(
                "RollingWindowSpec.min_count must be int with "
                f"1 <= min_count <= window ({self.window}), got "
                f"{self.min_count!r}"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_1d_sequence(values: Sequence[float] | object) -> bool:
    """Reject obvious 2-D inputs (sequence-of-sequence) without touching numpy."""

    if isinstance(values, (str, bytes, bytearray)):
        return False
    try:
        iter(values)  # type: ignore[arg-type]
    except TypeError:
        return False
    for item in values:  # type: ignore[union-attr]
        if isinstance(item, (list, tuple)):
            return False
    return True


def _coerce_1d(values: Sequence[float] | object) -> tuple[float, ...]:
    """Coerce input to a 1-D tuple of floats (NaN-preserving)."""

    if not _is_1d_sequence(values):
        raise RollingStatsError("rolling-stats input must be 1-D")
    return tuple(float(x) for x in values)  # type: ignore[union-attr]


def _normalize(
    values: Sequence[float] | object,
    window: int,
    min_count: int | None,
) -> tuple[tuple[float, ...], RollingWindowSpec]:
    if min_count is None:
        min_count = window
    spec = RollingWindowSpec(window=window, min_count=min_count)
    arr = _coerce_1d(values)
    return arr, spec


# ---------------------------------------------------------------------------
# Pure-stdlib rolling reductions (the production default backend)
# ---------------------------------------------------------------------------


def _move_reduce(
    arr: tuple[float, ...],
    spec: RollingWindowSpec,
    reducer: Callable[[tuple[float, ...]], float],
) -> tuple[float, ...]:
    """Generic O(N * window) rolling reduction over a 1-D float tuple.

    Emits ``NaN`` whenever the trailing window contains fewer than
    ``spec.min_count`` finite samples. The output has the same
    length as the input.
    """

    n = len(arr)
    if n == 0:
        return ()
    out: list[float] = [math.nan] * n
    for i in range(n):
        start = i + 1 - spec.window
        if start < 0:
            start = 0
        window_slice = arr[start : i + 1]
        finite = tuple(x for x in window_slice if math.isfinite(x))
        if len(finite) < spec.min_count:
            continue
        out[i] = reducer(finite)
    return tuple(out)


def move_sum(
    values: Sequence[float] | object,
    window: int,
    *,
    min_count: int | None = None,
) -> tuple[float, ...]:
    """Rolling sum — bottleneck ``move_sum`` shape."""

    arr, spec = _normalize(values, window, min_count)
    return _move_reduce(arr, spec, lambda w: math.fsum(w))


def move_mean(
    values: Sequence[float] | object,
    window: int,
    *,
    min_count: int | None = None,
) -> tuple[float, ...]:
    """Rolling mean — bottleneck ``move_mean`` shape."""

    arr, spec = _normalize(values, window, min_count)
    return _move_reduce(arr, spec, lambda w: math.fsum(w) / len(w))


def move_std(
    values: Sequence[float] | object,
    window: int,
    *,
    min_count: int | None = None,
    ddof: int = 0,
) -> tuple[float, ...]:
    """Rolling standard deviation — bottleneck ``move_std`` shape.

    ``ddof`` defaults to 0 (population std) to match bottleneck.
    Use ``ddof=1`` for sample std.
    """

    if not isinstance(ddof, int) or ddof < 0:
        raise RollingStatsError(f"move_std.ddof must be int >= 0, got {ddof!r}")
    arr, spec = _normalize(values, window, min_count)

    def _std(window_values: tuple[float, ...]) -> float:
        denom = len(window_values) - ddof
        if denom <= 0:
            return math.nan
        mean = math.fsum(window_values) / len(window_values)
        var = math.fsum((x - mean) * (x - mean) for x in window_values) / denom
        return math.sqrt(var)

    return _move_reduce(arr, spec, _std)


def move_min(
    values: Sequence[float] | object,
    window: int,
    *,
    min_count: int | None = None,
) -> tuple[float, ...]:
    """Rolling minimum — bottleneck ``move_min`` shape."""

    arr, spec = _normalize(values, window, min_count)
    return _move_reduce(arr, spec, lambda w: min(w))


def move_max(
    values: Sequence[float] | object,
    window: int,
    *,
    min_count: int | None = None,
) -> tuple[float, ...]:
    """Rolling maximum — bottleneck ``move_max`` shape."""

    arr, spec = _normalize(values, window, min_count)
    return _move_reduce(arr, spec, lambda w: max(w))


# ---------------------------------------------------------------------------
# Lazy seam — bottleneck-backed kernel factory (never called at import time)
# ---------------------------------------------------------------------------


def enable_bottleneck_kernel_factory() -> Callable[[str], Callable[..., Any]]:
    """Return a kernel factory backed by ``bottleneck``.

    Importing :mod:`bottleneck` (and :mod:`numpy`) is deferred to
    factory-call time, so production environments without bottleneck
    installed import this module cleanly. The pure-stdlib backend is
    the production default and produces output that matches bottleneck
    element-wise (within IEEE-754 summation tolerance).

    The returned callable accepts a kernel name
    (``"sum"`` / ``"mean"`` / ``"std"`` / ``"min"`` / ``"max"``)
    and returns the bottleneck function (e.g. ``bn.move_mean``).
    """

    import bottleneck as bn  # noqa: F401 - lazy seam

    table: dict[str, Callable[..., Any]] = {
        "sum": bn.move_sum,
        "mean": bn.move_mean,
        "std": bn.move_std,
        "min": bn.move_min,
        "max": bn.move_max,
    }

    def _factory(kind: str) -> Callable[..., Any]:
        if kind not in table:
            raise RollingStatsError(
                f"bottleneck kernel factory: unknown kind {kind!r} "
                f"(expected one of {sorted(table)!r})"
            )
        return table[kind]

    return _factory
