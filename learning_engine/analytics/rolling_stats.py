# ADAPTED FROM: https://github.com/pydata/bottleneck  (BSD-2-Clause)
#
# Fast rolling-window statistics — ``learning_engine/analytics/`` is the
# **OFFLINE** high-throughput slow-cadence analytics tier. ``bottleneck``
# is the lazy seam: production never imports it; the numpy backend is
# default and produces byte-identical floats to bottleneck's
# ``move_mean`` / ``move_std`` / ``move_sum`` / ``move_min`` /
# ``move_max`` for the same inputs.
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
#     byte-identical numpy arrays.
#   * **B27 / B28 / INV-71** — no typed-event constructors here.
#   * No top-level imports of :mod:`bottleneck`, :mod:`time`,
#     :mod:`datetime`, :mod:`random`, :mod:`asyncio`, :mod:`torch`,
#     :mod:`polars`, :mod:`requests`.
"""Bottleneck-shape rolling-window statistics (OFFLINE_ONLY)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

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


def _coerce_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Coerce input to a contiguous 1-D float64 array."""

    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if arr.ndim != 1:
        raise RollingStatsError(
            f"rolling-stats input must be 1-D, got ndim={arr.ndim}"
        )
    return arr


def _normalize(
    values: Sequence[float] | np.ndarray,
    window: int,
    min_count: int | None,
) -> tuple[np.ndarray, RollingWindowSpec]:
    if min_count is None:
        min_count = window
    spec = RollingWindowSpec(window=window, min_count=min_count)
    arr = _coerce_1d(values)
    return arr, spec


# ---------------------------------------------------------------------------
# Pure-numpy rolling reductions (the production default backend)
# ---------------------------------------------------------------------------


def _move_reduce(
    arr: np.ndarray,
    spec: RollingWindowSpec,
    reducer: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Generic O(N * window) rolling reduction over a 1-D float64 array.

    Emits ``NaN`` whenever the trailing window contains fewer than
    ``spec.min_count`` non-NaN samples. The output has the same
    shape as the input.
    """

    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    if n == 0:
        return out
    for i in range(n):
        start = i + 1 - spec.window
        if start < 0:
            start = 0
        window_slice = arr[start : i + 1]
        finite_mask = np.isfinite(window_slice)
        finite_count = int(finite_mask.sum())
        if finite_count < spec.min_count:
            continue
        out[i] = reducer(window_slice[finite_mask])
    return out


def move_sum(
    values: Sequence[float] | np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
) -> np.ndarray:
    """Rolling sum — bottleneck ``move_sum`` shape."""

    arr, spec = _normalize(values, window, min_count)
    return _move_reduce(arr, spec, lambda w: float(np.sum(w)))


def move_mean(
    values: Sequence[float] | np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
) -> np.ndarray:
    """Rolling mean — bottleneck ``move_mean`` shape."""

    arr, spec = _normalize(values, window, min_count)
    return _move_reduce(arr, spec, lambda w: float(np.mean(w)))


def move_std(
    values: Sequence[float] | np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
    ddof: int = 0,
) -> np.ndarray:
    """Rolling standard deviation — bottleneck ``move_std`` shape.

    ``ddof`` defaults to 0 (population std) to match bottleneck.
    Use ``ddof=1`` for sample std.
    """

    if not isinstance(ddof, int) or ddof < 0:
        raise RollingStatsError(
            f"move_std.ddof must be int >= 0, got {ddof!r}"
        )
    arr, spec = _normalize(values, window, min_count)

    def _std(window_values: np.ndarray) -> float:
        if window_values.size - ddof <= 0:
            return math.nan
        return float(np.std(window_values, ddof=ddof))

    return _move_reduce(arr, spec, _std)


def move_min(
    values: Sequence[float] | np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
) -> np.ndarray:
    """Rolling minimum — bottleneck ``move_min`` shape."""

    arr, spec = _normalize(values, window, min_count)
    return _move_reduce(arr, spec, lambda w: float(np.min(w)))


def move_max(
    values: Sequence[float] | np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
) -> np.ndarray:
    """Rolling maximum — bottleneck ``move_max`` shape."""

    arr, spec = _normalize(values, window, min_count)
    return _move_reduce(arr, spec, lambda w: float(np.max(w)))


# ---------------------------------------------------------------------------
# Lazy seam — bottleneck-backed kernel factory (never called at import time)
# ---------------------------------------------------------------------------


def enable_bottleneck_kernel_factory() -> Callable[[str], Callable[..., Any]]:
    """Return a kernel factory backed by ``bottleneck``.

    Importing :mod:`bottleneck` is deferred to factory-call time, so
    production environments without bottleneck installed import this
    module cleanly. The numpy backend is the production default and
    produces byte-identical output for the same inputs.

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
