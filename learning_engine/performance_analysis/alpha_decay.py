# ADAPTED FROM: qlib/qlib/contrib/eva/alpha.py
# ADAPTED FROM: qlib/qlib/contrib/report/analysis_position/score_ic.py
"""Alpha-decay metrics — IC / Rank IC / ICIR across forecasting horizons.

Adapted from microsoft/qlib's ``contrib/eva/alpha.py`` Information Coefficient
(IC) and ICIR pattern: every prediction-horizon ``h`` is scored by the
*cross-sectional* correlation between the model's score at time ``t`` and the
realised forward return over ``[t, t + h]``.  The temporal mean and standard
deviation of those per-bucket correlations summarise how informative the
alpha is and how much that information *decays* as ``h`` grows.

This module is the **second** of the S-04 pyqlib triple
(``pnl_attribution.py`` + ``alpha_decay.py`` + ``execution_quality.py``);
it intentionally restricts itself to one concern: turning a stream of
:class:`ScoredObservation` records into a frozen :class:`AlphaDecayCurve`.

Tier
----
**OFFLINE.** ``learning_engine/performance_analysis/`` is a slow-cadence
analytics tier — never called from the hot path, never imported by
``hot_path/`` modules (authority_lint T1 / B1).

Design constraints
------------------
* **Pure functions.** No clock reads (``time.time()`` / ``datetime.now()``),
  no IO, no global mutable state. Replay-deterministic (INV-15): identical
  inputs produce byte-identical :class:`AlphaDecayCurve` outputs.
* **Frozen contracts.** :class:`ScoredObservation`, :class:`HorizonIC` and
  :class:`AlphaDecayCurve` are ``@dataclass(frozen=True, slots=True)`` so
  structural equality is preserved across replays.
* **Eager validation.** Constructors reject malformed input (``ValueError`` /
  ``TypeError``) so a downstream learning loop never observes partial state.
* **No new pip dependencies.** :data:`NEW_PIP_DEPENDENCIES` is empty — the
  qlib formulas only need :mod:`math` from the stdlib.
* **Stable accumulation order.** Per-bucket correlations are computed in
  the iteration order of ``(ts_ns, symbol)`` after a stable sort, so the
  emitted ``ic_mean`` / ``ic_std`` / ``icir`` values are identical across
  CPython versions.

Algorithmic summary
-------------------
For a single horizon ``h`` and a stream of :class:`ScoredObservation`
records ``(ts_ns, symbol, score, future_return)``:

1. Group observations by ``ts_ns`` (each group is a *cross-section* — the
   universe of symbols at that timestamp).
2. For each cross-section with at least two observations, compute the
   Pearson correlation between ``score`` and ``future_return`` (the
   per-bucket *IC*) and the same correlation on the *ranks* of those
   columns (the per-bucket *Rank IC*).  Buckets with zero variance in
   either column are skipped (the correlation is undefined).
3. ``ic_mean`` / ``ic_std`` are the arithmetic mean and **population**
   standard deviation of the per-bucket ICs across the surviving
   buckets.  ``icir = ic_mean / ic_std`` (``0.0`` if ``ic_std == 0``).
   Rank-IC analogues are computed identically.

The qlib reference computes IC per group with ``DataFrameGroupBy.corr``
and ICIR as ``IC.mean() / IC.std()`` — this module reproduces that exact
formula in pure Python without pulling pandas/polars/numpy.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from typing import Final

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()
"""S-04.2 introduces no new pip dependencies — pure stdlib."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ScoredObservation:
    """One ``(score, future_return)`` row at a ``(ts_ns, symbol)`` cell.

    Args:
        ts_ns: Cross-section timestamp in nanoseconds (the time at which
            the alpha *score* was emitted; the forward return covers the
            window ``[ts_ns, ts_ns + horizon_steps * step_ns]`` but the
            step size is opaque to this module).  ``int >= 0``.
        symbol: Instrument identifier; non-empty string.  Two rows with
            the same ``(ts_ns, symbol)`` are accepted (qlib does not
            de-duplicate either) but they will both contribute to the
            cross-section correlation.
        score: The alpha's predicted edge for ``symbol`` at ``ts_ns``.
            Finite float; NaN / ±Inf are rejected.
        future_return: Realised forward return over the prediction
            horizon (decimal — ``0.01`` is a one-percent move).  Finite
            float; NaN / ±Inf are rejected.

    Raises:
        TypeError: When any field has the wrong runtime type.
        ValueError: When ``ts_ns`` is negative, ``symbol`` is empty, or
            ``score`` / ``future_return`` is non-finite.
    """

    ts_ns: int
    symbol: str
    score: float
    future_return: float

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"ts_ns must be int; got {type(self.ts_ns).__name__}")
        if self.ts_ns < 0:
            raise ValueError(f"ts_ns must be >= 0; got {self.ts_ns}")
        if not isinstance(self.symbol, str):
            raise TypeError(f"symbol must be str; got {type(self.symbol).__name__}")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise TypeError(f"score must be float; got {type(self.score).__name__}")
        if not math.isfinite(float(self.score)):
            raise ValueError(f"score must be finite; got {self.score}")
        if not isinstance(self.future_return, (int, float)) or isinstance(self.future_return, bool):
            raise TypeError(f"future_return must be float; got {type(self.future_return).__name__}")
        if not math.isfinite(float(self.future_return)):
            raise ValueError(f"future_return must be finite; got {self.future_return}")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class HorizonIC:
    """IC / Rank-IC summary for one prediction horizon.

    Args:
        horizon_steps: The forward window in *steps* (interpretation of
            "step" is opaque to this module — bars, minutes, days, etc.).
            ``int > 0``.
        n_buckets: Number of cross-section buckets that survived the
            min-2-observations + non-zero-variance filter.  ``>= 0``.
        n_observations: Total :class:`ScoredObservation` rows fed in
            (before any filtering).  ``>= 0``.
        ic_mean: Arithmetic mean of the per-bucket Pearson IC values.
            ``0.0`` when ``n_buckets == 0``.
        ic_std: **Population** standard deviation of the per-bucket
            Pearson IC values.  ``0.0`` when ``n_buckets <= 1``.
        icir: ``ic_mean / ic_std`` (``0.0`` when ``ic_std == 0.0``).
        rank_ic_mean: As :attr:`ic_mean` but on the *ranks* of the score
            and forward-return columns within each bucket.
        rank_ic_std: As :attr:`ic_std` but for the rank-IC stream.
        rank_icir: ``rank_ic_mean / rank_ic_std``.

    Invariants enforced in :meth:`__post_init__`:
        * ``horizon_steps > 0``
        * ``n_buckets >= 0``
        * ``n_observations >= 0``
        * ``ic_std >= 0.0`` and ``rank_ic_std >= 0.0``
        * All metric fields are finite (no NaN / ±Inf).

    The class is hashable and frozen — two :class:`HorizonIC` records
    constructed from the same inputs compare equal byte-for-byte and can
    be stored in a :class:`set` or used as a :class:`dict` key.
    """

    horizon_steps: int
    n_buckets: int
    n_observations: int
    ic_mean: float
    ic_std: float
    icir: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float

    def __post_init__(self) -> None:
        if not isinstance(self.horizon_steps, int) or isinstance(self.horizon_steps, bool):
            raise TypeError(f"horizon_steps must be int; got {type(self.horizon_steps).__name__}")
        if self.horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be > 0; got {self.horizon_steps}")
        if self.n_buckets < 0:
            raise ValueError(f"n_buckets must be >= 0; got {self.n_buckets}")
        if self.n_observations < 0:
            raise ValueError(f"n_observations must be >= 0; got {self.n_observations}")
        for name, value in (
            ("ic_mean", self.ic_mean),
            ("ic_std", self.ic_std),
            ("icir", self.icir),
            ("rank_ic_mean", self.rank_ic_mean),
            ("rank_ic_std", self.rank_ic_std),
            ("rank_icir", self.rank_icir),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite; got {value}")
        if self.ic_std < 0.0:
            raise ValueError(f"ic_std must be >= 0; got {self.ic_std}")
        if self.rank_ic_std < 0.0:
            raise ValueError(f"rank_ic_std must be >= 0; got {self.rank_ic_std}")


@dataclasses.dataclass(frozen=True, slots=True)
class AlphaDecayCurve:
    """Stack of :class:`HorizonIC` records ordered by ``horizon_steps``.

    Args:
        horizons: Tuple of :class:`HorizonIC` records, **sorted strictly
            ascending** by ``horizon_steps`` (the ``compute_alpha_decay``
            factory enforces this).  May be empty.

    Raises:
        TypeError: If any element is not a :class:`HorizonIC`.
        ValueError: If two records share the same ``horizon_steps`` or
            the records are not in strictly-ascending order.
    """

    horizons: tuple[HorizonIC, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.horizons, tuple):
            raise TypeError(f"horizons must be tuple; got {type(self.horizons).__name__}")
        prev = -1
        for h in self.horizons:
            if not isinstance(h, HorizonIC):
                raise TypeError(f"horizons must contain HorizonIC; got {type(h).__name__}")
            if h.horizon_steps <= prev:
                raise ValueError(
                    "horizons must be strictly ascending by horizon_steps; "
                    f"got {h.horizon_steps} after {prev}"
                )
            prev = h.horizon_steps

    def horizons_steps(self) -> tuple[int, ...]:
        """Return the ``horizon_steps`` of every record in order."""
        return tuple(h.horizon_steps for h in self.horizons)

    def best_horizon(self) -> HorizonIC | None:
        """Return the :class:`HorizonIC` with the largest ``icir``.

        Returns ``None`` when the curve is empty.  Ties on ``icir`` are
        broken by **smaller** ``horizon_steps`` (earlier-decay wins) to
        keep the result deterministic across replays.
        """
        if not self.horizons:
            return None
        best = self.horizons[0]
        for h in self.horizons[1:]:
            if h.icir > best.icir:
                best = h
            elif h.icir == best.icir and h.horizon_steps < best.horizon_steps:
                best = h
        return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length float lists.

    Returns ``None`` when ``len(xs) < 2`` or either column has zero
    variance (the correlation is mathematically undefined in those
    cases; qlib's ``DataFrame.corr`` reports ``NaN`` for them and we
    filter those buckets out upstream).
    """
    n = len(xs)
    if n != len(ys):  # pragma: no cover - guarded by callers
        raise ValueError(f"xs/ys length mismatch: {n} vs {len(ys)}")
    if n < 2:
        return None
    mean_x = math.fsum(xs) / n
    mean_y = math.fsum(ys) / n
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx = x - mean_x
        dy = y - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _ranks(values: list[float]) -> list[float]:
    """Return average ranks (ties share the mean of the contested ranks).

    This is the same convention as :func:`scipy.stats.rankdata` /
    pandas' ``rank(method="average")`` — the Pearson correlation of two
    rank vectors with this convention equals Spearman's rank-IC
    (qlib's ``corr(method="spearman")``).
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _population_std(values: list[float]) -> float:
    """Population standard deviation; ``0.0`` when ``len(values) <= 1``."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = math.fsum(values) / n
    sq = math.fsum((v - mean) * (v - mean) for v in values)
    return math.sqrt(sq / n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def empty_horizon_ic(horizon_steps: int) -> HorizonIC:
    """Return the canonical zero :class:`HorizonIC` for ``horizon_steps``."""
    return HorizonIC(
        horizon_steps=horizon_steps,
        n_buckets=0,
        n_observations=0,
        ic_mean=0.0,
        ic_std=0.0,
        icir=0.0,
        rank_ic_mean=0.0,
        rank_ic_std=0.0,
        rank_icir=0.0,
    )


def compute_ic(
    observations: Iterable[ScoredObservation],
    *,
    horizon_steps: int,
) -> HorizonIC:
    """Cross-sectional IC / Rank-IC summary for one horizon.

    Walks :paramref:`observations` once, groups by ``ts_ns``, and emits
    a per-bucket Pearson IC plus a per-bucket Spearman (rank) IC.  The
    returned :class:`HorizonIC` reports the **mean / population std /
    ICIR** across surviving buckets.

    Buckets with fewer than two observations or with zero variance in
    either column are dropped silently — qlib's reference implementation
    treats those as ``NaN`` and excludes them from the aggregate; we do
    the same in pure Python.

    Args:
        observations: Iterable of :class:`ScoredObservation` rows.
        horizon_steps: The forward window the rows refer to (only
            recorded on the output; this function never inspects future
            timestamps, so the value is opaque metadata).  ``int > 0``.

    Returns:
        :class:`HorizonIC`.  When :paramref:`observations` is empty the
        canonical zero record from :func:`empty_horizon_ic` is returned.

    Raises:
        TypeError: If any element of :paramref:`observations` is not a
            :class:`ScoredObservation`.
        ValueError: If ``horizon_steps <= 0``.
    """
    if not isinstance(horizon_steps, int) or isinstance(horizon_steps, bool):
        raise TypeError(f"horizon_steps must be int; got {type(horizon_steps).__name__}")
    if horizon_steps <= 0:
        raise ValueError(f"horizon_steps must be > 0; got {horizon_steps}")

    buckets: dict[int, list[ScoredObservation]] = {}
    n_obs = 0
    for obs in observations:
        if not isinstance(obs, ScoredObservation):
            raise TypeError(
                f"observations must contain ScoredObservation; got {type(obs).__name__}"
            )
        buckets.setdefault(obs.ts_ns, []).append(obs)
        n_obs += 1

    if not buckets:
        return empty_horizon_ic(horizon_steps)

    ics: list[float] = []
    rank_ics: list[float] = []
    # iterate buckets in ascending ts_ns order so the emitted ic / rank_ic
    # streams are stable across replays (INV-15).
    for ts_ns in sorted(buckets):
        rows = buckets[ts_ns]
        if len(rows) < 2:
            continue
        # within a bucket, pre-sort by symbol so ties in score / return
        # accumulate in a fully deterministic order.
        rows = sorted(rows, key=lambda r: r.symbol)
        scores = [float(r.score) for r in rows]
        returns = [float(r.future_return) for r in rows]
        ic = _pearson(scores, returns)
        if ic is not None:
            ics.append(ic)
        rank_ic = _pearson(_ranks(scores), _ranks(returns))
        if rank_ic is not None:
            rank_ics.append(rank_ic)

    ic_mean = math.fsum(ics) / len(ics) if ics else 0.0
    ic_std = _population_std(ics)
    icir = ic_mean / ic_std if ic_std > 0.0 else 0.0
    rank_ic_mean = math.fsum(rank_ics) / len(rank_ics) if rank_ics else 0.0
    rank_ic_std = _population_std(rank_ics)
    rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0.0 else 0.0

    # Use the larger of the two surviving counts for n_buckets — the
    # canonical "did this horizon produce any signal at all" gate is
    # whether either stream had at least one bucket.  qlib's reference
    # records the IC and rank-IC counts separately on the report; we
    # surface the consolidated count and rely on n_observations for the
    # raw row count.
    n_buckets = max(len(ics), len(rank_ics))

    return HorizonIC(
        horizon_steps=horizon_steps,
        n_buckets=n_buckets,
        n_observations=n_obs,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        rank_ic_mean=rank_ic_mean,
        rank_ic_std=rank_ic_std,
        rank_icir=rank_icir,
    )


def compute_alpha_decay(
    observations_by_horizon: Mapping[int, Iterable[ScoredObservation]],
) -> AlphaDecayCurve:
    """Aggregate :func:`compute_ic` across every horizon into a curve.

    Args:
        observations_by_horizon: Mapping from ``horizon_steps`` (positive
            int) to an iterable of :class:`ScoredObservation` rows for
            that horizon.  An empty mapping yields an empty curve.

    Returns:
        :class:`AlphaDecayCurve` whose ``horizons`` tuple is sorted
        strictly ascending by ``horizon_steps`` — replay-deterministic.

    Raises:
        TypeError: If a key is not an :class:`int` or a value is not
            iterable.
        ValueError: If any key is ``<= 0`` or duplicated under a
            different :class:`int` representation.
    """
    if not isinstance(observations_by_horizon, Mapping):
        raise TypeError(
            f"observations_by_horizon must be Mapping; got {type(observations_by_horizon).__name__}"
        )
    horizons: list[HorizonIC] = []
    for horizon_steps in sorted(observations_by_horizon):
        if not isinstance(horizon_steps, int) or isinstance(horizon_steps, bool):
            raise TypeError(
                f"observations_by_horizon keys must be int; got {type(horizon_steps).__name__}"
            )
        if horizon_steps <= 0:
            raise ValueError(f"observations_by_horizon keys must be > 0; got {horizon_steps}")
        horizons.append(
            compute_ic(
                observations_by_horizon[horizon_steps],
                horizon_steps=horizon_steps,
            )
        )
    return AlphaDecayCurve(horizons=tuple(horizons))


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "AlphaDecayCurve",
    "HorizonIC",
    "ScoredObservation",
    "compute_alpha_decay",
    "compute_ic",
    "empty_horizon_ic",
]
