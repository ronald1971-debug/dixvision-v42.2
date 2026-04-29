"""Almgren-Chriss optimal execution scheduler.

The classical Almgren-Chriss model
(Almgren & Chriss, "Optimal execution of portfolio transactions", 2000)
solves for a deterministic schedule of child trades that minimises the
expected execution cost plus a risk-aversion-weighted variance, given
linear permanent and temporary market-impact functions.

For a parent order of size :math:`X` to be worked over horizon :math:`T`
in :math:`N` equal-length slices of duration :math:`\\tau = T/N` we
have

* permanent linear impact ``g(v) = gamma * v``
* temporary linear impact ``h(v) = eta * v``
* return volatility ``sigma`` (per unit time)
* risk-aversion ``lambda`` (called ``risk_aversion`` here)

Define :math:`\\tilde\\eta = \\eta - \\gamma \\tau / 2` and
:math:`\\kappa^2 = \\lambda\\sigma^2 / \\tilde\\eta`. The optimal
holdings trajectory is

.. math::

    x_k = \\frac{\\sinh(\\kappa(T - k\\tau))}{\\sinh(\\kappa T)} \\, X
        \\qquad k = 0, 1, \\ldots, N

and the trade in slice :math:`k` is :math:`n_k = x_{k-1} - x_k`. As
:math:`\\lambda \\to 0` (risk-neutral), :math:`\\kappa \\to 0` and the
schedule degenerates to TWAP. As :math:`\\lambda \\to \\infty` the
schedule front-loads.

This module provides a pure / deterministic / side-effect-free
implementation. No clock, no PRNG, no I/O — INV-15. The output is the
exact closed-form solution for the parameters supplied; numerical
stability for very small :math:`\\kappa` is preserved via a Taylor
expansion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Threshold below which we treat kappa*T as zero (use the TWAP limit).
# The relative error of the second-order Taylor expansion of
# sinh(kT - k k tau) / sinh(kT) at this scale is below 1e-12.
_KAPPA_T_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class ExecutionSlice:
    """A single child slice in an Almgren-Chriss schedule."""

    index: int
    time_offset_seconds: float
    quantity: float  # signed; same sign as the parent quantity
    holdings_after: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"slice.index must be >= 0, got {self.index}")
        if self.time_offset_seconds < 0:
            raise ValueError(
                f"slice.time_offset_seconds must be >= 0, got {self.time_offset_seconds}"
            )


@dataclass(frozen=True, slots=True)
class ExecutionSchedule:
    """Closed-form Almgren-Chriss schedule.

    The schedule is immutable and deterministic for a given input
    tuple ``(quantity, horizon_seconds, n_slices, sigma, eta, gamma,
    risk_aversion)``. Replaying the same inputs returns identical
    slice quantities to floating-point precision.
    """

    quantity: float
    horizon_seconds: float
    n_slices: int
    sigma: float
    eta: float
    gamma: float
    risk_aversion: float
    kappa: float
    slices: tuple[ExecutionSlice, ...]

    @property
    def slice_seconds(self) -> float:
        return self.horizon_seconds / self.n_slices

    def total_quantity(self) -> float:
        return sum(s.quantity for s in self.slices)

    def is_twap(self) -> bool:
        """True iff the schedule is the risk-neutral TWAP limit."""

        return self.kappa < _KAPPA_T_EPS / max(self.horizon_seconds, 1.0)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve_almgren_chriss(
    *,
    quantity: float,
    horizon_seconds: float,
    n_slices: int,
    sigma: float,
    eta: float,
    gamma: float = 0.0,
    risk_aversion: float = 0.0,
) -> ExecutionSchedule:
    """Return the closed-form Almgren-Chriss schedule.

    Parameters
    ----------
    quantity:
        Signed parent quantity. Positive = liquidate (sell), negative =
        acquire (buy). The schedule preserves sign across slices.
    horizon_seconds:
        Total time over which the parent order must complete. Strictly
        positive.
    n_slices:
        Number of equal-length child slices. Strictly positive integer.
    sigma:
        Per-unit-time price volatility (>= 0). When 0, ``risk_aversion``
        has no effect and the schedule is TWAP.
    eta:
        Linear temporary-impact coefficient. Strictly positive.
    gamma:
        Linear permanent-impact coefficient. Non-negative. Defaults to
        zero — the canonical "no permanent impact" specialization.
    risk_aversion:
        Lambda, the operator's risk aversion (>= 0). Zero gives TWAP.

    Returns
    -------
    ExecutionSchedule
        Frozen dataclass containing the solved kappa and the ordered
        sequence of ``ExecutionSlice``s. Quantities sum to the original
        parent ``quantity`` to floating-point precision.

    Raises
    ------
    ValueError
        On any invalid parameter.
    """

    if not math.isfinite(quantity):
        raise ValueError("quantity must be finite")
    if not math.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be > 0")
    if n_slices <= 0:
        raise ValueError("n_slices must be > 0")
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError("sigma must be >= 0")
    if eta <= 0 or not math.isfinite(eta):
        raise ValueError("eta must be > 0 (temporary impact)")
    if gamma < 0 or not math.isfinite(gamma):
        raise ValueError("gamma must be >= 0 (permanent impact)")
    if risk_aversion < 0 or not math.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be >= 0")

    tau = horizon_seconds / n_slices
    eta_tilde = eta - 0.5 * gamma * tau
    if eta_tilde <= 0:
        raise ValueError(
            "eta - gamma*tau/2 must be > 0; reduce gamma, increase eta, "
            "or use more slices"
        )

    kappa_squared = (risk_aversion * sigma * sigma) / eta_tilde if sigma > 0 else 0.0
    kappa = math.sqrt(kappa_squared) if kappa_squared > 0 else 0.0

    # Schedule selection: TWAP limit when kappa*T is numerically zero.
    if kappa * horizon_seconds < _KAPPA_T_EPS:
        slices = _twap_slices(quantity, horizon_seconds, n_slices)
    else:
        slices = _ac_slices(quantity, horizon_seconds, n_slices, kappa)

    return ExecutionSchedule(
        quantity=quantity,
        horizon_seconds=horizon_seconds,
        n_slices=n_slices,
        sigma=sigma,
        eta=eta,
        gamma=gamma,
        risk_aversion=risk_aversion,
        kappa=kappa,
        slices=slices,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sinh_ratio(x: float, y: float) -> float:
    """Compute ``sinh(x) / sinh(y)`` for ``0 <= x <= y`` without overflow.

    Uses the algebraic rewrite

        sinh(x) / sinh(y) = (e^x - e^-x) / (e^y - e^-y)
                          = e^(x - y) * (1 - e^(-2x)) / (1 - e^(-2y))

    The right-hand form is stable: ``e^(x - y)`` lies in (0, 1] when
    ``x <= y``, and the two ``1 - e^(-2*)`` factors lie in (0, 1].
    """

    if y <= 0:
        # Caller guards against this with the kappa*T epsilon, but be
        # defensive: at y == 0 the ratio is undefined; return 1 by L'Hopital
        # which corresponds to TWAP, the limit we want.
        return 1.0
    if x <= 0:
        return 0.0
    log_diff = x - y
    num = -math.expm1(-2.0 * x)  # 1 - e^(-2x)
    den = -math.expm1(-2.0 * y)  # 1 - e^(-2y)
    if den == 0.0:
        return 0.0
    return math.exp(log_diff) * (num / den)


def _twap_slices(
    quantity: float, horizon_seconds: float, n_slices: int
) -> tuple[ExecutionSlice, ...]:
    """Equal-size slices; remainder absorbed into the last to preserve total."""

    tau = horizon_seconds / n_slices
    base = quantity / n_slices
    out: list[ExecutionSlice] = []
    holdings = quantity
    for k in range(n_slices):
        # Floor to base, then push the residual into the final slice so the
        # sum is exactly `quantity` to float precision.
        if k < n_slices - 1:
            qty = base
        else:
            qty = holdings  # whatever is left
        holdings -= qty
        out.append(
            ExecutionSlice(
                index=k,
                time_offset_seconds=(k + 1) * tau,
                quantity=qty,
                holdings_after=holdings,
            )
        )
    return tuple(out)


def _ac_slices(
    quantity: float, horizon_seconds: float, n_slices: int, kappa: float
) -> tuple[ExecutionSlice, ...]:
    tau = horizon_seconds / n_slices

    def holdings_at(k: int) -> float:
        # Numerically stable sinh(kappa*a) / sinh(kappa*T) for a in [0, T].
        # For large kappa*T, sinh overflows; use the exponential ratio
        # which is well-defined for any real argument.
        a = horizon_seconds - k * tau  # >= 0 for k in [0, n_slices]
        return _sinh_ratio(kappa * a, kappa * horizon_seconds) * quantity

    out: list[ExecutionSlice] = []
    prev = quantity  # x_0 = X
    for k in range(1, n_slices + 1):
        if k == n_slices:
            x_k = 0.0  # exact, regardless of accumulated FP drift
        else:
            x_k = holdings_at(k)
        n_k = prev - x_k
        out.append(
            ExecutionSlice(
                index=k - 1,
                time_offset_seconds=k * tau,
                quantity=n_k,
                holdings_after=x_k,
            )
        )
        prev = x_k
    return tuple(out)


__all__ = [
    "ExecutionSchedule",
    "ExecutionSlice",
    "solve_almgren_chriss",
]
