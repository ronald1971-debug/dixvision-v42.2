"""ExposureManager — intelligence-side per-symbol exposure tracker.

Distinct from :mod:`governance_engine.control_plane.exposure_store`,
which is the durable SQLite-backed compliance ledger. This manager is
an *in-memory* projection the portfolio brain reads to make allocator
decisions; it is fed by the same execution-engine fill stream and is
expected to converge with the durable store on every restart (the
durable store is replayed into a fresh ExposureManager on boot).

Authority constraints (manifest §H1):

* Imports only :mod:`core.contracts` and the standard library.
* No engine cross-imports.
* No clock, no PRNG, no IO.
* Mutations are pure-on-update: same fill sequence in the same order
  always produces the same snapshot (INV-15).
"""

from __future__ import annotations

from collections.abc import Mapping

from core.contracts.portfolio import ExposureSnapshot


class ExposureManager:
    """In-memory per-symbol notional exposure (signed)."""

    def __init__(self) -> None:
        self._by_symbol: dict[str, float] = {}
        self._last_update_ns: int = 0

    def apply_fill(
        self,
        *,
        ts_ns: int,
        symbol: str,
        side: str,
        notional_usd: float,
    ) -> None:
        """Apply one fill to the running exposure book.

        Side ``"BUY"`` adds positive notional; ``"SELL"`` adds negative
        notional. Symbols with no prior entry are created on first
        touch. The manager rejects monotonically-out-of-order
        timestamps so the projection is replay-stable (INV-15).
        """

        if not symbol:
            raise ValueError("ExposureManager.apply_fill: symbol must be non-empty")
        if side not in ("BUY", "SELL"):
            raise ValueError(
                f"ExposureManager.apply_fill: side must be 'BUY' or 'SELL', "
                f"got {side!r}"
            )
        # NB: phrased as ``not (x >= 0.0)`` rather than ``x < 0.0`` so NaN
        # — which compares False against every numeric under IEEE 754 — is
        # rejected here instead of silently passing through. A NaN would
        # otherwise pass this guard, then ``self._by_symbol[symbol] += NaN``
        # would permanently corrupt the in-memory book so every later
        # ``notional()`` / ``view()`` call propagates NaN to the allocator.
        # Same NaN-vs-IEEE754 lesson as the
        # ``PortfolioAllocatorConfig.max_symbol_notional_usd`` validator and
        # the ``PortfolioAllocator.allocate(available_capital_usd=...)``
        # validator in the sibling module.
        if not (notional_usd >= 0.0):
            raise ValueError(
                "ExposureManager.apply_fill: notional_usd must be non-negative, "
                f"got {notional_usd!r}"
            )
        if ts_ns <= 0:
            raise ValueError(
                "ExposureManager.apply_fill: ts_ns must be positive"
            )
        if ts_ns < self._last_update_ns:
            raise ValueError(
                "ExposureManager.apply_fill: out-of-order ts_ns "
                f"(got {ts_ns}, last {self._last_update_ns})"
            )

        signed = notional_usd if side == "BUY" else -notional_usd
        self._by_symbol[symbol] = self._by_symbol.get(symbol, 0.0) + signed
        self._last_update_ns = ts_ns

    def notional(self, symbol: str) -> float:
        return float(self._by_symbol.get(symbol, 0.0))

    def view(self) -> Mapping[str, float]:
        """Read-only view of the current per-symbol exposure."""

        # Return a fresh dict copy so callers cannot mutate internal state.
        return dict(self._by_symbol)

    def snapshot(self, ts_ns: int) -> ExposureSnapshot:
        """Materialise an :class:`ExposureSnapshot` at ``ts_ns``."""

        if ts_ns <= 0:
            raise ValueError("ExposureManager.snapshot: ts_ns must be positive")
        if ts_ns < self._last_update_ns:
            raise ValueError(
                "ExposureManager.snapshot: ts_ns predates last update "
                f"(got {ts_ns}, last {self._last_update_ns})"
            )
        return ExposureSnapshot(ts_ns=ts_ns, by_symbol=dict(self._by_symbol))

    def reset(self) -> None:
        """Drop all in-memory state (used during replay)."""

        self._by_symbol.clear()
        self._last_update_ns = 0


__all__ = ["ExposureManager"]
