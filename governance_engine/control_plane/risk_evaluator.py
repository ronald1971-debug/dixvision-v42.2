"""GOV-CP-02 — Risk Evaluator.

Evaluates a proposed order (symbol, side, qty) against the
governance-owned exposure book and the policy-loaded limit
constraints. Produces a :class:`RiskAssessment` describing whether
the order is approved and, if not, which limits were breached.

This is the *governance* risk gate (slow path, exact, all limits
checked). It complements ``core/fast_risk_cache.py`` (the runtime
fast-path snapshot used by Execution); per ``manifest.md`` §0.5
GOV-CP-02 is authoritative — the cache is an optimisation, not the
arbiter.

Determinism contract: same exposure book + same constraints + same
request → same verdict (INV-15).
"""

from __future__ import annotations

from collections.abc import Mapping

from core.contracts.governance import (
    Constraint,
    ConstraintKind,
    ConstraintScope,
    RiskAssessment,
)
from governance_engine.control_plane.exposure_store import ExposureStore


class ExposureBook:
    """Authoritative per-symbol exposure book.

    AUDIT-P0.4 — when constructed with an :class:`ExposureStore`, the
    book hydrates from the store at boot and writes through every
    ``set`` / ``apply``. Without a store the book is in-memory-only
    (the historical Phase-1 default).

    The store is **not** the arbiter — the in-memory dict still
    answers every read on the hot path, the store is a pure
    persistence sink so a kill -9 plus relaunch resumes from the
    last committed exposure rather than zero.
    """

    def __init__(
        self,
        exposures: Mapping[str, float] | None = None,
        *,
        store: ExposureStore | None = None,
    ) -> None:
        self._store = store
        if store is not None:
            persisted = store.load_exposures()
            seed: dict[str, float] = dict(persisted)
            seed.update(exposures or {})
            self._exposures: dict[str, float] = seed
        else:
            self._exposures = dict(exposures or {})

    def get(self, symbol: str) -> float:
        return self._exposures.get(symbol, 0.0)

    def set(self, symbol: str, qty: float, *, ts_ns: int = 0) -> None:
        self._exposures[symbol] = qty
        if self._store is not None:
            self._store.write_exposure(
                symbol=symbol, qty=qty, ts_ns=ts_ns
            )

    def apply(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        ts_ns: int = 0,
    ) -> float:
        """Mutate exposure for ``symbol`` and return the new value."""

        signed = qty if side == "BUY" else -qty
        new = self._exposures.get(symbol, 0.0) + signed
        self._exposures[symbol] = new
        if self._store is not None:
            self._store.write_exposure(
                symbol=symbol, qty=new, ts_ns=ts_ns
            )
        return new

    def snapshot(self) -> Mapping[str, float]:
        return dict(self._exposures)


class RiskEvaluator:
    name: str = "risk_evaluator"
    spec_id: str = "GOV-CP-02"

    def __init__(
        self,
        *,
        exposure_book: ExposureBook | None = None,
        constraints: tuple[Constraint, ...] = (),
    ) -> None:
        self._book = exposure_book or ExposureBook()
        self._constraints = tuple(constraints)

    # ------------------------------------------------------------------
    # Constraint store
    # ------------------------------------------------------------------

    def load_constraints(self, constraints: tuple[Constraint, ...]) -> None:
        self._constraints = tuple(constraints)

    @property
    def book(self) -> ExposureBook:
        return self._book

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _limit(
        self, kind: ConstraintKind, *, symbol: str | None = None
    ) -> float | None:
        """Resolve the most-specific numeric limit for ``kind``.

        SYMBOL-scoped beats GLOBAL-scoped. Returns ``None`` if no
        applicable constraint is present.
        """

        symbol_value: float | None = None
        global_value: float | None = None
        for c in self._constraints:
            if c.kind is not kind:
                continue
            raw = c.params.get("limit")
            if raw is None:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if c.scope is ConstraintScope.SYMBOL and symbol is not None:
                if c.params.get("symbol") == symbol:
                    symbol_value = value
            elif c.scope is ConstraintScope.GLOBAL:
                global_value = value
        return symbol_value if symbol_value is not None else global_value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        *,
        ts_ns: int,
        symbol: str,
        side: str,
        qty: float,
    ) -> RiskAssessment:
        """Return whether the proposed order passes governance risk gates."""

        if qty <= 0.0:
            return RiskAssessment(
                ts_ns=ts_ns,
                symbol=symbol,
                side=side,
                qty=qty,
                approved=False,
                rejection_code="RISK_NON_POSITIVE_QTY",
                breached_limits=("RISK_NON_POSITIVE_QTY",),
                exposure_after=self._book.get(symbol),
            )
        if side not in ("BUY", "SELL"):
            return RiskAssessment(
                ts_ns=ts_ns,
                symbol=symbol,
                side=side,
                qty=qty,
                approved=False,
                rejection_code="RISK_INVALID_SIDE",
                breached_limits=("RISK_INVALID_SIDE",),
                exposure_after=self._book.get(symbol),
            )

        breached: list[str] = []

        max_qty = self._limit(ConstraintKind.MAX_POSITION_QTY, symbol=symbol)
        if max_qty is not None and qty > max_qty:
            breached.append(f"MAX_POSITION_QTY:{max_qty:g}")

        signed = qty if side == "BUY" else -qty
        exposure_after = self._book.get(symbol) + signed

        max_exposure = self._limit(
            ConstraintKind.MAX_SYMBOL_EXPOSURE, symbol=symbol
        )
        if max_exposure is not None and abs(exposure_after) > max_exposure:
            breached.append(f"MAX_SYMBOL_EXPOSURE:{max_exposure:g}")

        approved = not breached
        rejection_code = "" if approved else breached[0]

        return RiskAssessment(
            ts_ns=ts_ns,
            symbol=symbol,
            side=side,
            qty=qty,
            approved=approved,
            rejection_code=rejection_code,
            breached_limits=tuple(breached),
            exposure_after=exposure_after,
        )

    def commit(self, assessment: RiskAssessment) -> None:
        """Apply an approved assessment to the exposure book."""

        if not assessment.approved:
            raise ValueError("cannot commit a rejected assessment")
        self._book.apply(assessment.symbol, assessment.side, assessment.qty)


__all__ = ["ExposureBook", "RiskEvaluator"]
