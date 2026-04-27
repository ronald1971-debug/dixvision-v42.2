"""Canary stage — bounded live exposure gate."""

from __future__ import annotations

from dataclasses import dataclass

from evolution_engine.patch_pipeline.pipeline import PatchStage, StageVerdict


@dataclass(frozen=True, slots=True)
class CanaryVerdict:
    orders: int
    rejects: int
    realised_pnl: float
    error_rate: float


class CanaryStage:
    """GOV-G18-S5."""

    name: str = "canary"
    spec_id: str = "GOV-G18-S5"

    __slots__ = ("_min_orders", "_max_error_rate", "_min_pnl")

    def __init__(
        self,
        *,
        min_orders: int = 5,
        max_error_rate: float = 0.10,
        min_pnl: float = 0.0,
    ) -> None:
        if min_orders < 1:
            raise ValueError("min_orders must be >= 1")
        if not 0.0 <= max_error_rate <= 1.0:
            raise ValueError("max_error_rate must be in [0, 1]")
        self._min_orders = min_orders
        self._max_error_rate = max_error_rate
        self._min_pnl = min_pnl

    def evaluate(
        self,
        *,
        ts_ns: int,
        orders: int,
        rejects: int,
        realised_pnl: float,
    ) -> tuple[CanaryVerdict, StageVerdict]:
        if orders < 0 or rejects < 0 or rejects > orders:
            raise ValueError("invalid canary counts")
        error_rate = 0.0 if orders == 0 else rejects / orders
        cv = CanaryVerdict(
            orders=orders,
            rejects=rejects,
            realised_pnl=realised_pnl,
            error_rate=error_rate,
        )
        passed = (
            orders >= self._min_orders
            and error_rate <= self._max_error_rate
            and realised_pnl >= self._min_pnl
        )
        verdict = StageVerdict(
            ts_ns=ts_ns,
            stage=PatchStage.CANARY,
            passed=passed,
            detail=(
                f"orders={orders} rejects={rejects} pnl={realised_pnl:.4f}"
            ),
            meta={
                "orders": str(orders),
                "rejects": str(rejects),
                "realised_pnl": f"{realised_pnl:.6f}",
                "error_rate": f"{error_rate:.6f}",
            },
        )
        return cv, verdict


__all__ = ["CanaryStage", "CanaryVerdict"]
