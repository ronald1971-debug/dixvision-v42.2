"""
execution/trade_executor.py
Executes an Indira ``ExecutionEvent`` against the adapter registered with the
AdapterRouter. Logs every step to the ledger; never blocks the hot path.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from execution.adapter_router import get_adapter_router
from state.ledger.writer import get_writer
from system.metrics import get_metrics


@dataclass
class ExecuteResult:
    ok: bool
    adapter: str
    response: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class TradeExecutor:
    def __init__(self) -> None:
        self._router = get_adapter_router()
        self._writer = get_writer()
        self._metrics = get_metrics()

    def execute(self, event: Any) -> ExecuteResult:
        if getattr(event, "event_type", "") != "TRADE_EXECUTION":
            return ExecuteResult(False, "none", reason="not_a_trade_event")
        if not getattr(event, "allowed", False):
            return ExecuteResult(False, "none", reason="risk_disallowed")

        asset = str(event.asset)
        adapter = self._router.route(asset)
        if adapter is None:
            self._writer.write("MARKET", "ORDER_REJECTED", "trade_executor",
                               {"asset": asset, "reason": "no_adapter"})
            self._metrics.increment("trade_executor.no_adapter")
            return ExecuteResult(False, "none", reason="no_adapter")

        try:
            resp = adapter.place_order(
                symbol=asset, side=event.side,
                size=float(event.size_usd), order_type=str(event.order_type),
            )
        except Exception as e:
            self._writer.write("MARKET", "ORDER_ERROR", "trade_executor",
                               {"asset": asset, "error": str(e)})
            self._metrics.increment("trade_executor.error")
            return ExecuteResult(False, str(adapter), reason=f"adapter_error:{e}")

        self._writer.write("MARKET", "ORDER_SUBMITTED", "trade_executor", {
            "asset": asset, "side": event.side, "size_usd": event.size_usd,
            "adapter": type(adapter).__name__, "response": resp,
        })
        self._metrics.increment("trade_executor.submit")
        return ExecuteResult(True, type(adapter).__name__, response=resp)


_te: TradeExecutor | None = None
_lock = threading.Lock()


def get_trade_executor() -> TradeExecutor:
    global _te
    if _te is None:
        with _lock:
            if _te is None:
                _te = TradeExecutor()
    return _te
