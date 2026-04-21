"""
execution/emergency_executor.py
Invoked by the interrupt system. Translates a HazardAction into a concrete
system action: halt trading, enter safe mode, flatten positions, kill.
"""
from __future__ import annotations

import os
import threading
from typing import Any

from state.ledger.writer import get_writer


class EmergencyExecutor:
    def __init__(self) -> None:
        self._writer = get_writer()

    def execute(self, action: Any) -> None:
        act = str(getattr(action, "action", ""))
        reason = str(getattr(action, "reason", ""))
        hazard = str(getattr(action, "hazard_type", ""))
        severity = str(getattr(action, "severity", ""))

        self._writer.write("SYSTEM", "EMERGENCY_ACTION", "emergency_executor", {
            "action": act,
            "reason": reason,
            "hazard_type": hazard,
            "severity": severity,
        })

        if act == "halt_trading":
            self._halt_trading(reason=reason)
        elif act == "safe_mode":
            self._enter_safe_mode(reason=reason)
        elif act == "flatten_positions":
            self._enter_safe_mode(reason=reason)
            if getattr(action, "flatten", False):
                self._flatten_positions()
        elif act == "kill":
            self._kill(reason=reason)

    # -- concrete actions -------------------------------------------------

    def _halt_trading(self, reason: str = "") -> None:
        from governance.mode_manager import get_mode_manager

        get_mode_manager().halt(reason=reason)

    def _enter_safe_mode(self, reason: str = "") -> None:
        from governance.mode.safe_mode import enter_safe_mode

        enter_safe_mode(reason=reason)

    def _flatten_positions(self) -> None:
        from execution.adapter_router import get_adapter_router
        from mind.portfolio_manager import get_portfolio_manager

        pm = get_portfolio_manager().snapshot()
        router = get_adapter_router()
        for asset, pos in pm.positions.items():
            if abs(pos.size) <= 0:
                continue
            adapter = router.route(asset)
            if adapter is None:
                continue
            side = "SELL" if pos.size > 0 else "BUY"
            try:
                adapter.place_order(
                    symbol=asset, side=side,
                    size=abs(pos.size) * pos.avg_price,
                    order_type="MARKET",
                )
            except Exception:
                continue

    def _kill(self, reason: str = "") -> None:
        try:
            from enforcement.kill_switch import trigger

            trigger(reason=reason or "emergency_kill")
        except Exception:
            # As a last resort, terminate the process.
            os._exit(2)


_ee: EmergencyExecutor | None = None
_lock = threading.Lock()


def get_emergency_executor() -> EmergencyExecutor:
    global _ee
    if _ee is None:
        with _lock:
            if _ee is None:
                _ee = EmergencyExecutor()
    return _ee
