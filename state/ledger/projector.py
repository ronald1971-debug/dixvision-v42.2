"""
state/ledger/projector.py
DIX VISION v42.2 — State Projector (Materialized Views)

Rebuilds current state by replaying ledger events.
"""
from __future__ import annotations

from typing import Any

from state.ledger.event_store import get_event_store


class Projector:
    """Rebuilds materialized state from event history."""

    def project_market_state(self) -> dict[str, Any]:
        events = get_event_store().query(event_type="MARKET", limit=1000)
        state: dict[str, Any] = {"open_orders": [], "fills": [], "portfolio": {}}
        for ev in reversed(events):
            import json
            payload = ev["payload"] if isinstance(ev["payload"], dict) else json.loads(ev["payload"])
            sub = ev["sub_type"]
            if sub == "TRADE_EXECUTION":
                state["fills"].append(payload)
            elif sub == "ORDER_PLACED":
                state["open_orders"].append(payload)
            elif sub == "ORDER_CANCELLED":
                state["open_orders"] = [o for o in state["open_orders"]
                                         if o.get("order_id") != payload.get("order_id")]
        return state

    def project_governance_state(self) -> dict[str, Any]:
        events = get_event_store().query(event_type="GOVERNANCE", limit=200)
        state: dict[str, Any] = {"mode": "NORMAL", "last_decision": None}
        for ev in reversed(events):
            import json
            payload = ev["payload"] if isinstance(ev["payload"], dict) else json.loads(ev["payload"])
            if ev["sub_type"] == "MODE_CHANGE":
                state["mode"] = payload.get("new_mode", "NORMAL")
                break
        return state

    def project_hazard_state(self) -> dict[str, Any]:
        events = get_event_store().query(event_type="HAZARD", limit=100)
        import json
        return {
            "active_hazards": len([
                e for e in events
                if (e["payload"] if isinstance(e["payload"], dict) else json.loads(e["payload"])).get("status") == "ACTIVE"
            ]),
            "recent": events[:10],
        }

_projector: Projector | None = None

def get_projector() -> Projector:
    global _projector
    if _projector is None:
        _projector = Projector()
    return _projector
