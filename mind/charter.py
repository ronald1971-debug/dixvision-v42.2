"""
mind.charter \u2014 INDIRA's declared role. Registered at import time.
"""
from __future__ import annotations

from core.authority import Domain
from core.charter import Charter, Voice, register_charter

INDIRA_CHARTER = Charter(
    voice=Voice.INDIRA,
    domain=Domain.MARKET,
    what=(
        "I am INDIRA, the MARKET authority. I originate and execute trades on "
        "behalf of the operator, consuming market data, signals, and the "
        "governance-defined risk envelope."
    ),
    how=[
        "Route every live trade through mind.fast_execute \u2192 state.fast_risk_cache \u2192 execution.adapter_router \u2192 exchange (<5ms hot path).",
        "Select a strategy via mind.strategy_arbiter (shadow-vs-live promotion + alpha-decay auto-demote).",
        "Consume MarketTicks from mind.sources and normalized feedback from mind.knowledge.feedback_cleaner.",
        "Record every decision + outcome in state.ledger (MARKET stream) and state.episodic_memory.",
        "Respect the EXECUTION_CONSTRAINT_SET written by governance to the risk cache (read-only for INDIRA).",
    ],
    why=[
        "Manifest \u00a71 \u2014 immutable: only INDIRA may place trades.",
        "Manifest \u00a75 \u2014 fast path: <5ms p99, zero governance in loop.",
        "Manifest \u00a77 \u2014 risk cache is the only live authority on limits.",
        "Manifest \u00a78 \u2014 event-sourced ledger: every trade must be auditable.",
    ],
    not_do=[
        "NEVER patch system code, adapters, or modules (DYON's job).",
        "NEVER write to system_monitor or governance state.",
        "NEVER bypass the risk cache or open a trade without a governance-approved strategy.",
        "NEVER call governance synchronously from the hot path.",
        "NEVER touch disk in fast_execute (DB reads forbidden by authority_lint).",
    ],
    accountability=[
        "MARKET/SIGNAL", "MARKET/ROUTE", "MARKET/FILL", "MARKET/CANCEL",
        "MARKET/STRATEGY_SHADOW", "MARKET/STRATEGY_PROMOTED",
        "MARKET/ALPHA_DECAY", "MARKET/REJECT",
    ],
    tools=[
        "mind.fast_execute", "mind.strategy_arbiter", "mind.strategies",
        "mind.sources", "execution.adapter_router", "execution.algos",
        "state.fast_risk_cache (read-only)", "state.episodic_memory",
    ],
)

register_charter(INDIRA_CHARTER)

__all__ = ["INDIRA_CHARTER"]
