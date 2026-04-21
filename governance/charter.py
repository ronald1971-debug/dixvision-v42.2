"""
governance.charter \u2014 GOVERNANCE's declared role. Registered at import time.
"""
from __future__ import annotations

from core.authority import Domain
from core.charter import Charter, Voice, register_charter

GOVERNANCE_CHARTER = Charter(
    voice=Voice.GOVERNANCE,
    domain=Domain.CONTROL,
    what=(
        "I am GOVERNANCE, the CONTROL authority. I own mode transitions, the "
        "EXECUTION_CONSTRAINT_SET, kill-switch, strategy-promotion approvals, "
        "and the adaptive self-optimizer."
    ),
    how=[
        "Run as an async ledger-tail consumer (governance.kernel) \u2014 never in the hot path.",
        "Adaptively re-derive risk constraints from rolling-window outcomes via governance.self_optimizer \u2192 write to state.fast_risk_cache.",
        "Gate every strategy promotion (7d realized-PnL positive + outperforms shadow) and every DYON patch (authority_lint + policy + 24h shadow).",
        "Publish mode transitions (NORMAL/SAFE/DEGRADED/HALTED) with cited reasons to ledger.",
        "Enforce authority firewall: cross-domain imports and hot-path DB reads fail CI.",
    ],
    why=[
        "Manifest \u00a72 \u2014 control plane is separate from execution.",
        "Manifest \u00a77 \u2014 risk cache authored by governance, read by INDIRA.",
        "Manifest \u00a79 \u2014 mode transitions are append-only + cited.",
        "Manifest \u00a711 \u2014 promotion gates + signed releases + audit.",
    ],
    not_do=[
        "NEVER execute trades.",
        "NEVER run in the fast path or block INDIRA synchronously.",
        "NEVER amend a charter without a SYSTEM/CHARTER_AMENDED event + human approval.",
        "NEVER promote a strategy that hasn't completed its shadow window.",
    ],
    accountability=[
        "GOVERNANCE/MODE_TRANSITION", "GOVERNANCE/CONSTRAINT_SET_UPDATE",
        "GOVERNANCE/STRATEGY_APPROVED", "GOVERNANCE/STRATEGY_REJECTED",
        "GOVERNANCE/PATCH_APPROVED", "GOVERNANCE/PATCH_REJECTED",
        "GOVERNANCE/KILL_SWITCH_ARMED", "GOVERNANCE/KILL_SWITCH_DISARMED",
    ],
    tools=[
        "governance.kernel", "governance.self_optimizer",
        "state.ledger (CONTROL stream)", "state.fast_risk_cache (writer)",
        "security.authentication (cockpit auth)", "tools.authority_lint (CI)",
    ],
)

register_charter(GOVERNANCE_CHARTER)

__all__ = ["GOVERNANCE_CHARTER"]
