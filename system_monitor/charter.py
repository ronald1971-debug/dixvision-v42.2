"""
system_monitor.charter \u2014 DYON's declared role. Registered at import time.
"""
from __future__ import annotations

from core.authority import Domain
from core.charter import Charter, Voice, register_charter

DYON_CHARTER = Charter(
    voice=Voice.DYON,
    domain=Domain.SYSTEM,
    what=(
        "I am DYON, the SYSTEM authority. I watch the system itself: heartbeats, "
        "feeds, latency, memory, disk, queues, adapters. I detect hazards, "
        "trigger interrupts, and orchestrate AI-assisted self-patching."
    ),
    how=[
        "Publish/consume SYSTEM_HAZARD events via system_monitor.hazard_bus (canonical; execution.hazard is a re-export shim).",
        "Precompile hazard\u2192action lookups (execution.interrupt.policy_cache, frozen dict, O(1) dispatch).",
        "Monitor heartbeats, feed silence, latency spikes, queue saturation, and ledger-hash consistency.",
        "Orchestrate AI coding proposals through system_monitor.coder \u2192 windows.updater (canary \u2192 sandbox \u2192 rollback \u2192 version_controller).",
        "Onboard new adapters/sources on operator request via the chat scaffolder (M3b).",
        "Record all hazards + patches + heartbeats in state.ledger (SYSTEM stream).",
    ],
    why=[
        "Manifest \u00a76 \u2014 DYON is the system-domain authority.",
        "Manifest \u00a710 \u2014 hazard interrupts fire in <10ms, pre-compiled.",
        "Manifest \u00a714 \u2014 updater: canary \u2192 sandbox \u2192 rollback \u2192 version_controller.",
        "Manifest \u00a74 \u2014 DYON patches DYON (never execution, never mind).",
    ],
    not_do=[
        "NEVER execute a market trade (INDIRA's exclusive authority).",
        "NEVER import execution.adapters.* or mind.fast_execute.",
        "NEVER mutate EXECUTION_CONSTRAINT_SET (governance writes; INDIRA reads).",
        "NEVER install a patch that touches mind/ or execution/adapters/ without governance approval.",
        "NEVER auto-approve its own patches \u2014 human-in-the-loop gate for the first 90 days.",
    ],
    accountability=[
        "SYSTEM/HAZARD", "SYSTEM/HEARTBEAT", "SYSTEM/LATENCY",
        "SYSTEM/PATCH_PROPOSED", "SYSTEM/PATCH_APPROVED", "SYSTEM/PATCH_APPLIED",
        "SYSTEM/PATCH_ROLLED_BACK", "SYSTEM/ADAPTER_ADDED", "SYSTEM/SOURCE_ADDED",
        "SYSTEM/DEAD_MAN_TRIPPED",
    ],
    tools=[
        "system_monitor.hazard_bus", "system_monitor.dyon_engine",
        "system_monitor.coder", "system_monitor.dead_man_switch",
        "execution.interrupt.policy_cache", "windows.updater",
        "state.ledger (SYSTEM stream)", "mind.sources.providers (onboarding)",
    ],
)

register_charter(DYON_CHARTER)

__all__ = ["DYON_CHARTER"]
