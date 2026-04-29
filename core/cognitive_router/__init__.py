"""Cognitive Router Layer (CRL) — registry-driven AI provider selection.

This package is the single, registry-driven entry point used by both
chat widgets (Indira Chat / Dyon Chat) and any internal subsystem that
wants to delegate work to an AI provider. The router never hard-codes
provider names — every routing decision is made by reading
``registry/data_source_registry.yaml`` (rows where ``category: ai``)
and intersecting the operator-requested capabilities with each row's
declared ``capabilities`` tuple.

Wave-01 scope (this PR)
-----------------------

* Pure ``select_providers(...)`` — given a registry + a task class
  (``reasoning`` / ``code_gen`` / ``multimodal`` / ``realtime_search``
  / ``long_context`` / ``tool_use`` / ``agent_orchestration``), return
  the ordered tuple of ``SourceDeclaration`` rows that match. Pure,
  deterministic, no I/O.
* No live API calls. The wave-02 actor that turns ``select_providers``
  into a real HTTP/MCP transport is a separate PR (it needs
  per-provider credentials, retry, fallback audit, etc.).

Why this layer exists
---------------------

Per the operator directive 2025-04-21 (registry-driven AI providers,
no hard-coded names, future providers picked up automatically), the
chat widgets must call ``select_providers`` instead of mentioning any
specific provider by name. ``tools/authority_lint.py`` rule B23 (added
in this PR) enforces this at the source level: chat widget modules
may not contain string literals matching known provider tokens.
"""

from core.cognitive_router.router import (
    AIProvider,
    enabled_ai_providers,
    select_providers,
)
from core.cognitive_router.task_class import TaskClass

__all__ = [
    "AIProvider",
    "TaskClass",
    "enabled_ai_providers",
    "select_providers",
]
