"""Task-class taxonomy for the registry-driven Cognitive Router.

Each task class corresponds to a *kind* of work the operator (or an
internal subsystem) wants to delegate to an AI provider. The router
maps a :class:`TaskClass` to a tuple of capability tags, then picks
providers from the SCVS registry whose declared ``capabilities``
include every required tag.

Why an enum + capability mapping instead of hard-coded provider names
--------------------------------------------------------------------
The repo's design rule (registry-driven AI, no hard-coded provider
names) means widget code and routing decisions both project the same
single source of truth: ``registry/data_source_registry.yaml``. A
TaskClass is the only thing the chat widgets ever choose; capability
tags are the contract that lets a future AI (Claude / Qwen / Mistral
/ open-weights / anything 2027 brings) be picked up automatically the
moment its registry row lands. ``tools/authority_lint.py`` rule
``B23`` enforces this at the source level: chat widget modules may
not contain string literals matching known provider tokens.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from system_engine.scvs.source_registry import ALLOWED_AI_CAPABILITIES


class TaskClass(StrEnum):
    """Operator-facing AI task classes.

    The set is intentionally small. Every member is a distinct chat
    surface or subsystem responsibility — adding a new value should
    correspond to an actual product surface, not a slight variation
    on an existing one.
    """

    # Indira Chat — regime-aware reasoning over portfolio / belief
    # state / pressure vector. "Why did you buy BTCUSDT?", "What is
    # your view on EUR/USD over the next 4 hours?", etc.
    INDIRA_REASONING = "indira_reasoning"

    # Dyon Chat — code generation / connector authoring / patch
    # proposals. Deeply tool-using; benefits from long context windows
    # to load surrounding files.
    DYON_CODING = "dyon_coding"

    # Indira Chat — multimodal research (charts, PDFs, screenshots,
    # on-chain visualisations). Distinct from pure reasoning because
    # not every provider supports image/audio inputs.
    INDIRA_MULTIMODAL_RESEARCH = "indira_multimodal_research"

    # Dyon Chat — multi-step agent orchestration (run a Devin session,
    # spawn a sub-task, drive a tool chain). The router specifically
    # filters for ``agent_orchestration`` here so non-agentic providers
    # are excluded.
    DYON_AGENT_ORCHESTRATION = "dyon_agent_orchestration"

    # Indira Chat — real-time research over news / social / on-chain
    # signals where freshness matters more than depth.
    INDIRA_REALTIME_RESEARCH = "indira_realtime_research"


# Capability requirements per task class.
#
# A provider is eligible for a task class iff its declared
# ``capabilities`` is a *superset* of the requirements listed here.
# Any new entry's tuple MUST be a subset of
# :data:`ALLOWED_AI_CAPABILITIES` — the module assertion below catches
# typos at import time so a mis-typed capability never silently
# excludes every provider in production.
_REQUIREMENTS: Mapping[TaskClass, tuple[str, ...]] = MappingProxyType(
    {
        TaskClass.INDIRA_REASONING: ("reasoning",),
        TaskClass.DYON_CODING: ("code_gen", "long_context"),
        TaskClass.INDIRA_MULTIMODAL_RESEARCH: ("reasoning", "multimodal"),
        TaskClass.DYON_AGENT_ORCHESTRATION: (
            "agent_orchestration",
            "tool_use",
        ),
        TaskClass.INDIRA_REALTIME_RESEARCH: ("realtime_search",),
    }
)


def required_capabilities(task: TaskClass) -> tuple[str, ...]:
    """Return the (immutable) capability tuple required for ``task``."""

    return _REQUIREMENTS[task]


# Boot-time validation: every required capability must be a member of
# :data:`ALLOWED_AI_CAPABILITIES`. If this fails, the system_engine
# capability set and the cognitive router are out of sync — fix the
# requirements mapping above (do NOT relax this check).
for _task, _caps in _REQUIREMENTS.items():
    _unknown = set(_caps) - ALLOWED_AI_CAPABILITIES
    if _unknown:  # pragma: no cover — guarded at import
        raise RuntimeError(
            f"cognitive_router.task_class: TaskClass {_task.value!r}"
            f" requires unknown capabilities {sorted(_unknown)};"
            f" must be a subset of"
            f" {sorted(ALLOWED_AI_CAPABILITIES)}"
        )


__all__ = ["TaskClass", "required_capabilities"]
