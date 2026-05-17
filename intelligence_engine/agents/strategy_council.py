# ADAPTED FROM: https://github.com/geekan/MetaGPT (MIT)
#
# Tier-C C-14 — MetaGPT-shaped multi-role strategy critique council
# (intelligence-engine facade).
#
# The canonical types + :func:`run_critique_round` dispatcher live in
# :mod:`core.contracts.critique` so that both
# :mod:`intelligence_engine.agents.strategy_council` (runtime side)
# and :mod:`evolution_engine.critique_loop` (offline side) can depend
# on a single neutral source without crossing the L2/L3 authority
# boundary.
#
# This module is the **runtime-engine** facade: it re-exports the
# public C-14 surface so callers under :mod:`intelligence_engine` can
# import via the directive path
# ``intelligence_engine.agents.strategy_council``. A future PR will
# add LiteLLMRouter-backed :class:`StructuredSpeaker` implementations
# here — they remain in the intelligence engine because they call out
# to AI providers (B24).
#
# Authority constraints inherited from
# :mod:`core.contracts.critique`:
#
#   * **ADVISORY only** (INV-12) — value objects only.
#   * **RUNTIME_SAFE** — pure dispatcher; no clock / I/O / PRNG.
#   * **B1** — no execution_engine / governance_engine /
#     system_engine / learning_engine / evolution_engine /
#     core.contracts.events imports.
#   * **B24** — ``metagpt`` is permitted under
#     :mod:`intelligence_engine.agents` only.
#   * No top-level imports of :mod:`metagpt`, :mod:`openai`,
#     :mod:`anthropic`, :mod:`litellm`, :mod:`requests`,
#     :mod:`asyncio`, :mod:`time`, :mod:`datetime`, :mod:`random`.
#
# ``NEW_PIP_DEPENDENCIES = ("metagpt",)`` is the lazy seam advertised
# to ``tools/cli.py install-c-tier``.
"""C-14 strategy council — MetaGPT-shape iterative critique."""

from __future__ import annotations

from core.contracts.critique import (
    CANONICAL_CRITIC_ROLES,
    MAX_GOAL_LEN,
    MAX_PAYLOAD_KEY_LEN,
    MAX_PAYLOAD_KEYS,
    MAX_PAYLOAD_VALUE_LEN,
    MAX_PROFILE_LEN,
    MAX_RATIONALE_LEN,
    MAX_REFINEMENT_LEN,
    MAX_REFINEMENTS,
    ConsensusError,
    CriticMessage,
    CriticRole,
    CritiqueConsensus,
    CritiqueLog,
    CritiqueRecommendation,
    MessageError,
    RoleSpec,
    RoleSpecError,
    StrategyCouncilError,
    StructuredSpeaker,
    run_critique_round,
)

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "StrategyCouncilError",
    "RoleSpecError",
    "MessageError",
    "ConsensusError",
    "CritiqueRecommendation",
    "CriticRole",
    "CANONICAL_CRITIC_ROLES",
    "RoleSpec",
    "CriticMessage",
    "CritiqueLog",
    "CritiqueConsensus",
    "StructuredSpeaker",
    "run_critique_round",
    "MAX_GOAL_LEN",
    "MAX_PROFILE_LEN",
    "MAX_PAYLOAD_KEYS",
    "MAX_PAYLOAD_KEY_LEN",
    "MAX_PAYLOAD_VALUE_LEN",
    "MAX_RATIONALE_LEN",
    "MAX_REFINEMENT_LEN",
    "MAX_REFINEMENTS",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("metagpt",)
