"""DIX strategy arena — DEAP-style tournament selection over RealitySummary.

Phase 10 / A-03 leaf. The arena runs OFFLINE only and emits ranked
survivor lists for the kill-underperformers (A-03.2) and promotion-
engine (A-03.3) leaves to consume.

Authority constraints:

* OFFLINE_ONLY tier — never imported by execution_engine /
  governance_engine / system_engine / intelligence_engine on the hot
  path.
* No clock, no IO, no PRNG without caller-supplied seed.
* INV-13 / INV-14: arena outputs are advisory; demotions and
  promotions go through governance via a typed proposal (handled by
  A-03.2 / A-03.3, not by the arena itself).
"""

from __future__ import annotations

from simulation.strategy_arena.arena import (
    Arena,
    ArenaConfig,
    ArenaConfigError,
    ArenaInputError,
    Contestant,
    TournamentBracket,
    TournamentResult,
)
from simulation.strategy_arena.kill_underperformers import (
    DEMOTION_KIND,
    DEMOTION_TOUCHPOINT,
    DemotionRecommendation,
    KillUnderperformersInputError,
    build_demotion_recommendations,
)
from simulation.strategy_arena.promotion_engine import (
    PROMOTION_KIND,
    PROMOTION_TOUCHPOINT,
    ROLE_BOTH,
    ROLE_ELITE,
    ROLE_WINNER,
    PromotionEngineInputError,
    PromotionRecommendation,
    build_promotion_recommendations,
)

__all__ = [
    "Arena",
    "ArenaConfig",
    "ArenaConfigError",
    "ArenaInputError",
    "Contestant",
    "DEMOTION_KIND",
    "DEMOTION_TOUCHPOINT",
    "DemotionRecommendation",
    "KillUnderperformersInputError",
    "PROMOTION_KIND",
    "PROMOTION_TOUCHPOINT",
    "PromotionEngineInputError",
    "PromotionRecommendation",
    "ROLE_BOTH",
    "ROLE_ELITE",
    "ROLE_WINNER",
    "TournamentBracket",
    "TournamentResult",
    "build_demotion_recommendations",
    "build_promotion_recommendations",
]
