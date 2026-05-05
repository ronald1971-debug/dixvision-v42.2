"""Trader Intelligence consumer package (TI-CONS).

The meta layer (per dixvision_executive_summary.md "intelligence_engine/
meta/") is where the read-only trader-archetype registry is loaded and
exposed to downstream consumers (the strategy synthesizer + Darwinian
arena). This bootstrap module ships the loader; the synthesizer and
arena land in subsequent PRs.
"""

from intelligence_engine.meta.trader_archetypes import (
    TraderArchetype,
    TraderArchetypeRegistry,
    load_trader_archetypes,
)

__all__ = [
    "TraderArchetype",
    "TraderArchetypeRegistry",
    "load_trader_archetypes",
]
