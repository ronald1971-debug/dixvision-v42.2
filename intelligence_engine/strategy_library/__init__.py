"""Wave-04 PR-3 — strategy decomposition library.

Decomposes monolithic strategies into five reusable, structurally
equal value objects:

* :class:`EntryLogic` — when to open a position.
* :class:`ExitLogic` — when to close a position.
* :class:`RiskModel` — how to size + protect the position.
* :class:`Timeframe` — what holding-period bucket the strategy lives
  in.
* :class:`MarketCondition` — what regime the strategy is designed for.

Plus :class:`StrategyDecomposition` — a frozen tuple of the five
components — and :func:`signature_for` which mints the canonical
deterministic hash used by
:attr:`core.contracts.trader_intelligence.TraderModel.strategy_signatures`.

Design constraints (mirror :mod:`core.contracts.trader_intelligence`):

* Frozen, slotted, structural-equality dataclasses (TEST-01 replay
  parity, INV-15 deterministic primitives).
* No callables, no IO, no clocks. Pure value objects — same input
  always produces the same signature byte-for-byte.
* Authority symmetry: a :class:`StrategyDecomposition` is **just
  data**. Anyone can construct one. The decomposition gate is
  semantic (compatibility constraints in Wave-04 PR-4), not authority
  symmetry like B27/B28/B29.
* Library, not engine. This module exposes value objects + a frozen
  registry of canonical components. Wave-04 PR-4 builds the
  composition engine on top.

Refs:

* ``core.contracts.trader_intelligence`` — Wave-04 PR-1 contracts.
* ``intelligence_engine.plugins.microstructure.microstructure_v1`` —
  the first concrete strategy retroactively decomposed here as the
  reference entry in :mod:`.canonical`.
"""

from __future__ import annotations

from intelligence_engine.strategy_library.components import (
    EntryLogic,
    EntryStyle,
    ExitLogic,
    ExitStyle,
    MarketCondition,
    MarketRegime,
    RiskModel,
    SizingStyle,
    StopStyle,
    Timeframe,
)
from intelligence_engine.strategy_library.decomposition import (
    SIGNATURE_HASH_LEN,
    StrategyDecomposition,
    signature_for,
)
from intelligence_engine.strategy_library.registry import (
    CANONICAL_DECOMPOSITIONS,
    CANONICAL_ENTRY_LOGIC,
    CANONICAL_EXIT_LOGIC,
    CANONICAL_MARKET_CONDITIONS,
    CANONICAL_RISK_MODELS,
    CANONICAL_TIMEFRAMES,
)

__all__ = [
    "CANONICAL_DECOMPOSITIONS",
    "CANONICAL_ENTRY_LOGIC",
    "CANONICAL_EXIT_LOGIC",
    "CANONICAL_MARKET_CONDITIONS",
    "CANONICAL_RISK_MODELS",
    "CANONICAL_TIMEFRAMES",
    "EntryLogic",
    "EntryStyle",
    "ExitLogic",
    "ExitStyle",
    "MarketCondition",
    "MarketRegime",
    "RiskModel",
    "SIGNATURE_HASH_LEN",
    "SizingStyle",
    "StopStyle",
    "StrategyDecomposition",
    "Timeframe",
    "signature_for",
]
