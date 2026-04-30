"""Composite :class:`StrategyDecomposition` + canonical signature hash.

A :class:`StrategyDecomposition` is the five-tuple
``(entry, exit, risk, timeframe, market_condition)``. Its
:func:`signature_for` is the deterministic 64-hex-char hash
referenced by
:attr:`core.contracts.trader_intelligence.TraderModel.strategy_signatures`.

INV-15 / TEST-01: the signature is computed by
``json.dumps(payload, sort_keys=True, separators=(",", ":"))`` →
SHA-256 → hex. Same decomposition → same signature, byte-for-byte,
across runs and across machines.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from intelligence_engine.strategy_library.components import (
    EntryLogic,
    ExitLogic,
    MarketCondition,
    RiskModel,
    Timeframe,
)

SIGNATURE_HASH_LEN: int = 64
"""Width of the canonical hex digest (SHA-256)."""


@dataclass(frozen=True, slots=True)
class StrategyDecomposition:
    """Five-tuple of reusable components plus a stable handle.

    Attributes:
        decomposition_id: Stable handle for this specific composite
            (``"microstructure_v1"``, ``"breakout_pullback_v1"``, …).
            Distinct from each component's ``component_id``: a
            decomposition can share an :class:`EntryLogic` with another
            decomposition.
        entry: How positions are opened.
        exit_: How positions are closed (named with trailing
            underscore because ``exit`` is a Python builtin).
        risk: How positions are sized + protected.
        timeframe: Bar interval + holding period.
        market_condition: Regime + symbol universe the strategy
            targets.
    """

    decomposition_id: str
    entry: EntryLogic
    exit_: ExitLogic
    risk: RiskModel
    timeframe: Timeframe
    market_condition: MarketCondition


def _component_to_payload(obj: object) -> Mapping[str, object]:
    """Project one component to a deterministic JSON-friendly dict.

    Sorts mapping fields by key so byte-identical replay is preserved
    regardless of Python dict-construction order.
    """

    raw = {}
    for slot in obj.__slots__:  # type: ignore[attr-defined]
        value = getattr(obj, slot)
        if isinstance(value, Mapping):
            value = {k: value[k] for k in sorted(value)}
        elif isinstance(value, tuple):
            # Tuples (e.g. symbol_universe) are kept ordered as
            # declared — order is semantically meaningful.
            value = list(value)
        else:
            # StrEnum / str / float / int — JSON-safe as-is.
            value = str(value) if hasattr(value, "value") else value
        raw[slot] = value
    return raw


def signature_for(decomp: StrategyDecomposition) -> str:
    """Compute the canonical SHA-256 hex digest for one decomposition.

    The signature is what
    :attr:`core.contracts.trader_intelligence.TraderModel.strategy_signatures`
    stores. Two decompositions produce the same signature iff every
    field of every component is structurally equal — which is exactly
    the equivalence the composition engine (Wave-04 PR-4) needs.
    """

    payload = {
        "decomposition_id": decomp.decomposition_id,
        "entry": _component_to_payload(decomp.entry),
        "exit": _component_to_payload(decomp.exit_),
        "risk": _component_to_payload(decomp.risk),
        "timeframe": _component_to_payload(decomp.timeframe),
        "market_condition": _component_to_payload(decomp.market_condition),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "SIGNATURE_HASH_LEN",
    "StrategyDecomposition",
    "signature_for",
]
