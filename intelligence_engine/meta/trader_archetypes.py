"""Trader-archetype registry loader (TI-CONS).

Reads ``registry/trader_archetypes.yaml`` and exposes immutable
:class:`TraderArchetype` records to the meta layer (strategy
synthesizer + Darwinian arena) and to the composition engine
(Wave-04 PR-4).

Governance rules (INV-51):

* The loader is **read-only**. Engines never mutate archetype state
  on the hot path; transitions (``PROPOSED → ACTIVE → DECAYING →
  DEMOTED``) flow through the offline Governance patch pipeline.
* The five behavioural dimensions exposed here mirror the canonical
  axes used by :class:`core.contracts.trader_intelligence.PhilosophyProfile`
  (belief_system, risk_attitude, time_horizon, conviction_style,
  regime_performance), so registry rows feed the composition engine
  through the same shape that the streaming TI pipeline produces.
* The registry file is canonical at the repo root
  (``registry/trader_archetypes.yaml``); the loader resolves the
  path from the repo root by default but accepts an explicit path
  override for tests.

Replay determinism (INV-15): the loader sorts entries by
``archetype_id`` so the iteration order is stable across hosts /
filesystem orderings. Floats are stored at YAML precision; the
loader does not coerce them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class ArchetypeState(StrEnum):
    """Lifecycle state per INV-51."""

    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    DECAYING = "DECAYING"
    DEMOTED = "DEMOTED"


class RiskAttitude(StrEnum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    UNKNOWN = "UNKNOWN"


class TimeHorizon(StrEnum):
    SCALP = "SCALP"
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    POSITION = "POSITION"
    UNKNOWN = "UNKNOWN"


class ConvictionStyle(StrEnum):
    REACTIVE = "REACTIVE"
    PREDICTIVE = "PREDICTIVE"
    CONTRARIAN = "CONTRARIAN"
    SYSTEMATIC = "SYSTEMATIC"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass(frozen=True, slots=True)
class TraderArchetype:
    """Immutable archetype record loaded from registry YAML.

    Attributes mirror the YAML schema documented in
    ``registry/trader_archetypes.yaml``. Mappings are coerced to
    ``dict`` so callers can serialize them deterministically; the
    record itself is frozen.
    """

    archetype_id: str
    name: str
    state: ArchetypeState
    decay_rate: float
    performance_score: float
    seed_trader: str
    belief_system: Mapping[str, float]
    risk_attitude: RiskAttitude
    time_horizon: TimeHorizon
    conviction_style: ConvictionStyle
    regime_performance: Mapping[str, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay_rate <= 1.0:
            raise ValueError(
                f"{self.archetype_id}: decay_rate must be in [0, 1], "
                f"got {self.decay_rate!r}"
            )
        if not -1.0 <= self.performance_score <= 1.0:
            raise ValueError(
                f"{self.archetype_id}: performance_score must be in "
                f"[-1, 1], got {self.performance_score!r}"
            )
        for tag, strength in self.belief_system.items():
            if not 0.0 <= float(strength) <= 1.0:
                raise ValueError(
                    f"{self.archetype_id}: belief_system[{tag!r}]={strength!r} "
                    "out of [0, 1]"
                )
        for regime, score in self.regime_performance.items():
            if not -1.0 <= float(score) <= 1.0:
                raise ValueError(
                    f"{self.archetype_id}: regime_performance[{regime!r}]"
                    f"={score!r} out of [-1, 1]"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class TraderArchetypeRegistry:
    """Read-only collection of archetypes keyed by ``archetype_id``."""

    by_id: Mapping[str, TraderArchetype]

    def get(self, archetype_id: str) -> TraderArchetype | None:
        """Return the archetype for ``archetype_id`` or ``None`` if absent.

        Mirrors the codebase-wide registry convention (see
        :class:`governance_engine.strategy_registry.StrategyRegistry`,
        :class:`execution_engine.lifecycle.order_state_machine.OrderStateMachine`,
        and :mod:`core.contracts.source_trust_promotions`) so future
        consumers (strategy synthesizer + Darwinian arena) can use the
        ``if archetype is None: ...`` idiom uniformly. Use ``[]`` /
        ``__getitem__`` (or ``.by_id[...]``) when callers want a
        :class:`KeyError` on miss.
        """

        return self.by_id.get(archetype_id)

    def __contains__(self, archetype_id: object) -> bool:
        return archetype_id in self.by_id

    def __len__(self) -> int:
        return len(self.by_id)

    def __iter__(self):
        return iter(self.by_id.values())

    def active(self) -> tuple[TraderArchetype, ...]:
        return tuple(
            a for a in self.by_id.values() if a.state is ArchetypeState.ACTIVE
        )


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "registry" / "trader_archetypes.yaml"


def _coerce_archetype(archetype_id: str, body: Mapping[str, Any]) -> TraderArchetype:
    try:
        dims = body["dimensions"]
    except KeyError as exc:
        raise ValueError(
            f"{archetype_id}: missing required 'dimensions' block"
        ) from exc

    belief_system = {
        str(k): float(v) for k, v in dict(dims.get("belief_system", {})).items()
    }
    regime_performance = {
        str(k): float(v)
        for k, v in dict(dims.get("regime_performance", {})).items()
    }
    return TraderArchetype(
        archetype_id=archetype_id,
        name=str(body["name"]),
        state=ArchetypeState(str(body["state"])),
        decay_rate=float(body["decay_rate"]),
        performance_score=float(body["performance_score"]),
        seed_trader=str(body["seed_trader"]),
        belief_system=belief_system,
        risk_attitude=RiskAttitude(str(dims.get("risk_attitude", "UNKNOWN"))),
        time_horizon=TimeHorizon(str(dims.get("time_horizon", "UNKNOWN"))),
        conviction_style=ConvictionStyle(
            str(dims.get("conviction_style", "UNKNOWN"))
        ),
        regime_performance=regime_performance,
    )


def load_trader_archetypes(
    path: str | Path | None = None,
) -> TraderArchetypeRegistry:
    """Load archetypes from ``registry/trader_archetypes.yaml``.

    The loader sorts archetypes by ``archetype_id`` for replay-stable
    iteration order (INV-15) and rejects duplicate IDs.
    """

    resolved = Path(path) if path is not None else _default_registry_path()
    with resolved.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    raw = document.get("archetypes", {})
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"{resolved}: expected mapping under 'archetypes:', "
            f"got {type(raw).__name__}"
        )

    by_id: dict[str, TraderArchetype] = {}
    for archetype_id in sorted(raw):
        body = raw[archetype_id]
        if not isinstance(body, Mapping):
            raise ValueError(
                f"{archetype_id}: expected mapping body, got {type(body).__name__}"
            )
        if archetype_id in by_id:
            raise ValueError(f"duplicate archetype_id {archetype_id!r}")
        by_id[archetype_id] = _coerce_archetype(archetype_id, body)

    return TraderArchetypeRegistry(by_id=by_id)
