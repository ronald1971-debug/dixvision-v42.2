"""Tests for the trader-archetype registry loader (TI-CONS, INV-51)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intelligence_engine.meta.trader_archetypes import (
    ArchetypeState,
    ConvictionStyle,
    RiskAttitude,
    TimeHorizon,
    TraderArchetype,
    TraderArchetypeRegistry,
    load_trader_archetypes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "registry" / "trader_archetypes.yaml"


def test_canonical_registry_loads() -> None:
    registry = load_trader_archetypes()

    assert isinstance(registry, TraderArchetypeRegistry)
    assert len(registry) >= 10, "ship at least 10 hand-written archetypes"
    for archetype in registry:
        assert isinstance(archetype, TraderArchetype)
        assert archetype.archetype_id.startswith("TA-")


def test_iteration_is_sorted_by_id() -> None:
    registry = load_trader_archetypes()
    ids = [a.archetype_id for a in registry]
    assert ids == sorted(ids)


def test_active_filter() -> None:
    registry = load_trader_archetypes()
    active = registry.active()
    assert all(a.state is ArchetypeState.ACTIVE for a in active)
    assert len(active) >= 1


def test_known_archetype_shape() -> None:
    registry = load_trader_archetypes()
    a = registry.get("TA-TREND-001")
    assert a is not None
    assert a.name == "Classic Trend Follower"
    assert a.state is ArchetypeState.ACTIVE
    assert a.risk_attitude is RiskAttitude.MODERATE
    assert a.time_horizon is TimeHorizon.SWING
    assert a.conviction_style is ConvictionStyle.SYSTEMATIC
    assert 0.0 <= a.decay_rate <= 1.0
    assert -1.0 <= a.performance_score <= 1.0
    assert pytest.approx(a.belief_system["trend_following"]) == 0.95


def test_invalid_decay_rate_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "archetypes:\n"
        "  TA-BAD-001:\n"
        "    name: bad\n"
        "    state: ACTIVE\n"
        "    decay_rate: 2.0\n"
        "    performance_score: 0.0\n"
        "    seed_trader: x\n"
        "    dimensions:\n"
        "      belief_system: {trend_following: 0.5}\n"
        "      risk_attitude: MODERATE\n"
        "      time_horizon: SWING\n"
        "      conviction_style: SYSTEMATIC\n"
        "      regime_performance: {trending: 0.5}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="decay_rate"):
        load_trader_archetypes(bad)


def test_invalid_performance_score_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "archetypes:\n"
        "  TA-BAD-002:\n"
        "    name: bad\n"
        "    state: ACTIVE\n"
        "    decay_rate: 0.1\n"
        "    performance_score: 2.0\n"
        "    seed_trader: x\n"
        "    dimensions:\n"
        "      belief_system: {trend_following: 0.5}\n"
        "      risk_attitude: MODERATE\n"
        "      time_horizon: SWING\n"
        "      conviction_style: SYSTEMATIC\n"
        "      regime_performance: {trending: 0.5}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="performance_score"):
        load_trader_archetypes(bad)


def test_belief_strength_out_of_range_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "archetypes:\n"
        "  TA-BAD-003:\n"
        "    name: bad\n"
        "    state: ACTIVE\n"
        "    decay_rate: 0.1\n"
        "    performance_score: 0.0\n"
        "    seed_trader: x\n"
        "    dimensions:\n"
        "      belief_system: {trend_following: 1.5}\n"
        "      risk_attitude: MODERATE\n"
        "      time_horizon: SWING\n"
        "      conviction_style: SYSTEMATIC\n"
        "      regime_performance: {trending: 0.5}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="belief_system"):
        load_trader_archetypes(bad)


def test_canonical_registry_path_resolves() -> None:
    assert REGISTRY_PATH.exists(), str(REGISTRY_PATH)
    registry = load_trader_archetypes(REGISTRY_PATH)
    assert len(registry) >= 10


def test_get_returns_none_on_miss() -> None:
    registry = load_trader_archetypes()
    assert registry.get("TA-DOES-NOT-EXIST") is None


def test_contains_supports_both_id_and_archetype() -> None:
    registry = load_trader_archetypes()
    a = registry.get("TA-TREND-001")
    assert a is not None
    # iteration yields archetypes -> "in" must work on archetype objects
    for archetype in registry:
        assert archetype in registry
    # ...and on archetype-ids (the natural string lookup)
    assert "TA-TREND-001" in registry
    assert "TA-DOES-NOT-EXIST" not in registry
    # arbitrary objects are not members
    assert object() not in registry


def test_ids_iteration_order_is_sorted() -> None:
    registry = load_trader_archetypes()
    ids = registry.ids()
    assert list(ids) == sorted(ids)
    assert len(ids) == len(registry)
