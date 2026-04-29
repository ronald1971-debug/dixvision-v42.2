"""SCVS Phase 3 — schema enforcement / AI validation / fallback audit tests.

Covers SCVS-04, SCVS-07, SCVS-08, SCVS-09, SCVS-10. Pure / deterministic;
no clock, no PRNG, no network.
"""

from __future__ import annotations

import pytest

from core.contracts.events import EventKind, HazardSeverity, SystemEventKind
from system_engine.scvs.ai_validator import (
    AIOutcome,
    AIValidator,
)
from system_engine.scvs.fallback_audit import make_fallback_event
from system_engine.scvs.lint import find_redundant_sources
from system_engine.scvs.schema_guard import (
    ContractRegistry,
    SchemaGuard,
    SchemaSpec,
    ValidationOutcome,
)
from system_engine.scvs.source_manager import HAZ_CRITICAL_SOURCE_STALE
from system_engine.scvs.source_registry import (
    SourceCategory,
    SourceDeclaration,
    SourceRegistry,
)


def _decl(
    sid: str,
    *,
    enabled: bool = True,
    critical: bool = False,
    category: SourceCategory = SourceCategory.MARKET,
    schema: str = "core.market.Tick",
    provider: str = "x",
    endpoint: str = "https://x",
) -> SourceDeclaration:
    return SourceDeclaration(
        id=sid,
        name=sid,
        category=category,
        provider=provider,
        endpoint=endpoint,
        schema=schema,
        auth="none",
        enabled=enabled,
        critical=critical,
        liveness_threshold_ms=1_000,
    )


def _registry(*decls: SourceDeclaration) -> SourceRegistry:
    return SourceRegistry(version="v0.1.0", sources=decls)


def _contracts(**specs: SchemaSpec) -> ContractRegistry:
    return ContractRegistry(specs=dict(specs))


# ---------------------------------------------------------------------------
# SCVS-04 — schema enforcement
# ---------------------------------------------------------------------------


def test_schema_guard_accepts_well_formed_packet():
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts(**{"core.market.Tick": SchemaSpec(frozenset({"px", "qty"}))})
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=1_000_000_000)
    res = guard.validate(
        source_id="SRC-A",
        packet={"px": 1.0, "qty": 2.0},
        packet_ts_ns=10**9,
        now_ns=10**9 + 100,
    )
    assert res.outcome is ValidationOutcome.ACCEPTED


def test_schema_guard_rejects_missing_required_key():
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts(**{"core.market.Tick": SchemaSpec(frozenset({"px", "qty"}))})
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=1_000_000_000)
    res = guard.validate(
        source_id="SRC-A",
        packet={"px": 1.0},
        packet_ts_ns=10**9,
        now_ns=10**9 + 100,
    )
    assert res.outcome is ValidationOutcome.REJECTED_SCHEMA_MISMATCH
    assert "qty" in res.detail


def test_schema_guard_rejects_unknown_schema():
    reg = _registry(_decl("SRC-A", schema="not.registered"))
    guard = SchemaGuard(registry=reg, contracts=_contracts(), max_age_ns=10**9)
    res = guard.validate(
        source_id="SRC-A",
        packet={"px": 1.0},
        packet_ts_ns=10**9,
        now_ns=10**9 + 100,
    )
    assert res.outcome is ValidationOutcome.REJECTED_UNKNOWN_SCHEMA


def test_schema_guard_rejects_extras_when_disallowed():
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts(
        **{"core.market.Tick": SchemaSpec(frozenset({"px"}), allow_extras=False)}
    )
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=10**9)
    res = guard.validate(
        source_id="SRC-A",
        packet={"px": 1.0, "extra": "x"},
        packet_ts_ns=10**9,
        now_ns=10**9 + 100,
    )
    assert res.outcome is ValidationOutcome.REJECTED_SCHEMA_MISMATCH
    assert "extra" in res.detail


def test_schema_guard_rejects_unknown_source():
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts(**{"core.market.Tick": SchemaSpec(frozenset({"px"}))})
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=10**9)
    res = guard.validate(
        source_id="SRC-MISSING",
        packet={"px": 1.0},
        packet_ts_ns=10**9,
        now_ns=10**9 + 100,
    )
    assert res.outcome is ValidationOutcome.REJECTED_UNKNOWN_SOURCE


def test_schema_guard_rejects_disabled_source():
    reg = _registry(_decl("SRC-A", enabled=False))
    contracts = _contracts(**{"core.market.Tick": SchemaSpec(frozenset({"px"}))})
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=10**9)
    res = guard.validate(
        source_id="SRC-A",
        packet={"px": 1.0},
        packet_ts_ns=10**9,
        now_ns=10**9 + 100,
    )
    assert res.outcome is ValidationOutcome.REJECTED_DISABLED_SOURCE


def test_schema_guard_rejects_empty_packet():
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts(**{"core.market.Tick": SchemaSpec(frozenset({"px"}))})
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=10**9)
    res = guard.validate(
        source_id="SRC-A",
        packet={},
        packet_ts_ns=10**9,
        now_ns=10**9 + 100,
    )
    assert res.outcome is ValidationOutcome.REJECTED_EMPTY_PACKET


# ---------------------------------------------------------------------------
# SCVS-09 — stale-data rejection (per-packet)
# ---------------------------------------------------------------------------


def test_schema_guard_rejects_stale_packet():
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts(**{"core.market.Tick": SchemaSpec(frozenset({"px"}))})
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=500)
    # gap = 1000, threshold 500 → stale
    res = guard.validate(
        source_id="SRC-A",
        packet={"px": 1.0},
        packet_ts_ns=10**9,
        now_ns=10**9 + 1000,
    )
    assert res.outcome is ValidationOutcome.REJECTED_STALE


def test_schema_guard_rejects_future_timestamp():
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts(**{"core.market.Tick": SchemaSpec(frozenset({"px"}))})
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=10**9)
    res = guard.validate(
        source_id="SRC-A",
        packet={"px": 1.0},
        packet_ts_ns=10**9 + 5,
        now_ns=10**9,
    )
    assert res.outcome is ValidationOutcome.REJECTED_FUTURE_TS


def test_schema_guard_max_age_zero_disables_staleness():
    """``max_age_ns=0`` → staleness check off (synthetic / replay scenarios)."""
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts(**{"core.market.Tick": SchemaSpec(frozenset({"px"}))})
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=0)
    res = guard.validate(
        source_id="SRC-A",
        packet={"px": 1.0},
        packet_ts_ns=0,
        now_ns=10**18,
    )
    assert res.outcome is ValidationOutcome.ACCEPTED


def test_schema_guard_rejects_negative_max_age():
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts()
    with pytest.raises(ValueError):
        SchemaGuard(registry=reg, contracts=contracts, max_age_ns=-1)


# ---------------------------------------------------------------------------
# SCVS-07 — AI provider validation
# ---------------------------------------------------------------------------


def _ai(sid: str, *, critical: bool = False) -> SourceDeclaration:
    return _decl(
        sid,
        critical=critical,
        category=SourceCategory.AI,
        schema="core.ai.Response",
        provider="openai",
        endpoint="https://api.openai.com",
    )


def test_ai_validator_accepts_well_formed_response():
    reg = _registry(_ai("SRC-AI-A"))
    v = AIValidator(
        registry=reg,
        max_latency_ns=10**9,
        required_top_keys=frozenset({"text"}),
    )
    res, hazards = v.validate(
        source_id="SRC-AI-A",
        response={"text": "hello"},
        latency_ns=10**8,
        now_ns=10**9,
    )
    assert res.outcome is AIOutcome.ACCEPTED
    assert hazards == ()


def test_ai_validator_rejects_high_latency():
    reg = _registry(_ai("SRC-AI-A"))
    v = AIValidator(
        registry=reg,
        max_latency_ns=10**8,
        required_top_keys=frozenset(),
    )
    res, hazards = v.validate(
        source_id="SRC-AI-A",
        response={"text": "hello"},
        latency_ns=10**9,  # 10x over
        now_ns=10**9,
    )
    assert res.outcome is AIOutcome.REJECTED_LATENCY
    assert hazards == ()


def test_ai_validator_rejects_empty_response():
    reg = _registry(_ai("SRC-AI-A"))
    v = AIValidator(registry=reg, max_latency_ns=10**9)
    res, _ = v.validate(
        source_id="SRC-AI-A", response=None, latency_ns=10**8, now_ns=10**9
    )
    assert res.outcome is AIOutcome.REJECTED_EMPTY


def test_ai_validator_rejects_whitespace_only_string():
    """Whitespace-only fields are silent-empty fallbacks (common with degraded LLMs)."""
    reg = _registry(_ai("SRC-AI-A"))
    v = AIValidator(
        registry=reg,
        max_latency_ns=10**9,
        required_top_keys=frozenset({"text"}),
    )
    res, _ = v.validate(
        source_id="SRC-AI-A",
        response={"text": "   "},
        latency_ns=10**8,
        now_ns=10**9,
    )
    assert res.outcome is AIOutcome.REJECTED_EMPTY


def test_ai_validator_rejects_missing_required_keys():
    reg = _registry(_ai("SRC-AI-A"))
    v = AIValidator(
        registry=reg,
        max_latency_ns=10**9,
        required_top_keys=frozenset({"text", "tokens"}),
    )
    res, _ = v.validate(
        source_id="SRC-AI-A",
        response={"text": "hi"},
        latency_ns=10**8,
        now_ns=10**9,
    )
    assert res.outcome is AIOutcome.REJECTED_STRUCTURE


def test_ai_validator_rejects_non_ai_category():
    reg = _registry(_decl("SRC-A"))  # market, not ai
    v = AIValidator(registry=reg, max_latency_ns=10**9)
    res, _ = v.validate(
        source_id="SRC-A", response={"text": "hi"}, latency_ns=1, now_ns=10**9
    )
    assert res.outcome is AIOutcome.REJECTED_NOT_AI


def test_ai_validator_critical_failure_emits_haz13():
    reg = _registry(_ai("SRC-AI-A", critical=True))
    v = AIValidator(registry=reg, max_latency_ns=10**8)
    _res, hazards = v.validate(
        source_id="SRC-AI-A",
        response={"text": "hi"},
        latency_ns=10**9,
        now_ns=10**9 + 5,
    )
    assert len(hazards) == 1
    h = hazards[0]
    assert h.kind is EventKind.HAZARD
    assert h.code == HAZ_CRITICAL_SOURCE_STALE
    assert h.severity is HazardSeverity.HIGH
    assert h.ts_ns == 10**9 + 5
    assert h.meta["source_id"] == "SRC-AI-A"


def test_ai_validator_non_critical_failure_no_hazard():
    reg = _registry(_ai("SRC-AI-A", critical=False))
    v = AIValidator(registry=reg, max_latency_ns=10**8)
    _res, hazards = v.validate(
        source_id="SRC-AI-A",
        response={"text": "hi"},
        latency_ns=10**9,
        now_ns=10**9,
    )
    assert hazards == ()


def test_ai_validator_rejects_zero_max_latency():
    reg = _registry(_ai("SRC-AI-A"))
    with pytest.raises(ValueError):
        AIValidator(registry=reg, max_latency_ns=0)


# ---------------------------------------------------------------------------
# SCVS-08 — duplicate source detection (WARN-only)
# ---------------------------------------------------------------------------


def test_redundant_sources_clean_when_unique():
    reg = _registry(_decl("SRC-A"), _decl("SRC-B", endpoint="https://y"))
    assert find_redundant_sources(reg) == ()


def test_redundant_sources_flags_same_triple():
    reg = _registry(
        _decl("SRC-A", provider="openai", endpoint="https://o"),
        _decl("SRC-B", provider="openai", endpoint="https://o"),
    )
    warns = find_redundant_sources(reg)
    assert len(warns) == 1
    assert warns[0].rule == "SCVS-08"
    assert "SRC-A" in warns[0].detail and "SRC-B" in warns[0].detail


def test_redundant_sources_ignores_different_categories():
    reg = _registry(
        _decl("SRC-A", category=SourceCategory.MARKET, endpoint="https://e"),
        _decl("SRC-B", category=SourceCategory.NEWS, endpoint="https://e"),
    )
    assert find_redundant_sources(reg) == ()


# ---------------------------------------------------------------------------
# SCVS-10 — silent-fallback audit
# ---------------------------------------------------------------------------


def test_make_fallback_event_basic():
    ev = make_fallback_event(
        now_ns=10**9,
        failed_source_id="SRC-A",
        fallback_source_id="SRC-B",
        reason="stale",
    )
    assert ev.kind is EventKind.SYSTEM
    assert ev.sub_kind is SystemEventKind.SOURCE_FALLBACK_ACTIVATED
    assert ev.ts_ns == 10**9
    assert ev.payload["failed_source_id"] == "SRC-A"
    assert ev.payload["fallback_source_id"] == "SRC-B"
    assert ev.payload["reason"] == "stale"


def test_make_fallback_event_merges_extra_detail():
    ev = make_fallback_event(
        now_ns=10**9,
        failed_source_id="SRC-A",
        fallback_source_id="SRC-B",
        reason="stale",
        detail={"upstream_gap_ms": "1500"},
    )
    assert ev.payload["upstream_gap_ms"] == "1500"


def test_make_fallback_event_rejects_self_fallback():
    with pytest.raises(ValueError):
        make_fallback_event(
            now_ns=10**9,
            failed_source_id="SRC-A",
            fallback_source_id="SRC-A",
            reason="x",
        )


def test_make_fallback_event_requires_reason():
    with pytest.raises(ValueError):
        make_fallback_event(
            now_ns=10**9,
            failed_source_id="SRC-A",
            fallback_source_id="SRC-B",
            reason="",
        )


def test_make_fallback_event_rejects_reserved_key_collision():
    with pytest.raises(ValueError):
        make_fallback_event(
            now_ns=10**9,
            failed_source_id="SRC-A",
            fallback_source_id="SRC-B",
            reason="x",
            detail={"reason": "override-attempt"},
        )


# ---------------------------------------------------------------------------
# determinism (INV-15) — same inputs → same outputs across repeated calls
# ---------------------------------------------------------------------------


def test_schema_guard_replay_determinism():
    reg = _registry(_decl("SRC-A"))
    contracts = _contracts(**{"core.market.Tick": SchemaSpec(frozenset({"px"}))})
    guard = SchemaGuard(registry=reg, contracts=contracts, max_age_ns=500)
    inputs = [
        ("SRC-A", {"px": 1.0}, 10**9, 10**9 + 100),
        ("SRC-A", {"px": 2.0}, 10**9 + 200, 10**9 + 1500),  # stale
        ("SRC-A", {}, 10**9 + 300, 10**9 + 350),  # empty
    ]
    a = [
        guard.validate(
            source_id=sid, packet=p, packet_ts_ns=ts, now_ns=now
        ).outcome
        for sid, p, ts, now in inputs
    ]
    b = [
        guard.validate(
            source_id=sid, packet=p, packet_ts_ns=ts, now_ns=now
        ).outcome
        for sid, p, ts, now in inputs
    ]
    assert a == b
    assert a == [
        ValidationOutcome.ACCEPTED,
        ValidationOutcome.REJECTED_STALE,
        ValidationOutcome.REJECTED_EMPTY_PACKET,
    ]
