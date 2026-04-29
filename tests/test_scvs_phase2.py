"""SCVS Phase 2 — runtime source liveness manager tests."""

from __future__ import annotations

import pytest

from core.contracts.events import EventKind, HazardSeverity, SystemEventKind
from system_engine.scvs.source_manager import (
    HAZ_CRITICAL_SOURCE_STALE,
    SourceManager,
    SourceStatus,
)
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
    threshold_ms: int = 1_000,
    category: SourceCategory = SourceCategory.MARKET,
) -> SourceDeclaration:
    return SourceDeclaration(
        id=sid,
        name=sid,
        category=category,
        provider="x",
        endpoint="https://x",
        schema="x.X",
        auth="none",
        enabled=enabled,
        critical=critical,
        liveness_threshold_ms=threshold_ms,
    )


def _registry(*decls: SourceDeclaration) -> SourceRegistry:
    return SourceRegistry(version="v0.1.0", sources=decls)


# ---------------------------------------------------------------------------
# registry parsing
# ---------------------------------------------------------------------------


def test_registry_default_threshold_per_category(tmp_path):
    """Phase 2 default thresholds are picked up by category."""

    import yaml

    from system_engine.scvs.source_registry import load_source_registry

    p = tmp_path / "r.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1.0",
                "sources": [
                    {
                        "id": "SRC-MARKET-X-001",
                        "name": "x",
                        "category": "market",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "none",
                    },
                    {
                        "id": "SRC-SYNTHETIC-X-001",
                        "name": "x",
                        "category": "synthetic",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "none",
                    },
                ],
            }
        )
    )
    reg = load_source_registry(p)
    by = {s.id: s for s in reg.sources}
    assert by["SRC-MARKET-X-001"].liveness_threshold_ms == 5_000
    assert by["SRC-SYNTHETIC-X-001"].liveness_threshold_ms == 0


def test_registry_explicit_threshold_wins(tmp_path):
    import yaml

    from system_engine.scvs.source_registry import load_source_registry

    p = tmp_path / "r.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1.0",
                "sources": [
                    {
                        "id": "SRC-MARKET-X-001",
                        "name": "x",
                        "category": "market",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "none",
                        "liveness_threshold_ms": 250,
                    }
                ],
            }
        )
    )
    reg = load_source_registry(p)
    assert reg.sources[0].liveness_threshold_ms == 250


def test_registry_rejects_negative_threshold(tmp_path):
    import yaml

    from system_engine.scvs.source_registry import load_source_registry

    p = tmp_path / "r.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1.0",
                "sources": [
                    {
                        "id": "SRC-MARKET-X-001",
                        "name": "x",
                        "category": "market",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "none",
                        "liveness_threshold_ms": -1,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="liveness_threshold_ms must be >= 0"):
        load_source_registry(p)


# ---------------------------------------------------------------------------
# SourceManager — inputs
# ---------------------------------------------------------------------------


def test_manager_seeds_state_for_enabled_sources():
    reg = _registry(_decl("SRC-A"), _decl("SRC-B", enabled=False))
    mgr = SourceManager(registry=reg)
    # SRC-A is seeded, SRC-B is not (still placeholder).
    reports = mgr.reports(now_ns=0)
    assert {r.source_id for r in reports} == {"SRC-A"}


def test_manager_rejects_heartbeat_for_disabled_source():
    reg = _registry(_decl("SRC-A", enabled=False))
    mgr = SourceManager(registry=reg)
    with pytest.raises(ValueError, match="enabled=false"):
        mgr.record_heartbeat("SRC-A", ts_ns=1_000)


def test_manager_rejects_heartbeat_for_unknown_source():
    reg = _registry(_decl("SRC-A"))
    mgr = SourceManager(registry=reg)
    with pytest.raises(KeyError):
        mgr.record_heartbeat("SRC-MISSING", ts_ns=1_000)


def test_manager_rejects_data_for_disabled_source():
    reg = _registry(_decl("SRC-A", enabled=False))
    mgr = SourceManager(registry=reg)
    with pytest.raises(ValueError, match="enabled=false"):
        mgr.record_data("SRC-A", ts_ns=1_000)


# ---------------------------------------------------------------------------
# SourceManager — classification
# ---------------------------------------------------------------------------


def test_unknown_until_first_heartbeat():
    reg = _registry(_decl("SRC-A"))
    mgr = SourceManager(registry=reg)
    [r] = mgr.reports(now_ns=10**9)
    assert r.status == SourceStatus.UNKNOWN
    assert r.last_heartbeat_ns == 0
    assert r.gap_ns == 0


def test_live_within_threshold():
    reg = _registry(_decl("SRC-A", threshold_ms=1_000))  # 1 s
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    [r] = mgr.reports(now_ns=10**9 + 500_000_000)  # +500 ms
    assert r.status == SourceStatus.LIVE


def test_stale_when_gap_exceeds_threshold():
    reg = _registry(_decl("SRC-A", threshold_ms=1_000))
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    [r] = mgr.reports(now_ns=10**9 + 2_000_000_000)  # +2 s
    assert r.status == SourceStatus.STALE
    assert r.gap_ns == 2_000_000_000


def test_threshold_zero_disables_liveness_check():
    reg = _registry(_decl("SRC-A", threshold_ms=0))
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    [r] = mgr.reports(now_ns=10**12)  # 1000 s later
    assert r.status == SourceStatus.LIVE


# ---------------------------------------------------------------------------
# SourceManager — observe() emits transitions
# ---------------------------------------------------------------------------


def test_observe_emits_heartbeat_on_first_seen():
    reg = _registry(_decl("SRC-A", threshold_ms=1_000))
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    sys_events, hazards = mgr.observe(now_ns=10**9 + 100)
    assert len(sys_events) == 1
    assert hazards == ()
    e = sys_events[0]
    assert e.kind == EventKind.SYSTEM
    assert e.sub_kind == SystemEventKind.SOURCE_HEARTBEAT
    assert e.payload["source_id"] == "SRC-A"
    assert e.payload["from"] == SourceStatus.UNKNOWN.value
    assert e.payload["to"] == SourceStatus.LIVE.value


def test_observe_emits_stale_on_threshold_exceeded():
    reg = _registry(_decl("SRC-A", threshold_ms=1_000))
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    mgr.observe(now_ns=10**9 + 100)  # consume UNKNOWN→LIVE
    sys_events, hazards = mgr.observe(now_ns=10**9 + 2_000_000_000)
    assert len(sys_events) == 1
    assert sys_events[0].sub_kind == SystemEventKind.SOURCE_STALE
    assert hazards == ()  # not critical


def test_observe_emits_recovered_after_new_heartbeat():
    reg = _registry(_decl("SRC-A", threshold_ms=1_000))
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    mgr.observe(now_ns=10**9 + 100)
    mgr.observe(now_ns=10**9 + 2_000_000_000)  # → STALE
    mgr.record_heartbeat("SRC-A", ts_ns=10**9 + 2_500_000_000)
    sys_events, _ = mgr.observe(now_ns=10**9 + 2_500_000_001)
    assert len(sys_events) == 1
    assert sys_events[0].sub_kind == SystemEventKind.SOURCE_RECOVERED


def test_observe_is_idempotent_at_same_now():
    reg = _registry(_decl("SRC-A", threshold_ms=1_000))
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    first, _ = mgr.observe(now_ns=10**9 + 100)
    second, _ = mgr.observe(now_ns=10**9 + 100)
    assert len(first) == 1
    assert second == ()


# ---------------------------------------------------------------------------
# SCVS-06 — critical-source fail-closed
# ---------------------------------------------------------------------------


def test_critical_source_stale_emits_hazard():
    reg = _registry(_decl("SRC-A", threshold_ms=1_000, critical=True))
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    mgr.observe(now_ns=10**9 + 100)  # → LIVE
    sys_events, hazards = mgr.observe(now_ns=10**9 + 2_000_000_000)
    assert len(sys_events) == 1
    assert sys_events[0].sub_kind == SystemEventKind.SOURCE_STALE
    assert len(hazards) == 1
    h = hazards[0]
    assert h.code == HAZ_CRITICAL_SOURCE_STALE
    assert h.severity == HazardSeverity.HIGH
    assert h.meta["source_id"] == "SRC-A"


def test_non_critical_stale_emits_no_hazard():
    reg = _registry(_decl("SRC-A", threshold_ms=1_000, critical=False))
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    mgr.observe(now_ns=10**9 + 100)
    _, hazards = mgr.observe(now_ns=10**9 + 2_000_000_000)
    assert hazards == ()


def test_critical_recovery_emits_no_hazard():
    """SCVS-06 only escalates on the failing edge."""

    reg = _registry(_decl("SRC-A", threshold_ms=1_000, critical=True))
    mgr = SourceManager(registry=reg)
    mgr.record_heartbeat("SRC-A", ts_ns=10**9)
    mgr.observe(now_ns=10**9 + 100)
    mgr.observe(now_ns=10**9 + 2_000_000_000)  # critical STALE -> hazard
    mgr.record_heartbeat("SRC-A", ts_ns=10**9 + 2_500_000_000)
    sys_events, hazards = mgr.observe(now_ns=10**9 + 2_500_000_001)
    assert len(sys_events) == 1
    assert sys_events[0].sub_kind == SystemEventKind.SOURCE_RECOVERED
    assert hazards == ()


# ---------------------------------------------------------------------------
# Determinism — INV-15
# ---------------------------------------------------------------------------


def test_replay_produces_identical_event_sequence():
    """Same input timeline → identical SystemEvent + HazardEvent tuples."""

    def run() -> tuple[tuple, tuple]:
        reg = _registry(_decl("SRC-A", threshold_ms=1_000, critical=True))
        mgr = SourceManager(registry=reg)
        sys_all: list = []
        haz_all: list = []
        mgr.record_heartbeat("SRC-A", ts_ns=10**9)
        for ts in (
            10**9 + 100,
            10**9 + 2_000_000_000,  # → STALE + hazard
            10**9 + 3_000_000_000,
        ):
            s, h = mgr.observe(now_ns=ts)
            sys_all.extend(s)
            haz_all.extend(h)
        mgr.record_heartbeat("SRC-A", ts_ns=10**9 + 4_000_000_000)
        s, h = mgr.observe(now_ns=10**9 + 4_000_000_001)
        sys_all.extend(s)
        haz_all.extend(h)
        return tuple(sys_all), tuple(haz_all)

    assert run() == run()
