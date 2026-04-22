"""
tests/test_phase0_attestation.py
DIX VISION v42.2 — Phase 0 regression guards.

These tests are the *runtime enforcement* of the Phase 0 Build Plan
deliverables. If any of them fails, Phase 0 is broken and must be
repaired before Phase 1 work may proceed.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── Build Plan §1.1 — immutable_core hash integrity ───────────────────
def test_foundation_hash_file_matches_foundation_py() -> None:
    foundation_py = (ROOT / "immutable_core" / "foundation.py").read_bytes()
    expected = hashlib.sha256(foundation_py).hexdigest()
    recorded = (ROOT / "immutable_core" / "foundation.hash").read_text().strip()
    assert recorded == expected, (
        f"foundation.hash stale: file has {recorded}, actual SHA256 is {expected}. "
        "Run `python scripts/generate_hash.py` to regenerate."
    )


def test_genesis_json_foundation_hash_matches_hash_file() -> None:
    recorded = (ROOT / "immutable_core" / "foundation.hash").read_text().strip()
    genesis = json.loads((ROOT / "immutable_core" / "genesis.json").read_text())
    gh = genesis.get("foundation_hash", "")
    assert gh == recorded, (
        f"genesis.json foundation_hash={gh!r} does not match "
        f"immutable_core/foundation.hash={recorded!r}"
    )
    assert len(gh) == 64 and all(c in "0123456789abcdef" for c in gh), (
        "genesis.json foundation_hash must be a 64-char lowercase hex SHA256"
    )


def test_safety_axioms_lean_has_content() -> None:
    text = (ROOT / "immutable_core" / "safety_axioms.lean").read_text()
    # S1..S10 must all be mentioned — the file is the spec.
    for axiom in ("S1_max_drawdown_floor", "S2_per_trade_loss_floor",
                  "S3_fail_closed", "S4_credentials_local_only",
                  "S5_fast_path_latency_budget", "S6_frozen_hot_path",
                  "S7_ledger_append_only", "S8_foundation_hash_pinned",
                  "S9_kill_switch_uses_only_stdlib",
                  "S10_kill_switch_is_idempotent"):
        assert axiom in text, f"safety_axioms.lean missing axiom: {axiom}"


def test_hazard_axioms_lean_exists_with_h1_h10() -> None:
    path = ROOT / "immutable_core" / "hazard_axioms.lean"
    assert path.exists(), "immutable_core/hazard_axioms.lean is required (Phase 0 §1.1)"
    text = path.read_text()
    for axiom in ("H1_single_channel", "H2_dyon_sole_producer",
                  "H3_governance_sole_consumer", "H4_non_blocking_producer",
                  "H5_queue_overflow_fails_closed", "H6_critical_halts_trading",
                  "H7_high_enters_safe_mode", "H8_medium_observe",
                  "H9_every_hazard_logged",
                  "H10_override_requires_two_person_gate"):
        assert axiom in text, f"hazard_axioms.lean missing axiom: {axiom}"


# ── Build Plan §1.2 — contracts exist ─────────────────────────────────
def test_contracts_module_exports_risk_protocols() -> None:
    contracts = importlib.import_module("core.contracts")
    for name in ("IRiskCache", "IRiskConstraints", "ISystemHazardEvent",
                 "IHazardEmitter", "IGovernanceHazardSink"):
        assert hasattr(contracts, name), f"core.contracts missing {name}"
        assert name in contracts.__all__, f"{name} not in core.contracts.__all__"


def test_hazard_event_satisfies_system_hazard_contract() -> None:
    from core.contracts import ISystemHazardEvent
    from execution.hazard.async_bus import HazardEvent, HazardSeverity, HazardType

    e = HazardEvent(
        hazard_type=HazardType.FEED_SILENCE,
        severity=HazardSeverity.HIGH,
        source="dyon.test",
        details={"x": 1},
    )
    assert isinstance(e, ISystemHazardEvent), (
        "HazardEvent must satisfy ISystemHazardEvent (runtime_checkable)"
    )


def test_risk_cache_satisfies_risk_contract() -> None:
    from core.contracts import IRiskCache, IRiskConstraints
    from system.fast_risk_cache import get_risk_cache

    cache = get_risk_cache()
    assert isinstance(cache, IRiskCache), (
        "fast_risk_cache singleton must satisfy IRiskCache"
    )
    snapshot = cache.get()
    assert isinstance(snapshot, IRiskConstraints), (
        "FastRiskCache.get() must return an IRiskConstraints snapshot"
    )


# ── Build Plan §1.3 — registry lock ───────────────────────────────────
def test_registry_lock_prevents_post_boot_registration() -> None:
    from core.registry import Registry

    r = Registry()
    r.register("x", lambda: 1)
    r.lock()
    with pytest.raises(RuntimeError, match="locked"):
        r.register("y", lambda: 2)


def test_bootstrap_kernel_calls_registry_lock() -> None:
    """Cheap static guard — ensures the registry.lock() wiring is not
    silently removed from bootstrap_kernel.py. A full boot test would be
    expensive and flaky; this grep-style test is deterministic."""
    src = (ROOT / "bootstrap_kernel.py").read_text()
    assert "get_registry().lock()" in src, (
        "bootstrap_kernel.py must call get_registry().lock() after boot "
        "(Phase 0 Build Plan §1.3)"
    )
