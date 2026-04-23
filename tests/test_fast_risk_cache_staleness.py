"""Tests for system.fast_risk_cache — T0-1 staleness + monotonic version."""
from __future__ import annotations

import pytest

from system.fast_risk_cache import (
    DEFAULT_STALENESS_THRESHOLD_NS,
    FastRiskCache,
    RiskConstraints,
)


def _make_cache(
    threshold_ns: int = DEFAULT_STALENESS_THRESHOLD_NS,
    start_ns: int = 1_000_000_000,
) -> tuple[FastRiskCache, dict]:
    clock = {"v": start_ns}
    cache = FastRiskCache(
        staleness_threshold_ns=threshold_ns,
        clock_wall_ns=lambda: clock["v"],
    )
    return cache, clock


# ─────── version monotonicity ───────────────────────────────────────────


def test_version_starts_at_one() -> None:
    cache, _ = _make_cache()
    assert cache.version == 1


def test_version_bumps_monotonically_on_update() -> None:
    cache, _ = _make_cache()
    v1 = cache.version
    cache.update(max_order_size_usd=5_000.0)
    v2 = cache.version
    cache.update(trading_allowed=False)
    v3 = cache.version
    assert v1 < v2 < v3
    assert v3 == 3


def test_update_stamps_updated_at_ns() -> None:
    cache, clock = _make_cache(start_ns=100)
    clock["v"] = 200
    cache.update(max_order_size_usd=999.0)
    assert cache.updated_at_ns == 200


# ─────── staleness ──────────────────────────────────────────────────────


def test_is_fresh_within_threshold() -> None:
    cache, clock = _make_cache(threshold_ns=1000, start_ns=100)
    clock["v"] = 200  # 100 ns after creation
    assert cache.is_fresh() is True


def test_is_fresh_returns_false_when_stale() -> None:
    cache, clock = _make_cache(threshold_ns=1000, start_ns=100)
    clock["v"] = 2000  # 1900 ns after creation, threshold is 1000
    assert cache.is_fresh() is False


def test_staleness_ns_returns_age() -> None:
    cache, clock = _make_cache(threshold_ns=1000, start_ns=100)
    clock["v"] = 600
    assert cache.staleness_ns() == 500


def test_is_fresh_with_explicit_now_ns() -> None:
    cache, _ = _make_cache(threshold_ns=1000, start_ns=100)
    assert cache.is_fresh(now_ns=500) is True
    assert cache.is_fresh(now_ns=5000) is False


# ─────── allows_trade with staleness ────────────────────────────────────


def test_allows_trade_rejects_when_stale() -> None:
    """Core T0-1 invariant: stale cache must fail-closed."""
    cache, clock = _make_cache(threshold_ns=1000, start_ns=100)
    clock["v"] = 5000
    rc = cache.get()
    ok, reason = rc.allows_trade(
        10.0, 100_000.0,
        now_ns=clock["v"],
        staleness_threshold_ns=cache.staleness_threshold_ns,
    )
    assert ok is False
    assert reason == "risk_cache_stale"


def test_allows_trade_passes_when_fresh() -> None:
    cache, clock = _make_cache(threshold_ns=10**18, start_ns=100)
    clock["v"] = 200
    rc = cache.get()
    ok, reason = rc.allows_trade(
        10.0, 100_000.0,
        now_ns=clock["v"],
        staleness_threshold_ns=cache.staleness_threshold_ns,
    )
    assert ok is True
    assert reason == "ok"


def test_allows_trade_backward_compat_without_now_ns() -> None:
    """Existing callers that omit now_ns must not break — staleness
    check is skipped and the old behavior is preserved."""
    rc = RiskConstraints(trading_allowed=True)
    ok, reason = rc.allows_trade(10.0, 100_000.0)
    assert ok is True
    assert reason == "ok"


# ─────── safe mode / halt ───────────────────────────────────────────────


def test_enter_safe_mode_bumps_version() -> None:
    cache, _ = _make_cache()
    v_before = cache.version
    cache.enter_safe_mode()
    assert cache.version == v_before + 1
    rc = cache.get()
    assert rc.safe_mode is True
    assert rc.trading_allowed is False


def test_halt_and_resume_bump_versions() -> None:
    cache, _ = _make_cache()
    cache.halt_trading("test")
    v_halt = cache.version
    cache.resume_trading()
    v_resume = cache.version
    assert v_halt < v_resume


# ─────── edge cases ─────────────────────────────────────────────────────


def test_stale_overrides_trading_allowed() -> None:
    """Even if trading_allowed is True, staleness must prevail."""
    rc = RiskConstraints(trading_allowed=True, updated_at_ns=100)
    ok, reason = rc.allows_trade(
        10.0, 100_000.0, now_ns=99_999, staleness_threshold_ns=1000
    )
    assert ok is False
    assert reason == "risk_cache_stale"


def test_version_is_preserved_in_constraints_snapshot() -> None:
    cache, _ = _make_cache()
    cache.update(max_order_size_usd=42.0)
    rc = cache.get()
    assert rc.version == 2
    assert rc.max_order_size_usd == 42.0
