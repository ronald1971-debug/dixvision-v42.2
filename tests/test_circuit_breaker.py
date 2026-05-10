"""Tests for A-20.2 circuit breaker.

Coverage:

* ``StoplossEvent`` validation (negative ts, empty pair, wrong types)
* ``CircuitBreakerPolicy`` validation + ``canonical_text`` /
  ``policy_digest`` determinism + sensitivity
* ``CircuitBreaker.record`` monotonic ts enforcement + FIFO eviction
* ``CircuitBreaker.evaluate`` happy path (ARMED), trip on threshold,
  cooldown honoured, post-cooldown re-arm
* ``only_per_pair`` / ``only_per_side`` filters
* ``profit_limit`` filter (positive-profit exits are *not* counted)
* INV-15 byte-identical replay equality (3 runs)
* ``meta`` keys are sorted in the returned verdict
* AST guards: no engine cross-imports; no typed bus event
  construction; no clock / random / asyncio / os imports;
  ``# ADAPTED FROM`` header present
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from execution_engine.protections.circuit_breaker import (
    DEFAULT_LOOKBACK_NS,
    DEFAULT_STOP_DURATION_NS,
    DEFAULT_TRADE_LIMIT,
    MAX_EVENT_BUFFER,
    NEW_PIP_DEPENDENCIES,
    CircuitBreaker,
    CircuitBreakerPolicy,
    CircuitBreakerState,
    CircuitBreakerVerdict,
    Side,
    StoplossEvent,
    StoplossExitReason,
)

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "execution_engine"
    / "protections"
    / "circuit_breaker.py"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev(
    ts_ns: int,
    *,
    pair: str = "BTC/USDT",
    side: Side = Side.LONG,
    close_profit: float = -0.01,
    exit_reason: StoplossExitReason = StoplossExitReason.STOP_LOSS,
) -> StoplossEvent:
    return StoplossEvent(
        ts_ns=ts_ns,
        pair=pair,
        side=side,
        close_profit=close_profit,
        exit_reason=exit_reason,
    )


# ---------------------------------------------------------------------------
# StoplossEvent validation
# ---------------------------------------------------------------------------


def test_event_rejects_negative_ts() -> None:
    with pytest.raises(ValueError):
        _ev(-1)


def test_event_rejects_empty_pair() -> None:
    with pytest.raises(ValueError):
        _ev(0, pair="")


def test_event_rejects_wrong_side_type() -> None:
    with pytest.raises(TypeError):
        StoplossEvent(
            ts_ns=0,
            pair="BTC/USDT",
            side="LONG",  # type: ignore[arg-type]
            close_profit=-0.01,
            exit_reason=StoplossExitReason.STOP_LOSS,
        )


def test_event_rejects_wrong_exit_reason_type() -> None:
    with pytest.raises(TypeError):
        StoplossEvent(
            ts_ns=0,
            pair="BTC/USDT",
            side=Side.LONG,
            close_profit=-0.01,
            exit_reason="STOP_LOSS",  # type: ignore[arg-type]
        )


def test_event_is_frozen() -> None:
    ev = _ev(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.ts_ns = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CircuitBreakerPolicy validation + digest
# ---------------------------------------------------------------------------


def test_policy_defaults() -> None:
    p = CircuitBreakerPolicy()
    assert p.trade_limit == DEFAULT_TRADE_LIMIT
    assert p.lookback_ns == DEFAULT_LOOKBACK_NS
    assert p.stop_duration_ns == DEFAULT_STOP_DURATION_NS
    assert p.profit_limit == 0.0
    assert p.only_per_pair is False
    assert p.only_per_side is False


@pytest.mark.parametrize("trade_limit", [0, -1, -100])
def test_policy_rejects_bad_trade_limit(trade_limit: int) -> None:
    with pytest.raises(ValueError):
        CircuitBreakerPolicy(trade_limit=trade_limit)


@pytest.mark.parametrize("lookback_ns", [0, -1])
def test_policy_rejects_bad_lookback(lookback_ns: int) -> None:
    with pytest.raises(ValueError):
        CircuitBreakerPolicy(lookback_ns=lookback_ns)


def test_policy_rejects_negative_stop_duration() -> None:
    with pytest.raises(ValueError):
        CircuitBreakerPolicy(stop_duration_ns=-1)


def test_policy_canonical_text_deterministic() -> None:
    p1 = CircuitBreakerPolicy(trade_limit=3, lookback_ns=1_000_000_000)
    p2 = CircuitBreakerPolicy(trade_limit=3, lookback_ns=1_000_000_000)
    assert p1.canonical_text() == p2.canonical_text()
    assert p1.policy_digest() == p2.policy_digest()


def test_policy_digest_sensitive_to_every_field() -> None:
    base = CircuitBreakerPolicy()
    variants = [
        CircuitBreakerPolicy(trade_limit=base.trade_limit + 1),
        CircuitBreakerPolicy(lookback_ns=base.lookback_ns + 1),
        CircuitBreakerPolicy(profit_limit=base.profit_limit + 0.01),
        CircuitBreakerPolicy(stop_duration_ns=base.stop_duration_ns + 1),
        CircuitBreakerPolicy(only_per_pair=True),
        CircuitBreakerPolicy(only_per_side=True),
    ]
    base_digest = base.policy_digest()
    digests = {v.policy_digest() for v in variants}
    assert base_digest not in digests
    assert len(digests) == len(variants)


def test_policy_digest_is_16_bytes_hex() -> None:
    d = CircuitBreakerPolicy().policy_digest()
    assert len(d) == 32
    int(d, 16)  # parses as hex


# ---------------------------------------------------------------------------
# CircuitBreaker.record monotonicity + FIFO
# ---------------------------------------------------------------------------


def test_record_rejects_non_monotonic_ts() -> None:
    cb = CircuitBreaker()
    cb.record(_ev(10))
    with pytest.raises(ValueError):
        cb.record(_ev(5))


def test_record_allows_equal_ts() -> None:
    cb = CircuitBreaker()
    cb.record(_ev(10))
    cb.record(_ev(10, pair="ETH/USDT"))
    assert cb.event_count == 2


def test_record_fifo_eviction() -> None:
    cb = CircuitBreaker(max_buffer=3)
    for i in range(5):
        cb.record(_ev(i))
    assert cb.event_count == 3
    timestamps = [ev.ts_ns for ev in cb.events()]
    assert timestamps == [2, 3, 4]


def test_record_batch() -> None:
    cb = CircuitBreaker()
    cb.record_batch([_ev(0), _ev(1), _ev(2)])
    assert cb.event_count == 3


def test_constructor_rejects_bad_max_buffer() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(max_buffer=0)


# ---------------------------------------------------------------------------
# CircuitBreaker.evaluate happy paths
# ---------------------------------------------------------------------------


def test_evaluate_armed_when_under_threshold() -> None:
    cb = CircuitBreaker(policy=CircuitBreakerPolicy(trade_limit=4, lookback_ns=1_000))
    cb.record(_ev(100))
    cb.record(_ev(200))
    verdict = cb.evaluate(now_ns=300)
    assert verdict.state is CircuitBreakerState.ARMED
    assert verdict.is_locked is False
    assert verdict.qualifying_count == 2
    assert verdict.locked_until_ns == 0


def test_evaluate_trips_at_threshold() -> None:
    policy = CircuitBreakerPolicy(
        trade_limit=3,
        lookback_ns=1_000,
        stop_duration_ns=5_000,
    )
    cb = CircuitBreaker(policy=policy)
    cb.record(_ev(100))
    cb.record(_ev(200))
    cb.record(_ev(300))
    verdict = cb.evaluate(now_ns=400)
    assert verdict.state is CircuitBreakerState.TRIPPED
    assert verdict.is_locked is True
    assert verdict.qualifying_count == 3
    assert verdict.locked_until_ns == 400 + 5_000
    assert "tripped" in verdict.reason


def test_evaluate_cooldown_active() -> None:
    policy = CircuitBreakerPolicy(
        trade_limit=2,
        lookback_ns=1_000,
        stop_duration_ns=10_000,
    )
    cb = CircuitBreaker(policy=policy)
    cb.record(_ev(100))
    cb.record(_ev(200))
    trip = cb.evaluate(now_ns=300)
    assert trip.is_locked

    # Mid-cooldown: still locked even though zero new events arrive
    mid = cb.evaluate(now_ns=5_000)
    assert mid.is_locked
    assert mid.locked_until_ns == trip.locked_until_ns
    assert "cooldown active" in mid.reason


def test_evaluate_re_arms_after_cooldown_and_window_expiry() -> None:
    policy = CircuitBreakerPolicy(
        trade_limit=2,
        lookback_ns=1_000,
        stop_duration_ns=500,
    )
    cb = CircuitBreaker(policy=policy)
    cb.record(_ev(100))
    cb.record(_ev(200))
    trip = cb.evaluate(now_ns=300)
    assert trip.is_locked
    # After cooldown AND lookback window has rolled past every event:
    after = cb.evaluate(now_ns=300 + 500 + 1_001)
    assert after.is_locked is False
    assert after.state is CircuitBreakerState.ARMED


def test_evaluate_rejects_negative_now_ns() -> None:
    cb = CircuitBreaker()
    with pytest.raises(ValueError):
        cb.evaluate(now_ns=-1)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_profit_limit_excludes_winning_stoplosses() -> None:
    policy = CircuitBreakerPolicy(
        trade_limit=2,
        lookback_ns=1_000,
        profit_limit=0.0,
    )
    cb = CircuitBreaker(policy=policy)
    cb.record(_ev(100, close_profit=-0.01))
    cb.record(_ev(200, close_profit=0.05))  # winning trailing stop — excluded
    verdict = cb.evaluate(now_ns=300)
    assert verdict.qualifying_count == 1
    assert verdict.is_locked is False


def test_only_per_pair_filter() -> None:
    policy = CircuitBreakerPolicy(
        trade_limit=2,
        lookback_ns=1_000,
        only_per_pair=True,
    )
    cb = CircuitBreaker(policy=policy)
    cb.record(_ev(100, pair="BTC/USDT"))
    cb.record(_ev(200, pair="ETH/USDT"))
    cb.record(_ev(300, pair="BTC/USDT"))
    # Filtering on BTC/USDT — 2 events, trips.
    v_btc = cb.evaluate(now_ns=400, pair="BTC/USDT")
    assert v_btc.is_locked is True
    assert v_btc.qualifying_count == 2
    assert v_btc.lock_pair == "BTC/USDT"


def test_only_per_pair_isolates_other_pair() -> None:
    policy = CircuitBreakerPolicy(
        trade_limit=2,
        lookback_ns=1_000,
        only_per_pair=True,
        stop_duration_ns=10_000,
    )
    cb = CircuitBreaker(policy=policy)
    cb.record(_ev(100, pair="BTC/USDT"))
    cb.record(_ev(200, pair="BTC/USDT"))
    # ETH/USDT counter is still empty, but breaker's internal lock got
    # set by the BTC check. evaluate on ETH right after — locked
    # because cooldown is global once tripped.
    cb.evaluate(now_ns=300, pair="BTC/USDT")
    v_eth = cb.evaluate(now_ns=400, pair="ETH/USDT")
    # Within the global cooldown, every pair is locked.
    assert v_eth.is_locked is True


def test_only_per_side_filter() -> None:
    policy = CircuitBreakerPolicy(
        trade_limit=2,
        lookback_ns=1_000,
        only_per_side=True,
    )
    cb = CircuitBreaker(policy=policy)
    cb.record(_ev(100, side=Side.LONG))
    cb.record(_ev(200, side=Side.SHORT))
    cb.record(_ev(300, side=Side.LONG))
    v_long = cb.evaluate(now_ns=400, side=Side.LONG)
    assert v_long.qualifying_count == 2
    assert v_long.is_locked is True
    assert v_long.lock_side == "LONG"


def test_filters_default_to_global() -> None:
    policy = CircuitBreakerPolicy(trade_limit=2, lookback_ns=1_000)
    cb = CircuitBreaker(policy=policy)
    cb.record(_ev(100, pair="BTC/USDT", side=Side.LONG))
    cb.record(_ev(200, pair="ETH/USDT", side=Side.SHORT))
    verdict = cb.evaluate(now_ns=300, pair="BTC/USDT", side=Side.LONG)
    # only_per_pair / only_per_side are False → global count
    assert verdict.qualifying_count == 2
    assert verdict.lock_pair == "*"
    assert verdict.lock_side == "*"


def test_lookback_window_excludes_old_events() -> None:
    policy = CircuitBreakerPolicy(trade_limit=2, lookback_ns=1_000)
    cb = CircuitBreaker(policy=policy)
    cb.record(_ev(0))  # outside window
    cb.record(_ev(50))  # outside window
    cb.record(_ev(200))  # inside
    verdict = cb.evaluate(now_ns=1_200)
    # window = (1_200 - 1_000, 1_200] = (200, 1_200]; only ts >= 200 counts.
    assert verdict.qualifying_count == 1


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def _run_scenario() -> tuple[CircuitBreakerVerdict, ...]:
    policy = CircuitBreakerPolicy(
        trade_limit=3,
        lookback_ns=1_000,
        stop_duration_ns=2_000,
        profit_limit=0.0,
        only_per_pair=True,
        only_per_side=True,
    )
    cb = CircuitBreaker(policy=policy)
    verdicts: list[CircuitBreakerVerdict] = []
    cb.record(_ev(100, pair="BTC/USDT", side=Side.LONG, close_profit=-0.02))
    cb.record(_ev(150, pair="ETH/USDT", side=Side.LONG, close_profit=-0.01))
    verdicts.append(cb.evaluate(now_ns=200, pair="BTC/USDT", side=Side.LONG))
    cb.record(_ev(300, pair="BTC/USDT", side=Side.LONG, close_profit=-0.03))
    cb.record(_ev(400, pair="BTC/USDT", side=Side.LONG, close_profit=-0.01))
    verdicts.append(cb.evaluate(now_ns=500, pair="BTC/USDT", side=Side.LONG))
    verdicts.append(cb.evaluate(now_ns=1_500, pair="BTC/USDT", side=Side.LONG))
    verdicts.append(cb.evaluate(now_ns=10_000, pair="BTC/USDT", side=Side.LONG))
    return tuple(verdicts)


def test_three_run_byte_identical_replay() -> None:
    runs = [_run_scenario() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_replay_locks_at_expected_step() -> None:
    verdicts = _run_scenario()
    # Step 0: 1 qualifying for the (BTC/USDT, LONG) bucket → ARMED.
    assert verdicts[0].is_locked is False
    # Step 1: 3 qualifying → TRIPPED.
    assert verdicts[1].is_locked is True
    assert verdicts[1].qualifying_count == 3
    # Step 2: cooldown still active.
    assert verdicts[2].is_locked is True
    # Step 3: cooldown elapsed + events outside lookback → ARMED.
    assert verdicts[3].is_locked is False


# ---------------------------------------------------------------------------
# Verdict meta-key sorting
# ---------------------------------------------------------------------------


def test_verdict_meta_keys_are_sorted() -> None:
    cb = CircuitBreaker()
    cb.record(_ev(0))
    verdict = cb.evaluate(
        now_ns=1,
        meta={"zzz": "z", "aaa": "a", "mmm": "m"},
    )
    assert list(verdict.meta.keys()) == ["aaa", "mmm", "zzz"]


def test_verdict_meta_default_is_empty() -> None:
    cb = CircuitBreaker()
    verdict = cb.evaluate(now_ns=1)
    assert verdict.meta == {}


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


def _module_tree() -> ast.AST:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _imported_modules(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def _call_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
    return out


def test_no_forbidden_top_level_imports() -> None:
    mods = _imported_modules(_module_tree())
    for forbidden in (
        "random",
        "asyncio",
        "os",
        "datetime",
        "time",
        "numpy",
        "torch",
        "polars",
        "pandas",
        "freqtrade",
    ):
        assert forbidden not in mods, f"forbidden import: {forbidden}"


def test_no_engine_cross_imports() -> None:
    mods = _imported_modules(_module_tree())
    for forbidden in (
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "evolution_engine",
        "learning_engine",
    ):
        for m in mods:
            assert not m.startswith(forbidden), f"forbidden engine import: {m}"


def test_no_typed_bus_event_construction() -> None:
    calls = _call_names(_module_tree())
    for forbidden in (
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "GovernanceDecision",
        "LearningUpdate",
        "PatchProposal",
        "TraderObservation",
    ):
        assert forbidden not in calls, f"forbidden constructor call: {forbidden}"


def test_adapted_from_header_present() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: freqtrade/plugins/protections/stoploss_guard.py" in src
    assert "# ADAPTED FROM: freqtrade/plugins/protections/cooldown_period.py" in src


def test_new_pip_dependencies_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_max_event_buffer_constant_consistent() -> None:
    cb = CircuitBreaker()
    # default constructor uses MAX_EVENT_BUFFER
    assert cb._events.maxlen == MAX_EVENT_BUFFER  # noqa: SLF001
