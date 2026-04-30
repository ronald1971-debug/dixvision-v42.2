"""Tests for the continuous drift oracle (Reviewer #4 finding 3 / #5 AUTO).

Covers:

* :class:`DriftSample` clamping at construction time.
* :class:`DriftOracle` constructor input validation (threshold, samples,
  weights).
* ``observe`` writes exactly one ``DRIFT_OBSERVATION`` ledger row with
  the four axes plus the running composite.
* ``composite`` is a deterministic, weighted mean.
* ``check`` returns ``(True, "")`` for non-AUTO targets unconditionally
  even when the buffer is empty.
* ``check`` returns ``(False, "DRIFT_ORACLE_INSUFFICIENT_SAMPLES")`` for
  AUTO when the buffer has fewer than ``min_samples`` samples.
* ``check`` returns ``(False, "DRIFT_ORACLE_THRESHOLD_BREACH")`` for
  AUTO when composite is at or above ``threshold``.
* ``replay_from_ledger`` reconstructs the ring buffer from
  ``DRIFT_OBSERVATION`` rows.
* :class:`StateTransitionManager` integration: AUTO entry refused on
  insufficient samples / threshold breach; allowed when oracle is
  steady-state below threshold; backwards-compatible without an
  oracle; non-AUTO transitions never touched.
"""

from __future__ import annotations

import pytest

from core.contracts.governance import ModeTransitionRequest, SystemMode
from governance_engine.control_plane import (
    DEFAULT_COMPOSITE_THRESHOLD,
    DEFAULT_MIN_SAMPLES,
    LEDGER_KIND_DRIFT_OBSERVATION,
    DriftOracle,
    DriftSample,
    LedgerAuthorityWriter,
    PolicyEngine,
    StateTransitionManager,
)

# ---------------------------------------------------------------------------
# DriftSample
# ---------------------------------------------------------------------------


def test_drift_sample_from_raw_clamps_axes() -> None:
    s = DriftSample.from_raw(
        ts_ns=10,
        model_drift=-0.5,
        execution_drift=2.0,
        latency_drift=0.3,
        causal_drift=float("nan"),
    )
    assert s.ts_ns == 10
    assert s.model_drift == 0.0
    assert s.execution_drift == 1.0
    assert s.latency_drift == 0.3
    # NaN clamps to 1.0 (worst case) so a misbehaving producer cannot
    # silently lower the composite by emitting NaN.
    assert s.causal_drift == 1.0


def test_drift_sample_is_frozen() -> None:
    s = DriftSample(
        ts_ns=1,
        model_drift=0.1,
        execution_drift=0.1,
        latency_drift=0.1,
        causal_drift=0.1,
    )
    with pytest.raises((AttributeError, TypeError)):
        s.model_drift = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DriftOracle constructor validation
# ---------------------------------------------------------------------------


def test_oracle_rejects_non_unit_threshold() -> None:
    ledger = LedgerAuthorityWriter()
    with pytest.raises(ValueError):
        DriftOracle(ledger=ledger, threshold=0.0)
    with pytest.raises(ValueError):
        DriftOracle(ledger=ledger, threshold=1.5)


def test_oracle_rejects_zero_min_samples() -> None:
    ledger = LedgerAuthorityWriter()
    with pytest.raises(ValueError):
        DriftOracle(ledger=ledger, min_samples=0)


def test_oracle_rejects_window_smaller_than_min_samples() -> None:
    ledger = LedgerAuthorityWriter()
    with pytest.raises(ValueError):
        DriftOracle(ledger=ledger, min_samples=10, window_size=5)


def test_oracle_rejects_unbalanced_weights() -> None:
    ledger = LedgerAuthorityWriter()
    with pytest.raises(ValueError):
        DriftOracle(
            ledger=ledger,
            weights={
                "model_drift": 0.1,
                "execution_drift": 0.1,
                "latency_drift": 0.1,
                "causal_drift": 0.1,
            },
        )


def test_oracle_rejects_missing_weight_key() -> None:
    ledger = LedgerAuthorityWriter()
    with pytest.raises(ValueError):
        DriftOracle(
            ledger=ledger,
            weights={  # type: ignore[arg-type]
                "model_drift": 1.0,
            },
        )


def test_oracle_rejects_negative_weight() -> None:
    ledger = LedgerAuthorityWriter()
    with pytest.raises(ValueError):
        DriftOracle(
            ledger=ledger,
            weights={
                "model_drift": -0.5,
                "execution_drift": 0.5,
                "latency_drift": 0.5,
                "causal_drift": 0.5,
            },
        )


# ---------------------------------------------------------------------------
# observe / composite
# ---------------------------------------------------------------------------


def _quiet_sample(ts_ns: int, level: float = 0.05) -> DriftSample:
    return DriftSample.from_raw(
        ts_ns=ts_ns,
        model_drift=level,
        execution_drift=level,
        latency_drift=level,
        causal_drift=level,
    )


def test_observe_writes_one_ledger_row_with_composite() -> None:
    ledger = LedgerAuthorityWriter()
    oracle = DriftOracle(ledger=ledger, min_samples=1, window_size=8)

    oracle.observe(_quiet_sample(ts_ns=100, level=0.1))

    rows = ledger.read()
    assert len(rows) == 1
    row = rows[0]
    assert row.kind == LEDGER_KIND_DRIFT_OBSERVATION
    assert row.payload["sample_count"] == "1"
    # Composite of all four axes at 0.1, weights sum to 1.0 -> 0.1.
    assert pytest.approx(float(row.payload["composite"]), abs=1e-9) == 0.1
    assert pytest.approx(float(row.payload["model_drift"]), abs=1e-9) == 0.1


def test_composite_is_zero_for_empty_buffer() -> None:
    ledger = LedgerAuthorityWriter()
    oracle = DriftOracle(ledger=ledger, min_samples=1, window_size=8)
    assert oracle.composite() == 0.0


def test_composite_is_weighted_mean_over_window() -> None:
    ledger = LedgerAuthorityWriter()
    oracle = DriftOracle(ledger=ledger, min_samples=1, window_size=4)

    # Two quiet samples then two loud samples; mean composite should be
    # (0.05 + 0.05 + 0.5 + 0.5) / 4 = 0.275 (weights sum to 1.0).
    for ts in (1, 2):
        oracle.observe(_quiet_sample(ts_ns=ts, level=0.05))
    for ts in (3, 4):
        oracle.observe(_quiet_sample(ts_ns=ts, level=0.5))

    assert pytest.approx(oracle.composite(), abs=1e-9) == 0.275


def test_buffer_drops_oldest_when_full() -> None:
    ledger = LedgerAuthorityWriter()
    oracle = DriftOracle(ledger=ledger, min_samples=1, window_size=2)

    oracle.observe(_quiet_sample(ts_ns=1, level=1.0))
    oracle.observe(_quiet_sample(ts_ns=2, level=0.0))
    oracle.observe(_quiet_sample(ts_ns=3, level=0.0))

    # Oldest dropped: composite should be (0.0 + 0.0)/2 = 0.0.
    assert oracle.composite() == 0.0
    assert oracle.sample_count() == 2


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["SAFE", "PAPER", "SHADOW", "CANARY", "LIVE", "LOCKED"])
def test_check_passes_for_non_auto_targets_unconditionally(target: str) -> None:
    ledger = LedgerAuthorityWriter()
    # Empty oracle -- still passes for non-AUTO targets.
    oracle = DriftOracle(ledger=ledger, min_samples=4, window_size=4)
    ok, code = oracle.check(target)
    assert ok is True
    assert code == ""


def test_check_rejects_auto_when_insufficient_samples() -> None:
    ledger = LedgerAuthorityWriter()
    oracle = DriftOracle(ledger=ledger, min_samples=4, window_size=8)
    for ts in range(1, 4):  # only 3 samples (need 4)
        oracle.observe(_quiet_sample(ts_ns=ts, level=0.05))

    ok, code = oracle.check("AUTO")
    assert ok is False
    assert code == "DRIFT_ORACLE_INSUFFICIENT_SAMPLES"


def test_check_rejects_auto_on_threshold_breach() -> None:
    ledger = LedgerAuthorityWriter()
    oracle = DriftOracle(ledger=ledger, threshold=0.2, min_samples=2, window_size=4)
    for ts in range(1, 5):
        oracle.observe(_quiet_sample(ts_ns=ts, level=0.5))

    ok, code = oracle.check("AUTO")
    assert ok is False
    assert code == "DRIFT_ORACLE_THRESHOLD_BREACH"


def test_check_passes_auto_when_steady_state_below_threshold() -> None:
    ledger = LedgerAuthorityWriter()
    oracle = DriftOracle(ledger=ledger, threshold=0.5, min_samples=4, window_size=8)
    for ts in range(1, 9):
        oracle.observe(_quiet_sample(ts_ns=ts, level=0.05))

    ok, code = oracle.check("AUTO")
    assert ok is True
    assert code == ""


# ---------------------------------------------------------------------------
# replay_from_ledger
# ---------------------------------------------------------------------------


def test_replay_reconstitutes_window_from_ledger() -> None:
    ledger = LedgerAuthorityWriter()
    o1 = DriftOracle(ledger=ledger, min_samples=2, window_size=4)
    for ts in range(1, 5):
        o1.observe(_quiet_sample(ts_ns=ts, level=0.2))
    composite_before = o1.composite()

    # Fresh oracle, same ledger; replay should adopt the same window.
    o2 = DriftOracle(ledger=ledger, min_samples=2, window_size=4)
    assert o2.sample_count() == 0
    o2.replay_from_ledger()
    assert o2.sample_count() == 4
    assert pytest.approx(o2.composite(), abs=1e-9) == composite_before


def test_replay_keeps_only_latest_window_size_rows() -> None:
    ledger = LedgerAuthorityWriter()
    o1 = DriftOracle(ledger=ledger, min_samples=2, window_size=2)
    # Push more rows than the window holds -- ring buffer drops oldest;
    # the ledger keeps all rows. Replay should still respect the window.
    for ts in range(1, 6):
        o1.observe(_quiet_sample(ts_ns=ts, level=0.1))

    o2 = DriftOracle(ledger=ledger, min_samples=2, window_size=2)
    o2.replay_from_ledger()
    assert o2.sample_count() == 2
    samples = o2.samples()
    assert tuple(s.ts_ns for s in samples) == (4, 5)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_constants_are_self_consistent() -> None:
    """Defaults must satisfy the constructor's own invariants."""

    ledger = LedgerAuthorityWriter()
    oracle = DriftOracle(ledger=ledger)
    assert oracle.threshold == DEFAULT_COMPOSITE_THRESHOLD
    assert oracle.min_samples == DEFAULT_MIN_SAMPLES


# ---------------------------------------------------------------------------
# StateTransitionManager integration
# ---------------------------------------------------------------------------


def _ratchet_to(
    state: StateTransitionManager,
    target: SystemMode,
    *,
    operator_authorized: bool = True,
    ts_ns_start: int = 1,
) -> None:
    """Drive the FSM forward from current to target step-by-step."""

    chain = (
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.SHADOW,
        SystemMode.CANARY,
        SystemMode.LIVE,
        SystemMode.AUTO,
    )
    target_idx = chain.index(target)
    ts = ts_ns_start
    while state.current_mode() is not target:
        cur = state.current_mode()
        cur_idx = chain.index(cur)
        next_mode = chain[cur_idx + 1]
        decision = state.propose(
            ModeTransitionRequest(
                ts_ns=ts,
                requestor="op",
                current_mode=cur,
                target_mode=next_mode,
                reason="ratchet",
                operator_authorized=operator_authorized,
            )
        )
        assert decision.approved is True, (
            f"failed to ratchet {cur} -> {next_mode}: {decision.rejection_code}"
        )
        ts += 1
        if chain.index(next_mode) >= target_idx:
            break


def test_state_manager_without_drift_oracle_unchanged() -> None:
    """Backwards compatibility: no DriftOracle, no drift check."""

    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    state = StateTransitionManager(policy=policy, ledger=ledger)

    _ratchet_to(state, SystemMode.LIVE)
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=100,
            requestor="op",
            current_mode=SystemMode.LIVE,
            target_mode=SystemMode.AUTO,
            reason="enable AUTO",
            operator_authorized=True,
        )
    )
    assert decision.approved is True
    assert state.current_mode() is SystemMode.AUTO


def test_state_manager_refuses_auto_on_insufficient_samples() -> None:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    oracle = DriftOracle(ledger=ledger, min_samples=4, window_size=8)
    state = StateTransitionManager(policy=policy, ledger=ledger, drift_oracle=oracle)

    _ratchet_to(state, SystemMode.LIVE)
    # No samples observed -- AUTO must refuse.
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=100,
            requestor="op",
            current_mode=SystemMode.LIVE,
            target_mode=SystemMode.AUTO,
            reason="enable AUTO",
            operator_authorized=True,
        )
    )
    assert decision.approved is False
    assert decision.rejection_code == "DRIFT_ORACLE_INSUFFICIENT_SAMPLES"
    assert state.current_mode() is SystemMode.LIVE


def test_state_manager_refuses_auto_on_threshold_breach() -> None:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    oracle = DriftOracle(ledger=ledger, threshold=0.2, min_samples=2, window_size=4)
    state = StateTransitionManager(policy=policy, ledger=ledger, drift_oracle=oracle)

    _ratchet_to(state, SystemMode.LIVE)
    for ts in range(50, 56):
        oracle.observe(_quiet_sample(ts_ns=ts, level=0.5))

    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=100,
            requestor="op",
            current_mode=SystemMode.LIVE,
            target_mode=SystemMode.AUTO,
            reason="enable AUTO",
            operator_authorized=True,
        )
    )
    assert decision.approved is False
    assert decision.rejection_code == "DRIFT_ORACLE_THRESHOLD_BREACH"
    assert state.current_mode() is SystemMode.LIVE


def test_state_manager_allows_auto_when_steady_state() -> None:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    oracle = DriftOracle(ledger=ledger, threshold=0.5, min_samples=2, window_size=4)
    state = StateTransitionManager(policy=policy, ledger=ledger, drift_oracle=oracle)

    _ratchet_to(state, SystemMode.LIVE)
    for ts in range(50, 56):
        oracle.observe(_quiet_sample(ts_ns=ts, level=0.05))

    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=100,
            requestor="op",
            current_mode=SystemMode.LIVE,
            target_mode=SystemMode.AUTO,
            reason="enable AUTO",
            operator_authorized=True,
        )
    )
    assert decision.approved is True
    assert state.current_mode() is SystemMode.AUTO


def test_state_manager_drift_does_not_block_non_auto_transitions() -> None:
    """Non-AUTO transitions must not be touched by the oracle."""

    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    oracle = DriftOracle(ledger=ledger, threshold=0.1, min_samples=4, window_size=8)
    state = StateTransitionManager(policy=policy, ledger=ledger, drift_oracle=oracle)

    # Empty oracle. Must still be able to ratchet SAFE -> ... -> LIVE.
    _ratchet_to(state, SystemMode.LIVE)
    assert state.current_mode() is SystemMode.LIVE


def test_state_manager_drift_rejection_emits_ledger_row() -> None:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    oracle = DriftOracle(ledger=ledger, min_samples=4, window_size=8)
    state = StateTransitionManager(policy=policy, ledger=ledger, drift_oracle=oracle)

    _ratchet_to(state, SystemMode.LIVE)
    state.propose(
        ModeTransitionRequest(
            ts_ns=100,
            requestor="op",
            current_mode=SystemMode.LIVE,
            target_mode=SystemMode.AUTO,
            reason="enable AUTO",
            operator_authorized=True,
        )
    )
    rejected = [r for r in ledger.read() if r.kind == "MODE_TRANSITION_REJECTED"]
    assert any(r.payload["rejection_code"] == "DRIFT_ORACLE_INSUFFICIENT_SAMPLES" for r in rejected)
