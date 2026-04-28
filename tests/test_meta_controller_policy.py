"""Phase 6.T1b — Execution Policy + Shadow Policy tests.

Covers:

* INV-48 — meta-controller must degrade to O(1) on latency budget
  exceedance.
* INV-52 — shadow path emits ``META_DIVERGENCE`` SystemEvents and
  never reaches PolicyEngine.
* Pure-function semantics, frozen dataclass, replay-determinism.
* Authority lint rule **B17** rejects ``governance_engine`` imports
  inside ``shadow_policy``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.coherence.belief_state import Regime
from core.coherence.performance_pressure import PressureVector
from core.contracts.events import Side, SystemEventKind
from intelligence_engine.meta_controller.policy import (
    EXECUTION_POLICY_VERSION,
    FALLBACK_POLICY,
    SHADOW_POLICY_VERSION,
    ExecutionDecision,
    compute_shadow_decision,
    decide_execution_policy,
    divergence_payload,
    emit_divergence_event,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pressure(*, safety_modifier: float = 1.0, uncertainty: float = 0.1) -> PressureVector:
    return PressureVector(
        ts_ns=1,
        perf=0.5,
        risk=0.5,
        drift=0.0,
        latency=0.0,
        uncertainty=uncertainty,
        safety_modifier=safety_modifier,
        cross_signal_entropy=0.0,
        signal_count=0,
    )


# ---------------------------------------------------------------------------
# ExecutionDecision shape
# ---------------------------------------------------------------------------


def test_execution_decision_is_frozen() -> None:
    d = FALLBACK_POLICY
    assert dataclasses.is_dataclass(d)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.side = Side.BUY  # type: ignore[misc]


def test_execution_decision_validates_ranges() -> None:
    with pytest.raises(ValueError, match="size_fraction"):
        ExecutionDecision(
            side=Side.BUY,
            size_fraction=1.5,
            confidence=0.5,
            rationale="primary",
            fallback=False,
        )
    with pytest.raises(ValueError, match="confidence"):
        ExecutionDecision(
            side=Side.BUY,
            size_fraction=0.5,
            confidence=-0.1,
            rationale="primary",
            fallback=False,
        )


def test_fallback_policy_is_constant() -> None:
    """INV-48 sentinel: a HOLD with zero size and the canonical rationale."""
    assert FALLBACK_POLICY.side is Side.HOLD
    assert FALLBACK_POLICY.size_fraction == 0.0
    assert FALLBACK_POLICY.confidence == 0.0
    assert FALLBACK_POLICY.fallback is True
    assert FALLBACK_POLICY.rationale == "latency_budget_exceeded:fallback"
    assert FALLBACK_POLICY.version == EXECUTION_POLICY_VERSION


# ---------------------------------------------------------------------------
# INV-48 latency-budget guard
# ---------------------------------------------------------------------------


def test_latency_budget_exceeded_returns_fallback_constant() -> None:
    """INV-48: budget exceeded -> exactly FALLBACK_POLICY (identity check)."""
    out = decide_execution_policy(
        regime=Regime.TREND_UP,
        pressure=_pressure(),
        proposed_side=Side.BUY,
        proposed_size=0.7,
        proposed_confidence=0.8,
        latency_budget_ns=1_000,
        elapsed_ns=2_000,
    )
    assert out is FALLBACK_POLICY


def test_within_latency_budget_runs_primary() -> None:
    out = decide_execution_policy(
        regime=Regime.TREND_UP,
        pressure=_pressure(),
        proposed_side=Side.BUY,
        proposed_size=0.7,
        proposed_confidence=0.8,
        latency_budget_ns=10_000,
        elapsed_ns=1_000,
    )
    assert out is not FALLBACK_POLICY
    assert out.side is Side.BUY
    assert out.rationale == "primary"
    assert out.fallback is False


# ---------------------------------------------------------------------------
# Pressure / regime guards
# ---------------------------------------------------------------------------


def test_zero_safety_modifier_short_circuits_to_hold() -> None:
    out = decide_execution_policy(
        regime=Regime.TREND_UP,
        pressure=_pressure(safety_modifier=0.0),
        proposed_side=Side.BUY,
        proposed_size=0.9,
        proposed_confidence=0.9,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    assert out.fallback is True
    assert out.side is Side.HOLD
    assert out.size_fraction == 0.0
    assert out.rationale == "safety_modifier_zero:fallback"


def test_unknown_regime_short_circuits_to_hold() -> None:
    out = decide_execution_policy(
        regime=Regime.UNKNOWN,
        pressure=_pressure(),
        proposed_side=Side.BUY,
        proposed_size=0.9,
        proposed_confidence=0.9,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    assert out.fallback is True
    assert out.side is Side.HOLD
    assert out.rationale == "unknown_regime:fallback"


def test_trend_down_rejects_buy_proposal() -> None:
    """A BUY proposal in TREND_DOWN collapses to HOLD."""
    out = decide_execution_policy(
        regime=Regime.TREND_DOWN,
        pressure=_pressure(),
        proposed_side=Side.BUY,
        proposed_size=0.9,
        proposed_confidence=0.9,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    assert out.side is Side.HOLD
    assert out.size_fraction == 0.0
    assert out.rationale == "primary"
    assert out.fallback is False


def test_vol_spike_collapses_to_hold() -> None:
    out = decide_execution_policy(
        regime=Regime.VOL_SPIKE,
        pressure=_pressure(),
        proposed_side=Side.BUY,
        proposed_size=0.9,
        proposed_confidence=0.9,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    assert out.side is Side.HOLD


def test_safety_modifier_scales_size_and_confidence() -> None:
    out = decide_execution_policy(
        regime=Regime.TREND_UP,
        pressure=_pressure(safety_modifier=0.5),
        proposed_side=Side.BUY,
        proposed_size=0.8,
        proposed_confidence=0.8,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    assert out.side is Side.BUY
    assert out.size_fraction == pytest.approx(0.4)
    assert out.confidence == pytest.approx(0.4)


def test_proposed_size_clamped() -> None:
    out = decide_execution_policy(
        regime=Regime.TREND_UP,
        pressure=_pressure(safety_modifier=1.0),
        proposed_side=Side.BUY,
        proposed_size=1.5,
        proposed_confidence=1.5,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    assert out.size_fraction == 1.0
    assert out.confidence == 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_decide_is_replay_deterministic() -> None:
    kwargs = dict(
        regime=Regime.TREND_UP,
        pressure=_pressure(safety_modifier=0.7),
        proposed_side=Side.BUY,
        proposed_size=0.6,
        proposed_confidence=0.6,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    a = decide_execution_policy(**kwargs)  # type: ignore[arg-type]
    b = decide_execution_policy(**kwargs)  # type: ignore[arg-type]
    assert a == b


# ---------------------------------------------------------------------------
# Shadow policy (INV-52)
# ---------------------------------------------------------------------------


def test_shadow_ignores_latency_fallback() -> None:
    """INV-52 shadow does NOT degrade on latency; primary does."""
    primary = decide_execution_policy(
        regime=Regime.TREND_UP,
        pressure=_pressure(),
        proposed_side=Side.BUY,
        proposed_size=0.7,
        proposed_confidence=0.8,
        latency_budget_ns=1_000,
        elapsed_ns=2_000,
    )
    shadow = compute_shadow_decision(
        regime=Regime.TREND_UP,
        pressure=_pressure(),
        proposed_side=Side.BUY,
        proposed_size=0.7,
        proposed_confidence=0.8,
        latency_budget_ns=1_000,
        elapsed_ns=2_000,
    )
    assert primary is FALLBACK_POLICY
    assert shadow.fallback is False
    assert shadow.side is Side.BUY


def test_shadow_ignores_safety_modifier_damping() -> None:
    """INV-52 shadow uses raw size; primary folds in safety_modifier."""
    primary = decide_execution_policy(
        regime=Regime.TREND_UP,
        pressure=_pressure(safety_modifier=0.4),
        proposed_side=Side.BUY,
        proposed_size=0.8,
        proposed_confidence=0.8,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    shadow = compute_shadow_decision(
        regime=Regime.TREND_UP,
        pressure=_pressure(safety_modifier=0.4),
        proposed_side=Side.BUY,
        proposed_size=0.8,
        proposed_confidence=0.8,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    assert primary.size_fraction == pytest.approx(0.32)
    assert shadow.size_fraction == pytest.approx(0.8)


def test_shadow_passthrough_when_safety_modifier_is_one() -> None:
    """Shadow == primary when no damping and within budget."""
    kwargs = dict(
        regime=Regime.TREND_UP,
        pressure=_pressure(safety_modifier=1.0),
        proposed_side=Side.BUY,
        proposed_size=0.5,
        proposed_confidence=0.5,
        latency_budget_ns=10_000,
        elapsed_ns=0,
    )
    primary = decide_execution_policy(**kwargs)  # type: ignore[arg-type]
    shadow = compute_shadow_decision(**kwargs)  # type: ignore[arg-type]
    assert primary == shadow


# ---------------------------------------------------------------------------
# Divergence event
# ---------------------------------------------------------------------------


def test_divergence_event_returns_none_when_equal() -> None:
    d = ExecutionDecision(
        side=Side.BUY,
        size_fraction=0.5,
        confidence=0.5,
        rationale="primary",
        fallback=False,
    )
    assert emit_divergence_event(ts_ns=1, primary=d, shadow=d) is None


def test_divergence_event_emits_meta_divergence_kind() -> None:
    primary = FALLBACK_POLICY
    shadow = ExecutionDecision(
        side=Side.BUY,
        size_fraction=0.5,
        confidence=0.5,
        rationale="primary",
        fallback=False,
    )
    event = emit_divergence_event(ts_ns=42, primary=primary, shadow=shadow)
    assert event is not None
    assert event.sub_kind is SystemEventKind.META_DIVERGENCE
    assert event.ts_ns == 42
    assert event.source.startswith("intelligence.meta_controller")
    assert event.payload["side_diverged"] == "true"
    assert event.payload["primary_rationale"] == "latency_budget_exceeded:fallback"
    assert event.payload["shadow_rationale"] == "primary"
    assert event.payload["version"] == SHADOW_POLICY_VERSION


def test_divergence_payload_keys_are_stable() -> None:
    primary = ExecutionDecision(
        side=Side.BUY,
        size_fraction=0.4,
        confidence=0.4,
        rationale="primary",
        fallback=False,
    )
    shadow = ExecutionDecision(
        side=Side.SELL,
        size_fraction=0.6,
        confidence=0.6,
        rationale="primary",
        fallback=False,
    )
    payload = divergence_payload(primary=primary, shadow=shadow)
    expected_keys = {
        "primary_side",
        "primary_size",
        "primary_confidence",
        "primary_rationale",
        "primary_fallback",
        "shadow_side",
        "shadow_size",
        "shadow_confidence",
        "shadow_rationale",
        "shadow_fallback",
        "side_diverged",
        "version",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["side_diverged"] == "true"


# ---------------------------------------------------------------------------
# Authority lint B17 — shadow may not import governance_engine
# ---------------------------------------------------------------------------


def test_shadow_policy_does_not_import_governance_engine() -> None:
    """B17 enforced via authority_lint — also verified statically here.

    We parse the AST and check imports rather than searching the raw
    text, since the docstring legitimately *mentions* the engine name
    when explaining the invariant.
    """
    import ast

    src = (
        REPO_ROOT
        / "intelligence_engine"
        / "meta_controller"
        / "policy"
        / "shadow_policy.py"
    ).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("governance_engine")
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("governance_engine")


def test_authority_lint_b17_rule_flags_synthetic_import(tmp_path: Path) -> None:
    """Synthetic violation — fixture is constructed under tests/fixtures so
    the repo-wide lint scan ignores it; we exercise the rule directly."""
    from tools.authority_lint import _check_b17

    out = _check_b17(
        importer="intelligence_engine.meta_controller.policy.shadow_policy",
        target="governance_engine.policy",
        file=tmp_path / "shadow_policy.py",
        line=1,
    )
    assert out is not None
    assert out.rule == "B17"


def test_authority_lint_b17_rule_passes_clean_imports() -> None:
    from tools.authority_lint import _check_b17

    out = _check_b17(
        importer="intelligence_engine.meta_controller.policy.shadow_policy",
        target="core.contracts.events",
        file=Path("/dev/null"),
        line=1,
    )
    assert out is None
