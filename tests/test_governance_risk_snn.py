"""B-15 — BindsNET STDP-trained governance-risk SNN test suite.

Pins sensory-tier authority discipline, frozen-weight immutability,
PostPre STDP correctness, LIF dynamics, and INV-15 byte-identical
replay.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from sensory.neuromorphic import governance_risk_snn
from sensory.neuromorphic.contracts import RiskPulse
from sensory.neuromorphic.governance_risk_snn import (
    MAX_HIDDEN_DIM,
    MAX_INPUT_DIM,
    MAX_TRAIN_STEPS,
    MAX_WINDOW,
    NEW_PIP_DEPENDENCIES,
    SNN_GOVERNANCE_VERSION,
    FrozenSNNWeights,
    GovernanceRiskSNN,
    LIFParams,
    SNNGovernanceError,
    STDPConfig,
    bindsnet_diehl_cook_factory,
    identity_governance_weights,
    stdp_train_offline,
)

MODULE_PATH: Path = Path(governance_risk_snn.__file__)
MODULE_SOURCE: str = MODULE_PATH.read_text(encoding="utf-8")
MODULE_AST: ast.Module = ast.parse(MODULE_SOURCE)


# ================================================================== authority


def _iter_top_level_imports(tree: ast.Module) -> list[ast.AST]:
    out: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(node)
    return out


def test_authority_adapted_from_header() -> None:
    assert MODULE_SOURCE.startswith("# ADAPTED FROM: BindsNET/bindsnet")


def test_authority_version_string() -> None:
    assert SNN_GOVERNANCE_VERSION == "snn-governance-risk/v1"


def test_authority_pip_dependencies_bindsnet_only() -> None:
    assert NEW_PIP_DEPENDENCIES == ("bindsnet",)


def test_authority_no_top_level_bindsnet_or_torch() -> None:
    forbidden = {
        "bindsnet",
        "torch",
        "norse",
        "numpy",
        "scipy",
        "pandas",
        "polars",
        "brian2",
        "snntorch",
    }
    for node in _iter_top_level_imports(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden, (
                    f"forbidden top-level import: {alias.name}"
                )
        else:
            assert node.module is not None
            assert node.module.split(".")[0] not in forbidden, (
                f"forbidden top-level import: from {node.module}"
            )


def test_authority_no_runtime_imports() -> None:
    forbidden_roots = {
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "secrets",
        "socket",
        "subprocess",
        "requests",
        "httpx",
        "websockets",
        "aiohttp",
        "urllib",
        "psutil",
    }
    for node in _iter_top_level_imports(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_roots, (
                    f"forbidden runtime import: {alias.name}"
                )
        else:
            assert node.module is not None
            assert node.module.split(".")[0] not in forbidden_roots, (
                f"forbidden runtime import: from {node.module}"
            )


def test_authority_no_engine_cross_imports() -> None:
    forbidden_roots = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "evolution_engine",
        "learning_engine",
    }
    for node in _iter_top_level_imports(MODULE_AST):
        if isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert node.module.split(".")[0] not in forbidden_roots, (
                f"sensor tier must not import from {node.module}"
            )


def test_authority_no_typed_event_construction() -> None:
    """INV-19: sensor MUST NOT construct typed bus events directly."""

    forbidden_types = {
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "ExecutionIntent",
        "GovernanceDecision",
        "PatchProposal",
        "TradeOutcome",
        "SystemEvent",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                assert fn.id not in forbidden_types, f"sensor must not construct {fn.id}"
            elif isinstance(fn, ast.Attribute):
                assert fn.attr not in forbidden_types, f"sensor must not construct {fn.attr}"


def test_authority_no_core_events_import() -> None:
    for node in _iter_top_level_imports(MODULE_AST):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "core.contracts.events", (
                "sensor must not import core.contracts.events"
            )


def test_authority_emits_only_risk_pulse() -> None:
    """The single output type must be RiskPulse."""

    risk_pulse_calls = 0
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "RiskPulse":
                risk_pulse_calls += 1
    assert risk_pulse_calls >= 1


def test_authority_module_has_no_mutable_globals() -> None:
    for name in dir(governance_risk_snn):
        if name.startswith("_"):
            continue
        attr = getattr(governance_risk_snn, name)
        assert not isinstance(attr, (list, dict, set)), (
            f"governance_risk_snn.{name} is mutable container"
        )


# =================================================================== freezing


def test_stdp_config_frozen() -> None:
    cfg = STDPConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.eta_post = 0.5  # type: ignore[misc]


def test_lif_params_frozen() -> None:
    lif = LIFParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        lif.tau_mem = 0.5  # type: ignore[misc]


def test_frozen_snn_weights_frozen() -> None:
    w = identity_governance_weights(2, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.input_dim = 99  # type: ignore[misc]


def test_governance_risk_snn_frozen() -> None:
    det = GovernanceRiskSNN(weights=identity_governance_weights(2, 2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        det.risk_kind = "OTHER"  # type: ignore[misc]


# ============================================================ stdp config validation


def test_stdp_rejects_negative_eta_post() -> None:
    with pytest.raises(SNNGovernanceError):
        STDPConfig(eta_post=-0.001)


def test_stdp_rejects_negative_eta_pre() -> None:
    with pytest.raises(SNNGovernanceError):
        STDPConfig(eta_pre=-0.001)


def test_stdp_rejects_nonpositive_tau_pre() -> None:
    with pytest.raises(SNNGovernanceError):
        STDPConfig(tau_pre=0.0)


def test_stdp_rejects_nonpositive_tau_post() -> None:
    with pytest.raises(SNNGovernanceError):
        STDPConfig(tau_post=0.0)


def test_stdp_rejects_nonpositive_dt() -> None:
    with pytest.raises(SNNGovernanceError):
        STDPConfig(dt=0.0)


def test_stdp_rejects_w_min_ge_w_max() -> None:
    with pytest.raises(SNNGovernanceError):
        STDPConfig(w_min=1.0, w_max=1.0)


def test_stdp_rejects_nan_w_min() -> None:
    with pytest.raises(SNNGovernanceError):
        STDPConfig(w_min=float("nan"))


# ============================================================ lif params validation


def test_lif_rejects_nonpositive_tau_mem() -> None:
    with pytest.raises(SNNGovernanceError):
        LIFParams(tau_mem=0.0)


def test_lif_rejects_dt_greater_than_tau() -> None:
    with pytest.raises(SNNGovernanceError):
        LIFParams(tau_mem=1.0e-3, dt=1.0e-2)


def test_lif_rejects_nan_threshold() -> None:
    with pytest.raises(SNNGovernanceError):
        LIFParams(v_threshold=float("nan"))


def test_lif_rejects_nan_reset() -> None:
    with pytest.raises(SNNGovernanceError):
        LIFParams(v_reset=float("nan"))


# ================================================================ weights validation


def test_weights_rejects_input_dim_zero() -> None:
    with pytest.raises(SNNGovernanceError):
        FrozenSNNWeights(weight=(), input_dim=0, hidden_dim=2)


def test_weights_rejects_input_dim_too_large() -> None:
    with pytest.raises(SNNGovernanceError):
        identity_governance_weights(MAX_INPUT_DIM + 1, 2)


def test_weights_rejects_hidden_dim_too_large() -> None:
    with pytest.raises(SNNGovernanceError):
        identity_governance_weights(2, MAX_HIDDEN_DIM + 1)


def test_weights_rejects_row_length_mismatch() -> None:
    with pytest.raises(SNNGovernanceError):
        FrozenSNNWeights(
            weight=((1.0, 0.0), (0.0,)),
            input_dim=2,
            hidden_dim=2,
        )


def test_weights_rejects_row_count_mismatch() -> None:
    with pytest.raises(SNNGovernanceError):
        FrozenSNNWeights(
            weight=((1.0, 0.0),),
            input_dim=2,
            hidden_dim=2,
        )


def test_weights_rejects_nan_entry() -> None:
    with pytest.raises(SNNGovernanceError):
        FrozenSNNWeights(
            weight=((float("nan"), 0.0), (0.0, 1.0)),
            input_dim=2,
            hidden_dim=2,
        )


def test_weights_rejects_inf_entry() -> None:
    with pytest.raises(SNNGovernanceError):
        FrozenSNNWeights(
            weight=((float("inf"), 0.0), (0.0, 1.0)),
            input_dim=2,
            hidden_dim=2,
        )


def test_identity_weights_shape() -> None:
    w = identity_governance_weights(3, 3)
    assert w.input_dim == 3
    assert w.hidden_dim == 3
    for i in range(3):
        for j in range(3):
            assert w.weight[i][j] == (1.0 if i == j else 0.0)


def test_identity_weights_rectangular_input_larger() -> None:
    w = identity_governance_weights(4, 2)
    assert w.weight[0] == (1.0, 0.0)
    assert w.weight[1] == (0.0, 1.0)
    assert w.weight[2] == (0.0, 0.0)
    assert w.weight[3] == (0.0, 0.0)


def test_identity_weights_rectangular_hidden_larger() -> None:
    w = identity_governance_weights(2, 4)
    assert w.weight[0] == (1.0, 0.0, 0.0, 0.0)
    assert w.weight[1] == (0.0, 1.0, 0.0, 0.0)


def test_weights_digest_blake2b16_hex() -> None:
    w = identity_governance_weights(2, 2)
    d = w.digest()
    assert len(d) == 32
    int(d, 16)  # validates hex


def test_weights_digest_deterministic() -> None:
    w1 = identity_governance_weights(3, 3)
    w2 = identity_governance_weights(3, 3)
    assert w1.digest() == w2.digest()


def test_weights_digest_sensitive_to_entry() -> None:
    w1 = identity_governance_weights(2, 2)
    w2 = FrozenSNNWeights(
        weight=((1.0, 0.5), (0.0, 1.0)),
        input_dim=2,
        hidden_dim=2,
    )
    assert w1.digest() != w2.digest()


# ===================================================================== STDP math


def _zero_train(n_steps: int, dim: int) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple([False] * dim) for _ in range(n_steps))


def _all_spike(n_steps: int, dim: int) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple([True] * dim) for _ in range(n_steps))


def test_stdp_zero_train_preserves_weights() -> None:
    w0 = identity_governance_weights(2, 2)
    cfg = STDPConfig()
    w1 = stdp_train_offline(
        initial_weights=w0,
        pre_spikes=_zero_train(10, 2),
        post_spikes=_zero_train(10, 2),
        stdp=cfg,
    )
    assert w1.weight == w0.weight


def test_stdp_ltp_strengthens_correlated_connection() -> None:
    """When pre fires shortly before post, weight should increase (LTP)."""

    w0 = FrozenSNNWeights(
        weight=((0.5, 0.0), (0.0, 0.5)),
        input_dim=2,
        hidden_dim=2,
    )
    cfg = STDPConfig(eta_post=0.1, eta_pre=0.0)  # LTP only
    # Step 0: pre[0] fires. Step 1: post[0] fires.
    pre = ((True, False), (False, False))
    post = ((False, False), (True, False))
    w1 = stdp_train_offline(initial_weights=w0, pre_spikes=pre, post_spikes=post, stdp=cfg)
    assert w1.weight[0][0] > w0.weight[0][0]


def test_stdp_ltd_weakens_anticorrelated_connection() -> None:
    """When post fires shortly before pre, weight should decrease (LTD)."""

    w0 = FrozenSNNWeights(
        weight=((0.5, 0.0), (0.0, 0.5)),
        input_dim=2,
        hidden_dim=2,
    )
    cfg = STDPConfig(eta_post=0.0, eta_pre=0.1)  # LTD only
    # Step 0: post[0] fires. Step 1: pre[0] fires.
    pre = ((False, False), (True, False))
    post = ((True, False), (False, False))
    w1 = stdp_train_offline(initial_weights=w0, pre_spikes=pre, post_spikes=post, stdp=cfg)
    assert w1.weight[0][0] < w0.weight[0][0]


def test_stdp_clips_to_w_max() -> None:
    w0 = FrozenSNNWeights(
        weight=((0.99, 0.0), (0.0, 0.99)),
        input_dim=2,
        hidden_dim=2,
    )
    cfg = STDPConfig(eta_post=10.0, eta_pre=0.0, w_max=1.0)
    # Many pre spikes followed by post spikes — would push weight >> 1.
    pre = tuple(tuple([True, False]) for _ in range(50))
    post = tuple(tuple([True, False]) for _ in range(50))
    w1 = stdp_train_offline(initial_weights=w0, pre_spikes=pre, post_spikes=post, stdp=cfg)
    assert w1.weight[0][0] <= 1.0 + 1e-12


def test_stdp_clips_to_w_min() -> None:
    w0 = FrozenSNNWeights(
        weight=((0.01, 0.0), (0.0, 0.01)),
        input_dim=2,
        hidden_dim=2,
    )
    cfg = STDPConfig(eta_post=0.0, eta_pre=10.0, w_min=0.0)
    pre = tuple(tuple([True, False]) for _ in range(50))
    post = tuple(tuple([True, False]) for _ in range(50))
    w1 = stdp_train_offline(initial_weights=w0, pre_spikes=pre, post_spikes=post, stdp=cfg)
    assert w1.weight[0][0] >= -1e-12


def test_stdp_returns_new_instance_not_mutation() -> None:
    w0 = identity_governance_weights(2, 2)
    weight_before = w0.weight
    cfg = STDPConfig(eta_post=0.1)
    _ = stdp_train_offline(
        initial_weights=w0,
        pre_spikes=((True, True), (True, True)),
        post_spikes=((True, True), (True, True)),
        stdp=cfg,
    )
    # Original instance unchanged.
    assert w0.weight == weight_before


def test_stdp_deterministic_three_runs() -> None:
    w0 = identity_governance_weights(3, 2)
    cfg = STDPConfig(eta_post=0.05, eta_pre=0.03)
    pre = tuple(tuple([(t + i) % 2 == 0 for i in range(3)]) for t in range(20))
    post = tuple(tuple([t % 3 == 0, t % 5 == 0]) for t in range(20))
    a = stdp_train_offline(initial_weights=w0, pre_spikes=pre, post_spikes=post, stdp=cfg)
    b = stdp_train_offline(initial_weights=w0, pre_spikes=pre, post_spikes=post, stdp=cfg)
    c = stdp_train_offline(initial_weights=w0, pre_spikes=pre, post_spikes=post, stdp=cfg)
    assert a.weight == b.weight == c.weight
    assert a.digest() == b.digest() == c.digest()


def test_stdp_rejects_empty_train() -> None:
    cfg = STDPConfig()
    w0 = identity_governance_weights(2, 2)
    with pytest.raises(SNNGovernanceError):
        stdp_train_offline(
            initial_weights=w0,
            pre_spikes=(),
            post_spikes=(),
            stdp=cfg,
        )


def test_stdp_rejects_length_mismatch() -> None:
    cfg = STDPConfig()
    w0 = identity_governance_weights(2, 2)
    with pytest.raises(SNNGovernanceError):
        stdp_train_offline(
            initial_weights=w0,
            pre_spikes=_zero_train(10, 2),
            post_spikes=_zero_train(11, 2),
            stdp=cfg,
        )


def test_stdp_rejects_pre_row_dim_mismatch() -> None:
    cfg = STDPConfig()
    w0 = identity_governance_weights(2, 2)
    bad_pre = (tuple([True, False, False]),)
    with pytest.raises(SNNGovernanceError):
        stdp_train_offline(
            initial_weights=w0,
            pre_spikes=bad_pre,
            post_spikes=((True, False),),
            stdp=cfg,
        )


# ===================================================================== detector


def _detector(input_dim: int = 2, hidden_dim: int = 2) -> GovernanceRiskSNN:
    return GovernanceRiskSNN(
        weights=identity_governance_weights(input_dim, hidden_dim),
        lif=LIFParams(tau_mem=20.0e-3, v_threshold=0.5, dt=1.0e-3),
    )


def test_detector_emits_risk_pulse() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1_000,
        source="governance.decision_audit",
        window=[(1.0, 0.0), (1.0, 0.0)],
    )
    assert isinstance(pulse, RiskPulse)
    assert pulse.ts_ns == 1_000
    assert pulse.source == "governance.decision_audit"
    assert pulse.risk_kind == "GOVERNANCE_PATTERN_RISK"
    assert pulse.sample_count == 2


def test_detector_zero_input_zero_risk() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1,
        source="gov",
        window=[(0.0, 0.0)] * 16,
    )
    assert pulse.risk_score == 0.0


def test_detector_strong_input_drives_spikes() -> None:
    """Sustained high input on channel 0 should drive risk_score > 0."""

    det = GovernanceRiskSNN(
        weights=identity_governance_weights(2, 2),
        lif=LIFParams(tau_mem=20.0e-3, v_threshold=0.2, dt=1.0e-3),
    )
    pulse = det.detect(
        ts_ns=1,
        source="gov",
        window=[(50.0, 0.0)] * 8,
    )
    assert pulse.risk_score > 0.0


def test_detector_risk_score_in_unit_interval() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1,
        source="gov",
        window=[(1000.0, 1000.0)] * 4,
    )
    assert 0.0 <= pulse.risk_score <= 1.0


def test_detector_records_weights_digest() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1,
        source="gov",
        window=[(0.0, 0.0)] * 2,
    )
    assert pulse.evidence["weights_digest"] == det.weights.digest()
    assert len(pulse.evidence["weights_digest"]) == 32


def test_detector_records_spike_count() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1,
        source="gov",
        window=[(0.0, 0.0)] * 2,
    )
    assert pulse.evidence["spike_count"] == "0"


def test_detector_override_risk_kind() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1,
        source="gov",
        risk_kind="CUSTOM_RISK",
        window=[(0.0, 0.0)] * 2,
    )
    assert pulse.risk_kind == "CUSTOM_RISK"


def test_detector_rejects_empty_window() -> None:
    det = _detector()
    with pytest.raises(SNNGovernanceError):
        det.detect(ts_ns=1, source="gov", window=[])


def test_detector_rejects_oversized_window() -> None:
    det = _detector()
    with pytest.raises(SNNGovernanceError):
        det.detect(
            ts_ns=1,
            source="gov",
            window=[(0.0, 0.0)] * (MAX_WINDOW + 1),
        )


def test_detector_rejects_negative_ts_ns() -> None:
    det = _detector()
    with pytest.raises(SNNGovernanceError):
        det.detect(ts_ns=-1, source="gov", window=[(0.0, 0.0)])


def test_detector_rejects_empty_source() -> None:
    det = _detector()
    with pytest.raises(SNNGovernanceError):
        det.detect(ts_ns=1, source="", window=[(0.0, 0.0)])


def test_detector_rejects_row_dim_mismatch() -> None:
    det = _detector(input_dim=2, hidden_dim=2)
    with pytest.raises(SNNGovernanceError):
        det.detect(
            ts_ns=1,
            source="gov",
            window=[(0.0, 0.0, 0.0)],
        )


def test_detector_rejects_nan_in_window() -> None:
    det = _detector()
    with pytest.raises(SNNGovernanceError):
        det.detect(
            ts_ns=1,
            source="gov",
            window=[(float("nan"), 0.0)],
        )


def test_detector_rejects_empty_risk_kind() -> None:
    det = _detector()
    with pytest.raises(SNNGovernanceError):
        det.detect(ts_ns=1, source="gov", window=[(0.0, 0.0)], risk_kind="")


def test_detector_evidence_merges_caller_metadata() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1,
        source="gov",
        window=[(0.0, 0.0)] * 2,
        evidence={"strategy_id": "alpha", "regime": "RANGE"},
    )
    assert pulse.evidence["strategy_id"] == "alpha"
    assert pulse.evidence["regime"] == "RANGE"
    assert "weights_digest" in pulse.evidence


def test_detector_stateless_across_calls() -> None:
    det = _detector()
    w1 = [(0.0, 0.0)] * 4
    a = det.detect(ts_ns=1, source="gov", window=w1)
    # Inject noise then re-run with same window — must give same result.
    _ = det.detect(ts_ns=2, source="gov", window=[(100.0, 100.0)] * 8)
    b = det.detect(ts_ns=1, source="gov", window=w1)
    assert a == b


# ================================================================ INV-15 replay


def test_replay_byte_identical_three_runs() -> None:
    det = _detector(input_dim=3, hidden_dim=2)
    window = [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)] * 5
    a = det.detect(ts_ns=12345, source="gov.audit", window=window)
    b = det.detect(ts_ns=12345, source="gov.audit", window=window)
    c = det.detect(ts_ns=12345, source="gov.audit", window=window)
    assert a == b == c


def test_replay_ts_change_changes_pulse() -> None:
    det = _detector()
    window = [(0.0, 0.0)]
    a = det.detect(ts_ns=1, source="gov", window=window)
    b = det.detect(ts_ns=2, source="gov", window=window)
    assert a.ts_ns != b.ts_ns
    assert a != b


def test_replay_input_change_changes_score() -> None:
    det = GovernanceRiskSNN(
        weights=identity_governance_weights(2, 2),
        lif=LIFParams(tau_mem=20.0e-3, v_threshold=0.05, dt=1.0e-3),
    )
    a = det.detect(ts_ns=1, source="gov", window=[(0.0, 0.0)] * 8)
    b = det.detect(ts_ns=1, source="gov", window=[(50.0, 0.0)] * 8)
    assert a.risk_score != b.risk_score


def test_replay_evidence_key_order_invariant() -> None:
    det = _detector()
    a = det.detect(
        ts_ns=1,
        source="gov",
        window=[(0.0, 0.0)],
        evidence={"a": "1", "b": "2"},
    )
    b = det.detect(
        ts_ns=1,
        source="gov",
        window=[(0.0, 0.0)],
        evidence={"b": "2", "a": "1"},
    )
    assert a == b


# ============================================================ production seam


def test_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        bindsnet_diehl_cook_factory("path/to/weights.pt")


# ============================================================ misc smoke tests


def test_constants_positive() -> None:
    assert MAX_WINDOW > 0
    assert MAX_INPUT_DIM > 0
    assert MAX_HIDDEN_DIM > 0
    assert MAX_TRAIN_STEPS > 0


def test_stdp_pre_trace_decays_between_steps() -> None:
    """A gap of many silent steps should attenuate the LTP delta."""

    w0 = FrozenSNNWeights(
        weight=((0.5, 0.0), (0.0, 0.5)),
        input_dim=2,
        hidden_dim=2,
    )
    cfg = STDPConfig(eta_post=0.1, eta_pre=0.0, tau_pre=2.0e-3, dt=1.0e-3)
    short_gap_pre = ((True, False), (False, False))
    short_gap_post = ((False, False), (True, False))
    long_gap_pre = ((True, False),) + _zero_train(20, 2) + ((False, False),)
    long_gap_post = _zero_train(21, 2) + ((True, False),)
    short = stdp_train_offline(
        initial_weights=w0,
        pre_spikes=short_gap_pre,
        post_spikes=short_gap_post,
        stdp=cfg,
    )
    long_ = stdp_train_offline(
        initial_weights=w0,
        pre_spikes=long_gap_pre,
        post_spikes=long_gap_post,
        stdp=cfg,
    )
    short_delta = short.weight[0][0] - w0.weight[0][0]
    long_delta = long_.weight[0][0] - w0.weight[0][0]
    assert short_delta > long_delta


def test_lif_params_default_consistent_with_b14() -> None:
    """Hyperparameters mirror sensory.neuromorphic.snn_lif.LIFConfig."""

    lif = LIFParams()
    assert math.isclose(lif.tau_mem, 20.0e-3)
    assert math.isclose(lif.dt, 1.0e-3)
    assert lif.v_threshold == 1.0


def test_stdp_config_default_w_range() -> None:
    cfg = STDPConfig()
    assert cfg.w_min == 0.0
    assert cfg.w_max == 1.0
