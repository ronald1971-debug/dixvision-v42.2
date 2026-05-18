"""B-17 snntorch_detector — authority + freezing + math + INV-15 tests."""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from sensory.neuromorphic import snntorch_detector as snn
from sensory.neuromorphic.snntorch_detector import (
    RESET_SUBTRACT,
    RESET_ZERO,
    LeakyConfig,
    LeakyState,
    LeakyWeights,
    SNNTorchDetector,
    SNNTorchDetectorError,
    SNNTorchLeakyCell,
    SpikePulse,
    benchmark_against_norse,
    beta_from_tau,
    identity_leaky_weights,
    initial_leaky_state,
    leaky_feed_forward_step,
    snntorch_cell_factory,
)

_MODULE_PATH = Path(snn.__file__)
_REPO_ROOT = _MODULE_PATH.parents[2]


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


# -------------------------------------------------------- authority pins


def test_authority_adapted_from_header() -> None:
    text = _MODULE_PATH.read_text(encoding="utf-8")
    assert text.startswith("# ADAPTED FROM: jeshraghian/snntorch"), (
        "snntorch_detector.py must declare its adaptation source on line 1"
    )


def test_authority_pip_dependencies_snntorch_torch() -> None:
    assert snn.NEW_PIP_DEPENDENCIES == ("snntorch", "torch")


def test_authority_no_top_level_research_imports() -> None:
    forbidden = {
        "snntorch",
        "torch",
        "norse",
        "numpy",
        "scipy",
        "pandas",
        "polars",
        "bindsnet",
        "brian2",
    }
    tree = _module_ast()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".", 1)[0]
                assert head not in forbidden, f"top-level import of {alias.name!r} is forbidden"
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            assert module not in forbidden, f"top-level from-import of {node.module!r} is forbidden"


def test_authority_no_clock_random_or_io() -> None:
    forbidden = {
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
        "aiohttp",
        "websockets",
        "urllib",
        "psutil",
    }
    tree = _module_ast()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".", 1)[0]
                assert head not in forbidden, f"top-level import of {alias.name!r} is forbidden"
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            assert module not in forbidden, f"top-level from-import of {node.module!r} is forbidden"


def test_authority_no_engine_cross_imports() -> None:
    forbidden_prefixes = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "evolution_engine",
        "learning_engine",
    )
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(forbidden_prefixes), f"forbidden cross-import: {module!r}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes), (
                    f"forbidden cross-import: {alias.name!r}"
                )


def test_authority_no_typed_event_construction() -> None:
    forbidden = {
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "ExecutionIntent",
        "GovernanceDecision",
        "PatchProposal",
        "RealitySummary",
        "RiskPulse",
    }
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden:
                pytest.fail(f"forbidden typed-event construction: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in forbidden:
                pytest.fail(f"forbidden typed-event construction: {func.attr}")


def test_authority_advisory_only_documented() -> None:
    text = _MODULE_PATH.read_text(encoding="utf-8")
    assert "advisory" in text.lower()
    assert "INV-19" in text or "NEUR-02" in text
    assert "INV-20" in text


# -------------------------------------------------------- freezing


def test_freezing_leaky_config() -> None:
    cfg = LeakyConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.beta = 0.5  # type: ignore[misc]


def test_freezing_leaky_weights() -> None:
    w = identity_leaky_weights(2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.bias = (0.0, 0.0)  # type: ignore[misc]


def test_freezing_leaky_state() -> None:
    s = initial_leaky_state(3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.v = (0.0, 0.0, 0.0)  # type: ignore[misc]


def test_freezing_spike_pulse() -> None:
    pulse = SpikePulse(
        ts_ns=1,
        source="snn",
        symbol="X",
        polarity="NEUTRAL",
        intensity=0.0,
        spike_count=0,
        sample_count=1,
        weights_digest="0" * 32,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        pulse.intensity = 0.5  # type: ignore[misc]


def test_freezing_backend_benchmark() -> None:
    b = benchmark_against_norse(input_current=(0.0,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.spike_count_delta = 1  # type: ignore[misc]


# -------------------------------------------------------- config validation


def test_config_beta_must_be_in_open_unit() -> None:
    with pytest.raises(SNNTorchDetectorError):
        LeakyConfig(beta=0.0)
    with pytest.raises(SNNTorchDetectorError):
        LeakyConfig(beta=1.0)
    with pytest.raises(SNNTorchDetectorError):
        LeakyConfig(beta=-0.1)
    with pytest.raises(SNNTorchDetectorError):
        LeakyConfig(beta=float("nan"))


def test_config_threshold_and_reset_must_be_finite() -> None:
    with pytest.raises(SNNTorchDetectorError):
        LeakyConfig(v_threshold=float("inf"))
    with pytest.raises(SNNTorchDetectorError):
        LeakyConfig(v_reset=float("nan"))


def test_config_reset_mechanism_validated() -> None:
    with pytest.raises(SNNTorchDetectorError):
        LeakyConfig(reset_mechanism="bogus")


def test_beta_from_tau_basic() -> None:
    beta = beta_from_tau(dt=1e-3, tau_mem=1e-2)
    assert math.isclose(beta, math.exp(-0.1))


def test_beta_from_tau_rejects_bad_inputs() -> None:
    with pytest.raises(SNNTorchDetectorError):
        beta_from_tau(dt=0.0, tau_mem=1e-2)
    with pytest.raises(SNNTorchDetectorError):
        beta_from_tau(dt=1e-3, tau_mem=0.0)
    with pytest.raises(SNNTorchDetectorError):
        beta_from_tau(dt=float("inf"), tau_mem=1.0)


# -------------------------------------------------------- weight validation


def test_weights_validates_dims_and_finiteness() -> None:
    with pytest.raises(SNNTorchDetectorError):
        LeakyWeights(weight=(), bias=(), input_dim=0, hidden_dim=1)
    with pytest.raises(SNNTorchDetectorError):
        LeakyWeights(weight=((float("nan"),),), bias=(0.0,), input_dim=1, hidden_dim=1)
    with pytest.raises(SNNTorchDetectorError):
        LeakyWeights(weight=((1.0,),), bias=(0.0, 0.0), input_dim=1, hidden_dim=1)


def test_identity_weights_is_diagonal() -> None:
    w = identity_leaky_weights(3)
    assert w.input_dim == 3
    assert w.hidden_dim == 3
    for i, row in enumerate(w.weight):
        for j, v in enumerate(row):
            expected = 1.0 if i == j else 0.0
            assert v == expected


def test_identity_weights_digest_stable() -> None:
    a = identity_leaky_weights(4).digest()
    b = identity_leaky_weights(4).digest()
    c = identity_leaky_weights(4).digest()
    assert a == b == c
    assert len(a) == 32


def test_identity_weights_digest_differs_on_dim() -> None:
    assert identity_leaky_weights(2).digest() != identity_leaky_weights(3).digest()


# -------------------------------------------------------- LIF math


def test_leaky_step_no_input_no_spike() -> None:
    cfg = LeakyConfig(beta=0.9)
    s0 = initial_leaky_state(3)
    s1, spikes = leaky_feed_forward_step(s0, (0.0, 0.0, 0.0), cfg)
    assert spikes == (False, False, False)
    assert s1.v == (0.0, 0.0, 0.0)


def test_leaky_step_strong_drive_spikes_with_subtract_reset() -> None:
    cfg = LeakyConfig(beta=0.9, v_threshold=1.0, reset_mechanism=RESET_SUBTRACT)
    s0 = initial_leaky_state(1)
    s1, spikes = leaky_feed_forward_step(s0, (2.0,), cfg)
    assert spikes == (True,)
    # 0.9*0 + 2.0 = 2.0, subtract threshold -> 1.0
    assert math.isclose(s1.v[0], 1.0)


def test_leaky_step_strong_drive_zero_reset() -> None:
    cfg = LeakyConfig(beta=0.9, v_threshold=1.0, reset_mechanism=RESET_ZERO)
    s0 = initial_leaky_state(1)
    s1, spikes = leaky_feed_forward_step(s0, (2.0,), cfg)
    assert spikes == (True,)
    assert s1.v[0] == 0.0


def test_leaky_step_subthreshold_accumulates() -> None:
    cfg = LeakyConfig(beta=0.9, v_threshold=10.0)
    s0 = initial_leaky_state(1)
    s1, spikes = leaky_feed_forward_step(s0, (0.5,), cfg)
    assert spikes == (False,)
    assert math.isclose(s1.v[0], 0.5)
    s2, spikes2 = leaky_feed_forward_step(s1, (0.5,), cfg)
    # 0.9*0.5 + 0.5 = 0.95
    assert math.isclose(s2.v[0], 0.95)
    assert spikes2 == (False,)


def test_leaky_step_length_mismatch_raises() -> None:
    cfg = LeakyConfig()
    s0 = initial_leaky_state(2)
    with pytest.raises(SNNTorchDetectorError):
        leaky_feed_forward_step(s0, (1.0,), cfg)


def test_leaky_step_nan_input_raises() -> None:
    cfg = LeakyConfig()
    s0 = initial_leaky_state(1)
    with pytest.raises(SNNTorchDetectorError):
        leaky_feed_forward_step(s0, (float("nan"),), cfg)


# -------------------------------------------------------- cell


def test_cell_forward_identity_passthrough() -> None:
    cell = SNNTorchLeakyCell(
        weights=identity_leaky_weights(2),
        config=LeakyConfig(beta=0.9, v_threshold=10.0),
    )
    state = initial_leaky_state(2)
    s1, spikes = cell.forward(state, (0.3, 0.7))
    assert spikes == (False, False)
    assert math.isclose(s1.v[0], 0.3)
    assert math.isclose(s1.v[1], 0.7)


def test_cell_dim_mismatch_raises() -> None:
    cell = SNNTorchLeakyCell(weights=identity_leaky_weights(2))
    bad_state = initial_leaky_state(3)
    with pytest.raises(SNNTorchDetectorError):
        cell.forward(bad_state, (0.0, 0.0))


# -------------------------------------------------------- detector


def _make_detector(dim: int = 2, threshold: float = 1.0) -> SNNTorchDetector:
    return SNNTorchDetector(
        cell=SNNTorchLeakyCell(
            weights=identity_leaky_weights(dim),
            config=LeakyConfig(beta=0.9, v_threshold=threshold),
        )
    )


def test_detector_silent_window_returns_neutral() -> None:
    det = _make_detector(dim=2, threshold=10.0)
    pulse = det.detect(
        ts_ns=1,
        source="snn",
        symbol="BTCUSDT",
        window=[(0.1, 0.1)] * 5,
    )
    assert pulse.spike_count == 0
    assert pulse.polarity == "NEUTRAL"
    assert pulse.intensity == 0.0
    assert pulse.sample_count == 5


def test_detector_dense_drive_emits_long_polarity() -> None:
    det = _make_detector(dim=2, threshold=1.0)
    pulse = det.detect(
        ts_ns=2,
        source="snn",
        symbol="BTCUSDT",
        window=[(5.0, 5.0)] * 4,
    )
    assert pulse.spike_count == 8
    assert pulse.intensity == 1.0
    assert pulse.polarity == "LONG"


def test_detector_short_polarity_via_sign() -> None:
    det = _make_detector(dim=2, threshold=1.0)
    pulse = det.detect(
        ts_ns=3,
        source="snn",
        symbol="BTCUSDT",
        window=[(5.0, 5.0)] * 4,
        polarity_sign=-1,
    )
    assert pulse.polarity == "SHORT"


def test_detector_neutral_when_polarity_sign_zero() -> None:
    det = _make_detector(dim=2, threshold=1.0)
    pulse = det.detect(
        ts_ns=4,
        source="snn",
        symbol="BTCUSDT",
        window=[(5.0, 5.0)] * 4,
        polarity_sign=0,
    )
    assert pulse.polarity == "NEUTRAL"


def test_detector_rejects_bad_ts() -> None:
    det = _make_detector()
    with pytest.raises(SNNTorchDetectorError):
        det.detect(ts_ns=-1, source="s", symbol="X", window=[])


def test_detector_rejects_oversized_window() -> None:
    det = _make_detector(dim=1)
    big = [(0.0,)] * (snn.MAX_WINDOW + 1)
    with pytest.raises(SNNTorchDetectorError):
        det.detect(ts_ns=1, source="s", symbol="X", window=big)


def test_detector_rejects_empty_source_symbol() -> None:
    det = _make_detector()
    with pytest.raises(SNNTorchDetectorError):
        det.detect(ts_ns=1, source="", symbol="X", window=[])
    with pytest.raises(SNNTorchDetectorError):
        det.detect(ts_ns=1, source="s", symbol="", window=[])


def test_detector_rejects_invalid_polarity_sign() -> None:
    det = _make_detector()
    with pytest.raises(SNNTorchDetectorError):
        det.detect(ts_ns=1, source="s", symbol="X", window=[(0.0, 0.0)], polarity_sign=2)


def test_spike_polarity_threshold_must_be_open_to_closed() -> None:
    with pytest.raises(SNNTorchDetectorError):
        SNNTorchDetector(
            cell=SNNTorchLeakyCell(weights=identity_leaky_weights(1)),
            spike_polarity_threshold=0.0,
        )
    with pytest.raises(SNNTorchDetectorError):
        SNNTorchDetector(
            cell=SNNTorchLeakyCell(weights=identity_leaky_weights(1)),
            spike_polarity_threshold=1.5,
        )


# -------------------------------------------------------- benchmark


def test_benchmark_silent_input_no_spikes() -> None:
    b = benchmark_against_norse(input_current=(0.0,) * 10)
    assert b.norse_spike_count == 0
    assert b.snntorch_spike_count == 0
    assert b.first_spike_step_norse == -1
    assert b.first_spike_step_snntorch == -1
    assert b.first_spike_step_delta == -1
    assert b.is_precision_match()


def test_benchmark_strong_drive_both_spike() -> None:
    b = benchmark_against_norse(input_current=(5.0,) * 30)
    assert b.norse_spike_count > 0
    assert b.snntorch_spike_count > 0
    assert b.first_spike_step_norse >= 0
    assert b.first_spike_step_snntorch >= 0


def test_benchmark_digest_stable_three_runs() -> None:
    trace = (0.1, 0.2, 0.3, 5.0, 5.0, 0.0, 0.0)
    d1 = benchmark_against_norse(input_current=trace).digest
    d2 = benchmark_against_norse(input_current=trace).digest
    d3 = benchmark_against_norse(input_current=trace).digest
    assert d1 == d2 == d3
    assert len(d1) == 32


def test_benchmark_digest_differs_on_input_change() -> None:
    a = benchmark_against_norse(input_current=(1.0, 1.0, 1.0)).digest
    b = benchmark_against_norse(input_current=(1.0, 1.0, 2.0)).digest
    assert a != b


def test_benchmark_digest_differs_on_reset_mechanism() -> None:
    trace = (2.0,) * 10
    a = benchmark_against_norse(input_current=trace, reset_mechanism=RESET_SUBTRACT).digest
    b = benchmark_against_norse(input_current=trace, reset_mechanism=RESET_ZERO).digest
    assert a != b


def test_benchmark_is_precision_match_count_tolerance() -> None:
    b = benchmark_against_norse(input_current=(5.0,) * 50)
    delta = abs(b.spike_count_delta)
    # the two integrators are NOT byte-equal; subtract reset differs from
    # zero reset and beta-decay differs from Euler decay. Confirm tolerance
    # gating works.
    assert b.is_precision_match(
        count_tolerance=delta,
        first_spike_step_tolerance=abs(b.first_spike_step_norse - b.first_spike_step_snntorch),
    )
    if delta > 0:
        assert not b.is_precision_match(count_tolerance=delta - 1)


def test_benchmark_rejects_empty_trace() -> None:
    with pytest.raises(SNNTorchDetectorError):
        benchmark_against_norse(input_current=())


def test_benchmark_rejects_non_finite_input() -> None:
    with pytest.raises(SNNTorchDetectorError):
        benchmark_against_norse(input_current=(float("nan"),))


def test_benchmark_rejects_bad_reset_mechanism() -> None:
    with pytest.raises(SNNTorchDetectorError):
        benchmark_against_norse(input_current=(1.0,), reset_mechanism="lolnope")


def test_benchmark_rejects_bad_dt_tau() -> None:
    with pytest.raises(SNNTorchDetectorError):
        benchmark_against_norse(input_current=(1.0,), dt=0.0)
    with pytest.raises(SNNTorchDetectorError):
        benchmark_against_norse(input_current=(1.0,), tau_mem=-1.0)


def test_benchmark_is_precision_match_validates_tolerances() -> None:
    b = benchmark_against_norse(input_current=(0.0,) * 5)
    with pytest.raises(SNNTorchDetectorError):
        b.is_precision_match(count_tolerance=-1)
    with pytest.raises(SNNTorchDetectorError):
        b.is_precision_match(first_spike_step_tolerance=-1)


def test_benchmark_pure_silent_match() -> None:
    b = benchmark_against_norse(input_current=(0.0,) * 20)
    assert b.is_precision_match(count_tolerance=0, first_spike_step_tolerance=0)


# -------------------------------------------------------- INV-15 determinism


def test_detector_three_run_pulse_equality() -> None:
    det = _make_detector(dim=2, threshold=1.0)
    window = [(2.0, 2.0)] * 4
    runs = [det.detect(ts_ns=42, source="snn", symbol="BTC", window=window) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_weights_digest_three_run_equality() -> None:
    w1 = identity_leaky_weights(3)
    w2 = identity_leaky_weights(3)
    w3 = identity_leaky_weights(3)
    assert w1.digest() == w2.digest() == w3.digest()


def test_benchmark_three_run_full_equality() -> None:
    trace = (0.0, 0.0, 1.5, 1.5, 0.0, 2.0, 2.0)
    a = benchmark_against_norse(input_current=trace)
    b = benchmark_against_norse(input_current=trace)
    c = benchmark_against_norse(input_current=trace)
    assert a == b == c


# -------------------------------------------------------- production seam


def test_snntorch_cell_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        snntorch_cell_factory()


# -------------------------------------------------------- pulse contract


def test_spike_pulse_validates_polarity() -> None:
    with pytest.raises(SNNTorchDetectorError):
        SpikePulse(
            ts_ns=1,
            source="snn",
            symbol="X",
            polarity="DIAGONAL",
            intensity=0.0,
            spike_count=0,
            sample_count=1,
            weights_digest="0" * 32,
        )


def test_spike_pulse_validates_digest_length() -> None:
    with pytest.raises(SNNTorchDetectorError):
        SpikePulse(
            ts_ns=1,
            source="snn",
            symbol="X",
            polarity="NEUTRAL",
            intensity=0.0,
            spike_count=0,
            sample_count=1,
            weights_digest="abc",
        )


def test_spike_pulse_validates_intensity_range() -> None:
    with pytest.raises(SNNTorchDetectorError):
        SpikePulse(
            ts_ns=1,
            source="snn",
            symbol="X",
            polarity="NEUTRAL",
            intensity=2.0,
            spike_count=0,
            sample_count=1,
            weights_digest="0" * 32,
        )


def test_spike_pulse_validates_sample_count() -> None:
    with pytest.raises(SNNTorchDetectorError):
        SpikePulse(
            ts_ns=1,
            source="snn",
            symbol="X",
            polarity="NEUTRAL",
            intensity=0.0,
            spike_count=0,
            sample_count=0,
            weights_digest="0" * 32,
        )


# -------------------------------------------------------- importer scan


def test_no_production_importers() -> None:
    """sensory.neuromorphic.snntorch_detector is RUNTIME_SAFE inference but
    must only be imported by sensory/, simulation/, tests/, or offline/."""

    allowed_prefixes = ("sensory/", "simulation/", "tests/", "offline/")
    needle = "sensory.neuromorphic.snntorch_detector"
    offenders: list[str] = []
    for path in _REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel == "sensory/neuromorphic/snntorch_detector.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if needle not in text:
            continue
        if any(rel.startswith(p) for p in allowed_prefixes):
            continue
        offenders.append(rel)
    assert not offenders, f"snntorch_detector imported by: {offenders}"


# -------------------------------------------------------- state validation


def test_initial_state_rejects_bad_dim() -> None:
    with pytest.raises(SNNTorchDetectorError):
        initial_leaky_state(0)
    with pytest.raises(SNNTorchDetectorError):
        initial_leaky_state(snn.MAX_HIDDEN_DIM + 1)


def test_initial_state_rejects_nan_init() -> None:
    with pytest.raises(SNNTorchDetectorError):
        initial_leaky_state(2, v_init=float("nan"))


def test_state_rejects_nan_values() -> None:
    with pytest.raises(SNNTorchDetectorError):
        LeakyState(v=(0.0, float("nan")))
