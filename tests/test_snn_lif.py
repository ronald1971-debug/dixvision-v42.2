"""B-14 — Tests for sensory/neuromorphic/snn_lif.py.

Authority pins, LIF math, weight immutability, Poisson encoder
determinism, SNNDetector round-trip, INV-15 byte-identical replay.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from sensory.neuromorphic import snn_lif
from sensory.neuromorphic.snn_lif import (
    MAX_HIDDEN_DIM,
    MAX_INPUT_DIM,
    MAX_WINDOW,
    NEW_PIP_DEPENDENCIES,
    SNN_LIF_VERSION,
    LIFCell,
    LIFConfig,
    LIFState,
    LIFWeights,
    SNNDetector,
    SNNLIFError,
    SpikePulse,
    identity_weights,
    initial_state,
    lif_feed_forward_step,
    poisson_encode,
    torch_lif_cell_factory,
)

MODULE_PATH = Path(snn_lif.__file__)
MODULE_SOURCE = MODULE_PATH.read_text()
MODULE_AST = ast.parse(MODULE_SOURCE)


# ================================================================== authority pins


def test_authority_adapted_from_header() -> None:
    assert MODULE_SOURCE.startswith("# ADAPTED FROM: norse/norse")


def test_authority_pip_dependencies_torch_only() -> None:
    assert NEW_PIP_DEPENDENCIES == ("torch",)


def _iter_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _iter_top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_authority_no_top_level_torch_or_norse() -> None:
    tops = _iter_top_level_imports(MODULE_AST)
    forbidden = {
        "torch",
        "norse",
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "bindsnet",
        "brian2",
        "snntorch",
        "tensorboard",
        "wandb",
        "mlflow",
    }
    for name in tops:
        root = name.split(".")[0]
        assert root not in forbidden, f"forbidden top-level import: {name}"


def test_authority_no_runtime_imports() -> None:
    forbidden_roots = {
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "socket",
        "secrets",
        "uuid",
        "requests",
        "httpx",
        "aiohttp",
        "websockets",
    }
    for name in _iter_imports(MODULE_AST):
        root = name.split(".")[0]
        assert root not in forbidden_roots, f"forbidden import for sensory tier: {name}"


def test_authority_no_engine_cross_imports() -> None:
    forbidden_roots = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "evolution_engine",
        "learning_engine",
    }
    for name in _iter_imports(MODULE_AST):
        root = name.split(".")[0]
        assert root not in forbidden_roots, f"forbidden engine cross-import: {name}"


def test_authority_no_typed_event_construction() -> None:
    """Sensor MUST NOT construct typed bus events (INV-19)."""

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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_types, (
                f"forbidden typed-event construction: {node.func.id}"
            )


def test_authority_no_core_events_import() -> None:
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "core.contracts.events", (
                "sensor must not import core.contracts.events"
            )


def test_module_has_no_module_state() -> None:
    """Module-level state must be only constants (no mutable globals)."""

    for name in dir(snn_lif):
        if name.startswith("_"):
            continue
        attr = getattr(snn_lif, name)
        # Functions, classes, frozen dataclasses, tuples, ints, strs allowed
        assert not isinstance(attr, (list, dict, set)), (
            f"snn_lif.{name} is mutable container ({type(attr).__name__})"
        )


# ================================================================== freezing


def test_lif_config_frozen() -> None:
    cfg = LIFConfig()
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.tau_mem = 0.5  # type: ignore[misc]


def test_lif_weights_frozen() -> None:
    w = identity_weights(2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.bias = (0.0, 0.0)  # type: ignore[misc]


def test_lif_state_frozen() -> None:
    s = LIFState(v=(0.0, 0.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.v = (1.0, 1.0)  # type: ignore[misc]


def test_lif_cell_frozen() -> None:
    cell = LIFCell(weights=identity_weights(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cell.weights = identity_weights(3)  # type: ignore[misc]


def test_spike_pulse_frozen() -> None:
    pulse = SpikePulse(
        ts_ns=1,
        source="SRC",
        symbol="SYM",
        polarity="NEUTRAL",
        intensity=0.0,
        spike_count=0,
        sample_count=1,
        weights_digest="0" * 32,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        pulse.intensity = 1.0  # type: ignore[misc]


# ================================================================== config validation


def test_config_rejects_nonpositive_tau() -> None:
    with pytest.raises(SNNLIFError):
        LIFConfig(tau_mem=0.0)


def test_config_rejects_nonpositive_dt() -> None:
    with pytest.raises(SNNLIFError):
        LIFConfig(dt=-1.0e-3)


def test_config_rejects_dt_greater_than_tau() -> None:
    with pytest.raises(SNNLIFError):
        LIFConfig(tau_mem=1e-3, dt=1e-2)


def test_config_rejects_nan_threshold() -> None:
    with pytest.raises(SNNLIFError):
        LIFConfig(v_threshold=float("nan"))


# ================================================================== weights validation


def test_weights_reject_bad_dims() -> None:
    with pytest.raises(SNNLIFError):
        identity_weights(0)
    with pytest.raises(SNNLIFError):
        identity_weights(MAX_HIDDEN_DIM + 1)


def test_weights_reject_dim_mismatch() -> None:
    with pytest.raises(SNNLIFError):
        LIFWeights(
            weight=((1.0,),),
            bias=(0.0, 0.0),
            input_dim=1,
            hidden_dim=2,
        )


def test_weights_reject_inf_entries() -> None:
    with pytest.raises(SNNLIFError):
        LIFWeights(
            weight=((float("inf"),),),
            bias=(0.0,),
            input_dim=1,
            hidden_dim=1,
        )


def test_weights_reject_nan_bias() -> None:
    with pytest.raises(SNNLIFError):
        LIFWeights(
            weight=((1.0,),),
            bias=(float("nan"),),
            input_dim=1,
            hidden_dim=1,
        )


def test_weights_digest_deterministic() -> None:
    w1 = identity_weights(3)
    w2 = identity_weights(3)
    assert w1.digest() == w2.digest()


def test_weights_digest_distinct_for_different_dims() -> None:
    assert identity_weights(2).digest() != identity_weights(3).digest()


# ================================================================== LIF step math


def test_lif_step_below_threshold_no_spike() -> None:
    cfg = LIFConfig(tau_mem=1e-2, dt=1e-3, v_threshold=1.0)
    state = LIFState(v=(0.0,))
    next_state, spikes = lif_feed_forward_step(state, (0.1,), cfg)
    assert spikes == (False,)
    # v_next = 0 + 0.1 * (0 - 0 + 0.1) = 0.01
    assert next_state.v[0] == pytest.approx(0.01)


def test_lif_step_above_threshold_spikes_and_resets() -> None:
    cfg = LIFConfig(
        tau_mem=1e-2,
        dt=1e-3,
        v_threshold=0.5,
        v_reset=0.0,
    )
    state = LIFState(v=(0.49,))
    # decay = 0.1; v_after = 0.49 + 0.1 * (0 - 0.49 + 100.0) = 0.49 + 9.951 = 10.441 -> spike
    next_state, spikes = lif_feed_forward_step(state, (100.0,), cfg)
    assert spikes == (True,)
    assert next_state.v[0] == 0.0


def test_lif_step_leak_toward_v_leak() -> None:
    cfg = LIFConfig(
        tau_mem=1e-2,
        dt=1e-3,
        v_threshold=10.0,
        v_leak=0.0,
    )
    state = LIFState(v=(1.0,))
    next_state, spikes = lif_feed_forward_step(state, (0.0,), cfg)
    assert spikes == (False,)
    # v_next = 1 + 0.1*(0 - 1 + 0) = 0.9
    assert next_state.v[0] == pytest.approx(0.9)


def test_lif_step_rejects_length_mismatch() -> None:
    cfg = LIFConfig()
    state = LIFState(v=(0.0, 0.0))
    with pytest.raises(SNNLIFError):
        lif_feed_forward_step(state, (0.1,), cfg)


def test_lif_step_rejects_nan_input() -> None:
    cfg = LIFConfig()
    state = LIFState(v=(0.0,))
    with pytest.raises(SNNLIFError):
        lif_feed_forward_step(state, (float("nan"),), cfg)


def test_lif_step_multineuron_independent() -> None:
    cfg = LIFConfig(v_threshold=0.5, v_reset=0.0)
    state = LIFState(v=(0.49, 0.0))
    next_state, spikes = lif_feed_forward_step(state, (100.0, 0.0), cfg)
    assert spikes[0] is True and spikes[1] is False
    assert next_state.v[0] == 0.0
    # second neuron leaks: v = 0 + 0.1 * (0 - 0 + 0) = 0
    assert next_state.v[1] == 0.0


# ================================================================== LIFCell


def test_lif_cell_forward_identity_pass_through() -> None:
    cfg = LIFConfig(v_threshold=0.5, v_reset=0.0)
    cell = LIFCell(weights=identity_weights(2), config=cfg)
    state = initial_state(2)
    next_state, spikes = cell.forward(state, (100.0, 0.0))
    assert spikes == (True, False)
    assert next_state.v[0] == 0.0


def test_lif_cell_rejects_input_dim_mismatch() -> None:
    cell = LIFCell(weights=identity_weights(2))
    state = initial_state(2)
    with pytest.raises(SNNLIFError):
        cell.forward(state, (1.0,))


def test_lif_cell_rejects_state_hidden_dim_mismatch() -> None:
    cell = LIFCell(weights=identity_weights(2))
    bad_state = LIFState(v=(0.0,))
    with pytest.raises(SNNLIFError):
        cell.forward(bad_state, (0.0, 0.0))


# ================================================================== Poisson encoder


def test_poisson_encode_deterministic() -> None:
    a = poisson_encode((10.0, 5.0), n_steps=20, dt=1e-3, seed=42)
    b = poisson_encode((10.0, 5.0), n_steps=20, dt=1e-3, seed=42)
    assert a == b


def test_poisson_encode_seed_changes_train() -> None:
    a = poisson_encode((500.0, 500.0), n_steps=64, dt=1e-3, seed=42)
    b = poisson_encode((500.0, 500.0), n_steps=64, dt=1e-3, seed=99)
    assert a != b


def test_poisson_encode_zero_rate_never_fires() -> None:
    train = poisson_encode((0.0, 0.0), n_steps=50, dt=1e-3, seed=42)
    flat = [s for row in train for s in row]
    assert not any(flat)


def test_poisson_encode_high_rate_fires_often() -> None:
    train = poisson_encode((900.0,), n_steps=200, dt=1e-3, seed=42)
    fires = sum(1 for row in train for s in row if s)
    # p = 900 * 1e-3 = 0.9 -> expect ~180 spikes over 200 steps
    assert fires > 130


def test_poisson_encode_rejects_bad_dt() -> None:
    with pytest.raises(SNNLIFError):
        poisson_encode((1.0,), n_steps=10, dt=0.0, seed=0)


def test_poisson_encode_rejects_bad_seed() -> None:
    with pytest.raises(SNNLIFError):
        poisson_encode((1.0,), n_steps=10, dt=1e-3, seed=-1)


def test_poisson_encode_rejects_negative_rate() -> None:
    with pytest.raises(SNNLIFError):
        poisson_encode((-1.0,), n_steps=10, dt=1e-3, seed=0)


def test_poisson_encode_rejects_oversize_window() -> None:
    with pytest.raises(SNNLIFError):
        poisson_encode((1.0,), n_steps=MAX_WINDOW + 1, dt=1e-3, seed=0)


def test_poisson_encode_rejects_oversize_channels() -> None:
    with pytest.raises(SNNLIFError):
        poisson_encode(
            tuple(1.0 for _ in range(MAX_INPUT_DIM + 1)),
            n_steps=1,
            dt=1e-3,
            seed=0,
        )


# ================================================================== SNNDetector


def _detector(dim: int = 2) -> SNNDetector:
    cell = LIFCell(
        weights=identity_weights(dim),
        config=LIFConfig(v_threshold=0.5, v_reset=0.0),
    )
    return SNNDetector(cell=cell, spike_polarity_threshold=0.5)


def test_detector_emits_spike_pulse() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1_000,
        source="BINANCE",
        symbol="BTCUSDT",
        window=[(100.0, 0.0), (100.0, 0.0)],
    )
    assert isinstance(pulse, SpikePulse)
    assert pulse.sample_count == 2
    assert pulse.spike_count >= 2  # both steps spike on channel 0


def test_detector_neutral_pulse_below_threshold() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1,
        source="SRC",
        symbol="SYM",
        window=[(0.0, 0.0)] * 4,
    )
    assert pulse.polarity == "NEUTRAL"
    assert pulse.spike_count == 0
    assert pulse.intensity == 0.0


def test_detector_long_polarity_when_active() -> None:
    det = _detector(dim=1)
    pulse = det.detect(
        ts_ns=1,
        source="SRC",
        symbol="SYM",
        window=[(100.0,)] * 4,
    )
    assert pulse.polarity == "LONG"
    assert pulse.intensity == 1.0


def test_detector_short_polarity_with_negative_sign() -> None:
    det = _detector(dim=1)
    pulse = det.detect(
        ts_ns=1,
        source="SRC",
        symbol="SYM",
        window=[(100.0,)] * 4,
        polarity_sign=-1,
    )
    assert pulse.polarity == "SHORT"


def test_detector_neutral_when_sign_zero() -> None:
    det = _detector(dim=1)
    pulse = det.detect(
        ts_ns=1,
        source="SRC",
        symbol="SYM",
        window=[(100.0,)] * 4,
        polarity_sign=0,
    )
    assert pulse.polarity == "NEUTRAL"


def test_detector_rejects_negative_ts() -> None:
    det = _detector()
    with pytest.raises(SNNLIFError):
        det.detect(
            ts_ns=-1,
            source="SRC",
            symbol="SYM",
            window=[(0.0, 0.0)],
        )


def test_detector_rejects_empty_source() -> None:
    det = _detector()
    with pytest.raises(SNNLIFError):
        det.detect(
            ts_ns=0,
            source="",
            symbol="SYM",
            window=[(0.0, 0.0)],
        )


def test_detector_rejects_empty_symbol() -> None:
    det = _detector()
    with pytest.raises(SNNLIFError):
        det.detect(
            ts_ns=0,
            source="SRC",
            symbol="",
            window=[(0.0, 0.0)],
        )


def test_detector_rejects_bad_polarity_sign() -> None:
    det = _detector()
    with pytest.raises(SNNLIFError):
        det.detect(
            ts_ns=0,
            source="SRC",
            symbol="SYM",
            window=[(0.0, 0.0)],
            polarity_sign=2,
        )


def test_detector_rejects_oversize_window() -> None:
    det = _detector(dim=1)
    with pytest.raises(SNNLIFError):
        det.detect(
            ts_ns=0,
            source="SRC",
            symbol="SYM",
            window=[(0.0,) for _ in range(MAX_WINDOW + 1)],
        )


def test_detector_rejects_bad_polarity_threshold() -> None:
    cell = LIFCell(weights=identity_weights(1))
    with pytest.raises(SNNLIFError):
        SNNDetector(cell=cell, spike_polarity_threshold=0.0)
    with pytest.raises(SNNLIFError):
        SNNDetector(cell=cell, spike_polarity_threshold=1.5)


def test_detector_records_weights_digest() -> None:
    det = _detector()
    pulse = det.detect(
        ts_ns=1,
        source="SRC",
        symbol="SYM",
        window=[(0.0, 0.0)],
    )
    assert pulse.weights_digest == det.cell.weights.digest()
    assert len(pulse.weights_digest) == 32


# ================================================================== INV-15 replay


def test_replay_byte_identical_three_runs() -> None:
    det = _detector()
    window = [(100.0, 0.0), (50.0, 25.0), (10.0, 90.0)]

    def run() -> SpikePulse:
        return det.detect(
            ts_ns=12345,
            source="BINANCE",
            symbol="BTCUSDT",
            window=window,
        )

    a, b, c = run(), run(), run()
    assert a == b == c


def test_replay_ts_change_changes_pulse() -> None:
    det = _detector()
    window = [(100.0, 0.0)]
    p1 = det.detect(ts_ns=1, source="S", symbol="X", window=window)
    p2 = det.detect(ts_ns=2, source="S", symbol="X", window=window)
    assert p1.ts_ns != p2.ts_ns


def test_replay_input_change_changes_intensity() -> None:
    det = _detector()
    p1 = det.detect(
        ts_ns=1,
        source="S",
        symbol="X",
        window=[(100.0, 0.0)],
    )
    p2 = det.detect(
        ts_ns=1,
        source="S",
        symbol="X",
        window=[(0.0, 0.0)],
    )
    assert p1.intensity != p2.intensity


# ================================================================== factory


def test_torch_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        torch_lif_cell_factory()


def test_version_constant() -> None:
    assert SNN_LIF_VERSION.startswith("snn-lif/")
