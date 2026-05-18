"""C-43 — Tests for sensory/neuromorphic/nengo_cognitive.py.

Authority pins, NEF math, weight immutability, deterministic
splitmix64 build, NengoCognitiveAnalyser round-trip, INV-15
byte-identical replay.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from sensory.neuromorphic import nengo_cognitive as nc
from sensory.neuromorphic.nengo_cognitive import (
    ANALYSIS_SOURCE,
    MAX_DIMENSIONS,
    MAX_NEURONS,
    MAX_REGIME_LABEL_LEN,
    MAX_SOURCE_LEN,
    MAX_SYMBOL_LEN,
    MAX_WINDOW,
    MIN_DIMENSIONS,
    MIN_NEURONS,
    MIN_WINDOW,
    NENGO_COGNITIVE_VERSION,
    NEW_PIP_DEPENDENCIES,
    REGIME_LONG,
    REGIME_NEUTRAL,
    REGIME_SHORT,
    NengoCognitiveAnalyser,
    NengoCognitiveEngine,
    NengoCognitiveError,
    NengoEnsemble,
    NengoEnsembleConfig,
    NengoEnsembleState,
    NengoEnsembleWeights,
    NengoForwardCallable,
    NengoRegimePulse,
    build_random_ensemble_weights,
    initial_state,
    lif_step,
    nengo_cognitive_engine,
    pure_python_nengo_cognitive_engine,
)

MODULE_PATH = Path(nc.__file__)
MODULE_SOURCE = MODULE_PATH.read_text()
MODULE_AST = ast.parse(MODULE_SOURCE)


# ============================================================== authority


def test_authority_adapted_from_header() -> None:
    assert MODULE_SOURCE.startswith("# ADAPTED FROM: nengo/nengo")


def test_authority_pip_dependencies_nengo_numpy_only() -> None:
    assert NEW_PIP_DEPENDENCIES == ("nengo", "numpy")


def _iter_top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _iter_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_authority_no_top_level_vendor_imports() -> None:
    tops = _iter_top_level_imports(MODULE_AST)
    forbidden = {
        "nengo",
        "numpy",
        "scipy",
        "torch",
        "norse",
        "pandas",
        "polars",
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
        "threading",
        "queue",
        "os",
        "sys",
        "pathlib",
        "subprocess",
        "socket",
        "http",
        "urllib",
        "requests",
        "logging",
    }
    tops = _iter_top_level_imports(MODULE_AST)
    for name in tops:
        root = name.split(".")[0]
        assert root not in forbidden_roots, f"forbidden top-level runtime import: {name}"


def test_authority_no_engine_cross_imports() -> None:
    imports = _iter_imports(MODULE_AST)
    forbidden_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "registry.",
        "ui.",
    )
    for name in imports:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), f"forbidden cross-engine import: {name}"


def test_authority_vendor_imports_confined_to_factory() -> None:
    """``nengo`` + ``numpy`` may only appear inside
    :func:`nengo_cognitive_engine`."""

    for node in ast.walk(MODULE_AST):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import) and node.names:
                module = node.names[0].name
            if module is None:
                continue
            root = module.split(".")[0]
            if root not in {"nengo", "numpy"}:
                continue
            # Walk up to a function def; must be
            # nengo_cognitive_engine.
            found = False
            for parent in ast.walk(MODULE_AST):
                if not isinstance(parent, ast.FunctionDef):
                    continue
                if parent.name != "nengo_cognitive_engine":
                    continue
                for child in ast.walk(parent):
                    if child is node:
                        found = True
                        break
                if found:
                    break
            assert found, f"vendor import {module!r} not confined to nengo_cognitive_engine factory"


def test_authority_version_string_stable() -> None:
    assert NENGO_COGNITIVE_VERSION == "nengo-cognitive/v1"


def test_authority_analysis_source_constant() -> None:
    assert ANALYSIS_SOURCE == "sensory.neuromorphic.nengo_cognitive"


def test_authority_bounds_constants() -> None:
    assert MIN_DIMENSIONS == 1
    assert MAX_DIMENSIONS == 32
    assert MIN_NEURONS == 1
    assert MAX_NEURONS == 4_096
    assert MIN_WINDOW == 1
    assert MAX_WINDOW == 4_096
    assert MAX_REGIME_LABEL_LEN == 64
    assert MAX_SOURCE_LEN == 64
    assert MAX_SYMBOL_LEN == 64


def test_authority_regime_constants() -> None:
    assert REGIME_LONG == "LONG"
    assert REGIME_SHORT == "SHORT"
    assert REGIME_NEUTRAL == "NEUTRAL"


# ============================================================== config


def test_config_defaults_match_nengo() -> None:
    c = NengoEnsembleConfig()
    assert c.tau_rc == 0.02
    assert c.tau_ref == 0.002
    assert c.v_threshold == 1.0
    assert c.v_reset == 0.0
    assert c.v_leak == 0.0
    assert c.dt == 0.001


def test_config_is_frozen_slotted() -> None:
    c = NengoEnsembleConfig()
    assert "__slots__" in NengoEnsembleConfig.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.tau_rc = 0.5  # type: ignore[misc]


def test_config_rejects_non_positive_tau_rc() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(tau_rc=0.0)
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(tau_rc=-1.0)


def test_config_rejects_non_finite_fields() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(tau_rc=float("nan"))
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(tau_ref=float("inf"))
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(v_threshold=float("nan"))
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(v_reset=float("-inf"))
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(v_leak=float("nan"))
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(dt=float("nan"))


def test_config_rejects_negative_tau_ref() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(tau_ref=-0.001)


def test_config_rejects_non_positive_dt() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(dt=0.0)
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(dt=-0.001)


def test_config_rejects_dt_larger_than_tau_rc() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleConfig(tau_rc=0.01, dt=0.05)


# ============================================================== weights


def _trivial_weights(n_neurons: int = 4, dimensions: int = 2) -> NengoEnsembleWeights:
    return build_random_ensemble_weights(n_neurons=n_neurons, dimensions=dimensions, seed=42)


def test_weights_is_frozen_slotted() -> None:
    w = _trivial_weights()
    assert "__slots__" in NengoEnsembleWeights.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.n_neurons = 99  # type: ignore[misc]


def test_weights_digest_is_16_hex() -> None:
    w = _trivial_weights()
    digest = w.digest()
    assert len(digest) == 32
    assert all(c in "0123456789abcdef" for c in digest)


def test_weights_digest_deterministic() -> None:
    w1 = build_random_ensemble_weights(n_neurons=8, dimensions=3, seed=7)
    w2 = build_random_ensemble_weights(n_neurons=8, dimensions=3, seed=7)
    assert w1.digest() == w2.digest()


def test_weights_digest_changes_with_seed() -> None:
    w1 = build_random_ensemble_weights(n_neurons=8, dimensions=3, seed=1)
    w2 = build_random_ensemble_weights(n_neurons=8, dimensions=3, seed=2)
    assert w1.digest() != w2.digest()


def test_weights_digest_changes_with_dimensions() -> None:
    w1 = build_random_ensemble_weights(n_neurons=8, dimensions=2, seed=7)
    w2 = build_random_ensemble_weights(n_neurons=8, dimensions=3, seed=7)
    assert w1.digest() != w2.digest()


def test_weights_rejects_dimensions_below_min() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleWeights(
            encoders=(),
            gains=(),
            biases=(),
            decoders=(),
            n_neurons=4,
            dimensions=0,
            weights_seed=0,
        )


def test_weights_rejects_dimensions_above_max() -> None:
    with pytest.raises(NengoCognitiveError):
        build_random_ensemble_weights(n_neurons=4, dimensions=MAX_DIMENSIONS + 1, seed=0)


def test_weights_rejects_neurons_below_min() -> None:
    with pytest.raises(NengoCognitiveError):
        build_random_ensemble_weights(n_neurons=0, dimensions=2, seed=0)


def test_weights_rejects_neurons_above_max() -> None:
    with pytest.raises(NengoCognitiveError):
        build_random_ensemble_weights(n_neurons=MAX_NEURONS + 1, dimensions=2, seed=0)


def test_weights_rejects_negative_seed() -> None:
    with pytest.raises(NengoCognitiveError):
        build_random_ensemble_weights(n_neurons=4, dimensions=2, seed=-1)


def test_weights_rejects_non_finite_encoders() -> None:
    base = _trivial_weights()
    bad_encoders = tuple(
        (float("nan"),) * base.dimensions if i == 0 else base.encoders[i]
        for i in range(base.n_neurons)
    )
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleWeights(
            encoders=bad_encoders,
            gains=base.gains,
            biases=base.biases,
            decoders=base.decoders,
            n_neurons=base.n_neurons,
            dimensions=base.dimensions,
            weights_seed=base.weights_seed,
        )


def test_weights_rejects_non_positive_gain() -> None:
    base = _trivial_weights()
    bad_gains = (0.0,) + base.gains[1:]
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleWeights(
            encoders=base.encoders,
            gains=bad_gains,
            biases=base.biases,
            decoders=base.decoders,
            n_neurons=base.n_neurons,
            dimensions=base.dimensions,
            weights_seed=base.weights_seed,
        )


def test_weights_rejects_inconsistent_row_count() -> None:
    base = _trivial_weights()
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleWeights(
            encoders=base.encoders[:-1],
            gains=base.gains,
            biases=base.biases,
            decoders=base.decoders,
            n_neurons=base.n_neurons,
            dimensions=base.dimensions,
            weights_seed=base.weights_seed,
        )


def test_weights_rejects_inconsistent_row_width() -> None:
    base = _trivial_weights()
    bad_encoders = tuple((1.0,) if i == 0 else base.encoders[i] for i in range(base.n_neurons))
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleWeights(
            encoders=bad_encoders,
            gains=base.gains,
            biases=base.biases,
            decoders=base.decoders,
            n_neurons=base.n_neurons,
            dimensions=base.dimensions,
            weights_seed=base.weights_seed,
        )


def test_weights_encoder_rows_are_unit_norm() -> None:
    w = build_random_ensemble_weights(n_neurons=32, dimensions=3, seed=7)
    for row in w.encoders:
        norm = math.sqrt(sum(v * v for v in row))
        assert abs(norm - 1.0) < 1e-9


def test_weights_gains_positive() -> None:
    w = build_random_ensemble_weights(n_neurons=32, dimensions=3, seed=7)
    for g in w.gains:
        assert g > 0.0


def test_build_rejects_max_rate_non_positive() -> None:
    with pytest.raises(NengoCognitiveError):
        build_random_ensemble_weights(n_neurons=4, dimensions=2, seed=0, max_rate=0.0)


def test_build_rejects_inverted_intercept_bounds() -> None:
    with pytest.raises(NengoCognitiveError):
        build_random_ensemble_weights(
            n_neurons=4,
            dimensions=2,
            seed=0,
            intercept_low=0.5,
            intercept_high=0.5,
        )


# ============================================================== state


def test_initial_state_is_rest_potential() -> None:
    s = initial_state(8, v_leak=0.2)
    assert s.v == (0.2,) * 8


def test_initial_state_rejects_invalid_neuron_count() -> None:
    with pytest.raises(NengoCognitiveError):
        initial_state(0)
    with pytest.raises(NengoCognitiveError):
        initial_state(MAX_NEURONS + 1)


def test_initial_state_rejects_non_finite_v_leak() -> None:
    with pytest.raises(NengoCognitiveError):
        initial_state(4, v_leak=float("nan"))


def test_state_is_frozen_slotted() -> None:
    s = initial_state(4)
    assert "__slots__" in NengoEnsembleState.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.v = (1.0,) * 4  # type: ignore[misc]


def test_state_rejects_non_finite_v() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoEnsembleState(v=(float("nan"),))


# ============================================================== lif_step


def test_lif_step_subthreshold_no_spike() -> None:
    cfg = NengoEnsembleConfig()
    s = initial_state(2)
    s2, spikes = lif_step(s, (0.1, 0.05), cfg)
    assert spikes == (False, False)
    assert all(0.0 < v < cfg.v_threshold for v in s2.v)


def test_lif_step_above_threshold_spikes_and_resets() -> None:
    cfg = NengoEnsembleConfig(dt=0.02, tau_rc=0.02)
    # one step with dt == tau_rc, J = 50 gives v_next ≈ 50 → spike
    s = initial_state(1)
    s2, spikes = lif_step(s, (50.0,), cfg)
    assert spikes == (True,)
    assert s2.v == (cfg.v_reset,)


def test_lif_step_rejects_length_mismatch() -> None:
    cfg = NengoEnsembleConfig()
    s = initial_state(2)
    with pytest.raises(NengoCognitiveError):
        lif_step(s, (0.1,), cfg)


def test_lif_step_rejects_non_finite_current() -> None:
    cfg = NengoEnsembleConfig()
    s = initial_state(2)
    with pytest.raises(NengoCognitiveError):
        lif_step(s, (0.1, float("nan")), cfg)


# ============================================================== ensemble


def test_ensemble_forward_returns_three_tuple() -> None:
    w = _trivial_weights(n_neurons=4, dimensions=2)
    e = NengoEnsemble(weights=w)
    s = initial_state(w.n_neurons, v_leak=e.config.v_leak)
    result = e.forward(s, (0.5, 0.0))
    assert isinstance(result, tuple) and len(result) == 3
    next_state, spikes, decoded = result
    assert isinstance(next_state, NengoEnsembleState)
    assert isinstance(spikes, tuple) and len(spikes) == w.n_neurons
    assert isinstance(decoded, tuple) and len(decoded) == w.dimensions


def test_ensemble_forward_state_advances() -> None:
    w = _trivial_weights()
    e = NengoEnsemble(weights=w)
    s = initial_state(w.n_neurons, v_leak=e.config.v_leak)
    s2, _, _ = e.forward(s, (1.0, 0.0))
    assert s2 != s  # at least one neuron updated


def test_ensemble_forward_rejects_state_length_mismatch() -> None:
    w = _trivial_weights(n_neurons=4)
    e = NengoEnsemble(weights=w)
    s = initial_state(3, v_leak=e.config.v_leak)
    with pytest.raises(NengoCognitiveError):
        e.forward(s, (0.5, 0.0))


def test_ensemble_forward_rejects_input_length_mismatch() -> None:
    w = _trivial_weights(dimensions=2)
    e = NengoEnsemble(weights=w)
    s = initial_state(w.n_neurons, v_leak=e.config.v_leak)
    with pytest.raises(NengoCognitiveError):
        e.forward(s, (0.5,))


def test_ensemble_forward_rejects_non_finite_input() -> None:
    w = _trivial_weights()
    e = NengoEnsemble(weights=w)
    s = initial_state(w.n_neurons, v_leak=e.config.v_leak)
    with pytest.raises(NengoCognitiveError):
        e.forward(s, (float("nan"), 0.0))


# ============================================================== engine seam


def test_pure_python_engine_implements_protocol() -> None:
    engine = pure_python_nengo_cognitive_engine()
    assert isinstance(engine, NengoCognitiveEngine)


def test_pure_python_engine_run_window_basic() -> None:
    engine = pure_python_nengo_cognitive_engine()
    w = build_random_ensemble_weights(n_neurons=16, dimensions=2, seed=0)
    cfg = NengoEnsembleConfig()
    window = tuple((0.5, 0.0) for _ in range(10))
    spike_count, decoded_mean = engine.run_window(weights=w, config=cfg, window=window)
    assert isinstance(spike_count, int) and spike_count >= 0
    assert isinstance(decoded_mean, tuple) and len(decoded_mean) == 2
    for value in decoded_mean:
        assert math.isfinite(value)


def test_pure_python_engine_deterministic() -> None:
    engine = pure_python_nengo_cognitive_engine()
    w = build_random_ensemble_weights(n_neurons=16, dimensions=2, seed=0)
    cfg = NengoEnsembleConfig()
    window = tuple((0.5, 0.0) for _ in range(10))
    r1 = engine.run_window(weights=w, config=cfg, window=window)
    r2 = engine.run_window(weights=w, config=cfg, window=window)
    assert r1 == r2


def test_production_factory_raises_not_implemented_on_run() -> None:
    try:
        engine = nengo_cognitive_engine()
    except ImportError:
        pytest.skip("nengo not installed; production factory unavailable")
    w = build_random_ensemble_weights(n_neurons=4, dimensions=2, seed=0)
    cfg = NengoEnsembleConfig()
    window = ((0.0, 0.0),)
    with pytest.raises(NotImplementedError):
        engine.run_window(weights=w, config=cfg, window=window)


# ============================================================== fake engine


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeNengoEngine:
    """Deterministic test double for :class:`NengoCognitiveEngine`."""

    fixed_spike_count: int
    fixed_decoded_mean: tuple[float, ...]

    def run_window(
        self,
        *,
        weights: NengoEnsembleWeights,
        config: NengoEnsembleConfig,
        window: tuple[tuple[float, ...], ...],
    ) -> tuple[int, tuple[float, ...]]:
        return self.fixed_spike_count, self.fixed_decoded_mean


def test_fake_engine_is_protocol() -> None:
    e = _FakeNengoEngine(fixed_spike_count=3, fixed_decoded_mean=(0.0,))
    assert isinstance(e, NengoCognitiveEngine)


# ============================================================== pulse


def test_pulse_is_frozen_slotted() -> None:
    p = NengoRegimePulse(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BULL",
        polarity=REGIME_LONG,
        confidence=0.5,
        decoded_value=(0.5,),
        spike_count=10,
        sample_count=5,
        weights_digest="0" * 32,
        evidence={"analyser": ANALYSIS_SOURCE},
    )
    assert "__slots__" in NengoRegimePulse.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.confidence = 0.9  # type: ignore[misc]


def test_pulse_rejects_negative_ts_ns() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=-1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_empty_source() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_empty_symbol() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_empty_regime_label() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_over_long_strings() -> None:
    big = "X" * 65
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source=big,
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol=big,
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label=big,
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_bad_polarity() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity="UP",
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_out_of_range_confidence() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=1.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=float("nan"),
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_non_finite_decoded_value() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(float("inf"),),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_bad_decoded_value_length() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(),
            spike_count=10,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_negative_spike_count() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=-1,
            sample_count=5,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_sample_count_below_one() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=0,
            weights_digest="0" * 32,
        )


def test_pulse_rejects_non_hex_digest() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="zz" + "0" * 30,
        )


def test_pulse_rejects_wrong_length_digest() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoRegimePulse(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            polarity=REGIME_LONG,
            confidence=0.5,
            decoded_value=(0.5,),
            spike_count=10,
            sample_count=5,
            weights_digest="abc",
        )


# ============================================================== analyser


def _build_analyser_with_fake(
    *, spike_count: int = 5, decoded_mean: tuple[float, ...] = (0.5,)
) -> NengoCognitiveAnalyser:
    return NengoCognitiveAnalyser(
        engine=_FakeNengoEngine(
            fixed_spike_count=spike_count,
            fixed_decoded_mean=decoded_mean,
        ),
        confidence_threshold=0.1,
    )


def test_analyser_is_frozen_slotted() -> None:
    a = _build_analyser_with_fake()
    assert "__slots__" in NengoCognitiveAnalyser.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.confidence_threshold = 0.9  # type: ignore[misc]


def test_analyser_rejects_invalid_threshold() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoCognitiveAnalyser(
            engine=_FakeNengoEngine(0, (0.0,)),
            confidence_threshold=0.0,
        )
    with pytest.raises(NengoCognitiveError):
        NengoCognitiveAnalyser(
            engine=_FakeNengoEngine(0, (0.0,)),
            confidence_threshold=1.5,
        )
    with pytest.raises(NengoCognitiveError):
        NengoCognitiveAnalyser(
            engine=_FakeNengoEngine(0, (0.0,)),
            confidence_threshold=float("nan"),
        )


def test_analyser_rejects_non_protocol_engine() -> None:
    with pytest.raises(NengoCognitiveError):
        NengoCognitiveAnalyser(engine="not-an-engine")  # type: ignore[arg-type]


def test_analyser_detect_returns_pulse() -> None:
    a = _build_analyser_with_fake(spike_count=12, decoded_mean=(0.7,))
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    window = ((0.1,), (0.2,), (0.3,))
    p = a.detect(
        ts_ns=42,
        source="BINANCE",
        symbol="BTCUSDT",
        regime_label="BULL",
        weights=w,
        window=window,
    )
    assert p.ts_ns == 42
    assert p.source == "BINANCE"
    assert p.symbol == "BTCUSDT"
    assert p.regime_label == "BULL"
    assert p.polarity == REGIME_LONG
    assert math.isclose(p.confidence, 0.7)
    assert p.decoded_value == (0.7,)
    assert p.spike_count == 12
    assert p.sample_count == 3
    assert p.weights_digest == w.digest()
    assert p.evidence["analyser"] == ANALYSIS_SOURCE


def test_analyser_detect_neutral_below_threshold() -> None:
    a = _build_analyser_with_fake(spike_count=1, decoded_mean=(0.05,))
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    window = ((0.1,),)
    p = a.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="REGIME_TEST",
        weights=w,
        window=window,
    )
    assert p.polarity == REGIME_NEUTRAL


def test_analyser_detect_polarity_long() -> None:
    a = _build_analyser_with_fake(spike_count=10, decoded_mean=(0.6,))
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    p = a.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BULL",
        weights=w,
        window=((0.5,),),
        polarity_sign=1,
    )
    assert p.polarity == REGIME_LONG


def test_analyser_detect_polarity_short() -> None:
    a = _build_analyser_with_fake(spike_count=10, decoded_mean=(-0.6,))
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    p = a.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BEAR",
        weights=w,
        window=((0.5,),),
        polarity_sign=1,
    )
    assert p.polarity == REGIME_SHORT


def test_analyser_detect_polarity_sign_flip() -> None:
    a = _build_analyser_with_fake(spike_count=10, decoded_mean=(0.6,))
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    p_pos = a.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BULL",
        weights=w,
        window=((0.5,),),
        polarity_sign=1,
    )
    p_neg = a.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BULL",
        weights=w,
        window=((0.5,),),
        polarity_sign=-1,
    )
    assert p_pos.polarity == REGIME_LONG
    assert p_neg.polarity == REGIME_SHORT


def test_analyser_detect_polarity_sign_zero_forces_neutral() -> None:
    a = _build_analyser_with_fake(spike_count=10, decoded_mean=(0.9,))
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    p = a.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BULL",
        weights=w,
        window=((0.5,),),
        polarity_sign=0,
    )
    assert p.polarity == REGIME_NEUTRAL


def test_analyser_detect_polarity_axis() -> None:
    a = NengoCognitiveAnalyser(
        engine=_FakeNengoEngine(
            fixed_spike_count=10,
            fixed_decoded_mean=(0.0, -0.6),
        ),
        confidence_threshold=0.1,
    )
    w = build_random_ensemble_weights(n_neurons=4, dimensions=2, seed=0)
    p = a.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BEAR",
        weights=w,
        window=((0.5, 0.0),),
        polarity_axis=1,
    )
    assert p.polarity == REGIME_SHORT


def test_analyser_detect_polarity_axis_out_of_range() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=((0.5,),),
            polarity_axis=99,
        )


def test_analyser_detect_polarity_sign_invalid() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=((0.5,),),
            polarity_sign=2,
        )


def test_analyser_detect_rejects_invalid_ts_ns() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=-1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=((0.5,),),
        )


def test_analyser_detect_rejects_empty_source() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=((0.5,),),
        )


def test_analyser_detect_rejects_empty_symbol() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="",
            regime_label="BULL",
            weights=w,
            window=((0.5,),),
        )


def test_analyser_detect_rejects_empty_regime_label() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="",
            weights=w,
            window=((0.5,),),
        )


def test_analyser_detect_rejects_non_weights_object() -> None:
    a = _build_analyser_with_fake()
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights="not-weights",  # type: ignore[arg-type]
            window=((0.5,),),
        )


def test_analyser_detect_rejects_non_finite_window_entry() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=((float("nan"),),),
        )


def test_analyser_detect_rejects_row_dimension_mismatch() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=2, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=((0.5,),),
        )


def test_analyser_detect_rejects_empty_window() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=(),
        )


def test_analyser_detect_rejects_oversized_window() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    big = tuple((0.0,) for _ in range(MAX_WINDOW + 1))
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=big,
        )


def test_analyser_detect_evidence_carries_analyser_tag() -> None:
    a = _build_analyser_with_fake()
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    p = a.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BULL",
        weights=w,
        window=((0.5,),),
        evidence={"k": "v"},
    )
    assert p.evidence["analyser"] == ANALYSIS_SOURCE
    assert p.evidence["k"] == "v"


def test_analyser_detect_confidence_clipped() -> None:
    a = _build_analyser_with_fake(spike_count=10, decoded_mean=(2.0,))
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    p = a.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BULL",
        weights=w,
        window=((0.5,),),
    )
    assert p.confidence == 1.0


def test_analyser_detect_rejects_engine_returning_wrong_decoded_length() -> None:
    a = NengoCognitiveAnalyser(
        engine=_FakeNengoEngine(
            fixed_spike_count=5,
            fixed_decoded_mean=(0.5, 0.5),
        ),
        confidence_threshold=0.1,
    )
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=((0.5,),),
        )


def test_analyser_detect_rejects_engine_returning_negative_spike_count() -> None:
    a = NengoCognitiveAnalyser(
        engine=_FakeNengoEngine(
            fixed_spike_count=-1,
            fixed_decoded_mean=(0.5,),
        ),
        confidence_threshold=0.1,
    )
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=((0.5,),),
        )


def test_analyser_detect_rejects_engine_returning_non_finite_decoded() -> None:
    a = NengoCognitiveAnalyser(
        engine=_FakeNengoEngine(
            fixed_spike_count=5,
            fixed_decoded_mean=(float("nan"),),
        ),
        confidence_threshold=0.1,
    )
    w = build_random_ensemble_weights(n_neurons=4, dimensions=1, seed=0)
    with pytest.raises(NengoCognitiveError):
        a.detect(
            ts_ns=1,
            source="X",
            symbol="BTC",
            regime_label="BULL",
            weights=w,
            window=((0.5,),),
        )


# ============================================================== INV-15 replay


def test_analyser_inv15_byte_identical_replay() -> None:
    """3-run identical-input replay equality (INV-15)."""

    engine = pure_python_nengo_cognitive_engine()
    analyser = NengoCognitiveAnalyser(engine=engine, confidence_threshold=0.01)
    weights = build_random_ensemble_weights(n_neurons=24, dimensions=2, seed=314159)
    window = tuple((math.sin(0.1 * i), math.cos(0.1 * i)) for i in range(50))
    pulses = [
        analyser.detect(
            ts_ns=987654321,
            source="BINANCE",
            symbol="BTCUSDT",
            regime_label="OSCILLATION",
            weights=weights,
            window=window,
        )
        for _ in range(3)
    ]
    assert pulses[0] == pulses[1] == pulses[2]


def test_pure_python_pipeline_discriminates_bull_vs_bear() -> None:
    """End-to-end: pure-python engine + analyser discriminate
    polarity from input vector sign."""

    engine = pure_python_nengo_cognitive_engine()
    analyser = NengoCognitiveAnalyser(engine=engine, confidence_threshold=0.005)
    weights = build_random_ensemble_weights(n_neurons=64, dimensions=2, seed=42)
    bull = tuple((1.0, 0.0) for _ in range(200))
    bear = tuple((-1.0, 0.0) for _ in range(200))
    p_bull = analyser.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BULL_TREND",
        weights=weights,
        window=bull,
    )
    p_bear = analyser.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="BEAR_TREND",
        weights=weights,
        window=bear,
    )
    assert p_bull.polarity == REGIME_LONG
    assert p_bear.polarity == REGIME_SHORT
    # Decoded values should have opposite sign on axis 0.
    assert p_bull.decoded_value[0] > 0.0
    assert p_bear.decoded_value[0] < 0.0


def test_pure_python_pipeline_neutral_input() -> None:
    """Neutral input -> NEUTRAL polarity."""

    engine = pure_python_nengo_cognitive_engine()
    analyser = NengoCognitiveAnalyser(engine=engine, confidence_threshold=0.05)
    weights = build_random_ensemble_weights(n_neurons=64, dimensions=2, seed=42)
    window = tuple((0.0, 0.0) for _ in range(50))
    p = analyser.detect(
        ts_ns=1,
        source="X",
        symbol="BTC",
        regime_label="REGIME_NONE",
        weights=weights,
        window=window,
    )
    assert p.polarity == REGIME_NEUTRAL


# ============================================================== protocol shape


def test_forward_callable_protocol_runtime_checkable() -> None:
    """``NengoForwardCallable`` is a runtime-checkable Protocol."""

    class _Forward:
        def forward(
            self,
            state: NengoEnsembleState,
            x,
        ) -> tuple[NengoEnsembleState, tuple[bool, ...], tuple[float, ...]]:
            return state, (False,), (0.0,)

    assert isinstance(_Forward(), NengoForwardCallable)


def test_engine_protocol_runtime_checkable() -> None:
    """``NengoCognitiveEngine`` is a runtime-checkable Protocol."""

    class _Engine:
        def run_window(self, *, weights, config, window):
            return 0, tuple(0.0 for _ in range(weights.dimensions))

    assert isinstance(_Engine(), NengoCognitiveEngine)


# ============================================================== __all__


def test_all_exports_stable() -> None:
    expected = {
        "ANALYSIS_SOURCE",
        "MAX_DIMENSIONS",
        "MAX_NEURONS",
        "MAX_REGIME_LABEL_LEN",
        "MAX_SOURCE_LEN",
        "MAX_SYMBOL_LEN",
        "MAX_WINDOW",
        "MIN_DIMENSIONS",
        "MIN_NEURONS",
        "MIN_WINDOW",
        "NENGO_COGNITIVE_VERSION",
        "NEW_PIP_DEPENDENCIES",
        "REGIME_LONG",
        "REGIME_NEUTRAL",
        "REGIME_SHORT",
        "NengoCognitiveAnalyser",
        "NengoCognitiveEngine",
        "NengoCognitiveError",
        "NengoEnsemble",
        "NengoEnsembleConfig",
        "NengoEnsembleState",
        "NengoEnsembleWeights",
        "NengoForwardCallable",
        "NengoRegimePulse",
        "build_random_ensemble_weights",
        "initial_state",
        "lif_step",
        "nengo_cognitive_engine",
        "pure_python_nengo_cognitive_engine",
    }
    assert set(nc.__all__) == expected
