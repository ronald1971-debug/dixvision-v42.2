"""B-19 — Tests for the SpykeTorch-adapted spike train encoder."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from sensory.neuromorphic import spyke_encoder as enc
from sensory.neuromorphic.spyke_encoder import (
    NEW_PIP_DEPENDENCIES,
    SPYKE_ENCODER_VERSION,
    EncodingMethod,
    SpikeEvent,
    SpikeTrain,
    SpykeBackend,
    SpykeEncoderError,
    encode,
    rate_encode,
    spike_train_to_step_inputs,
    spyketorch_intensity_to_latency_factory,
    temporal_encode,
)

_MODULE_PATH = Path(enc.__file__)
_MODULE_TEXT = _MODULE_PATH.read_text(encoding="utf-8")
_MODULE_AST = ast.parse(_MODULE_TEXT)


# ---------------------------------------------------------------------------
# Authority + adapted-from pins
# ---------------------------------------------------------------------------


def test_authority_adapted_from_header() -> None:
    assert _MODULE_TEXT.startswith("# ADAPTED FROM: SpykeTorch/snn.py")


def test_authority_module_version_constant() -> None:
    assert SPYKE_ENCODER_VERSION == "spyke-encoder/v1"


def test_authority_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("spyketorch",)


def test_authority_no_top_level_spyketorch_import() -> None:
    """SpykeTorch may only be referenced inside the production factory."""
    for node in ast.iter_child_nodes(_MODULE_AST):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            assert "spyketorch" not in text.lower(), f"top-level spyketorch import: {text}"
            assert "SpykeTorch" not in text, f"top-level SpykeTorch import: {text}"


def test_authority_no_runtime_random_clock_io_imports() -> None:
    forbidden = {
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "numpy",
        "torch",
        "polars",
        "pandas",
        "langsmith",
    }
    for node in ast.iter_child_nodes(_MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            assert root not in forbidden, f"forbidden from-import: {node.module}"


def test_authority_no_engine_cross_imports() -> None:
    forbidden_prefixes = (
        "governance_engine",
        "system_engine",
        "execution_engine",
        "intelligence_engine",
        "evolution_engine",
        "registry",
        "dashboard_backend",
    )
    for node in ast.walk(_MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_prefixes:
                    assert not alias.name.startswith(prefix), (
                        f"forbidden engine import: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for prefix in forbidden_prefixes:
                assert not node.module.startswith(prefix), (
                    f"forbidden engine from-import: {node.module}"
                )


def test_authority_no_typed_event_construction() -> None:
    """B27 / B28 / INV-71: sensory tier emits advisory records only."""
    forbidden_calls = {
        "HazardEvent",
        "SignalEvent",
        "PatchProposal",
        "GovernanceDecision",
        "LearningUpdate",
    }
    for node in ast.walk(_MODULE_AST):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in forbidden_calls:
                raise AssertionError(f"sensory tier may not construct typed event: {name}()")


def test_authority_no_production_tier_imports_spyke_encoder() -> None:
    """Importer scan: no production tier may import the encoder."""
    repo_root = Path(__file__).resolve().parent.parent
    production_dirs = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "registry",
        "dashboard_backend",
    )
    for sub in production_dirs:
        directory = repo_root / sub
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "spyke_encoder" not in text, (
                f"production tier {path} must not import spyke_encoder"
            )


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def test_spike_event_is_frozen() -> None:
    ev = SpikeEvent(neuron_idx=0, time_step=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.neuron_idx = 99  # type: ignore[misc]


def test_spike_train_is_frozen() -> None:
    train = temporal_encode([1.0, 0.5], num_steps=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        train.num_neurons = 99  # type: ignore[misc]


def test_spike_train_has_slots_no_dict() -> None:
    train = temporal_encode([1.0, 0.5], num_steps=4)
    assert not hasattr(train, "__dict__")


# ---------------------------------------------------------------------------
# SpikeEvent validation
# ---------------------------------------------------------------------------


def test_spike_event_rejects_negative_neuron_idx() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeEvent(neuron_idx=-1, time_step=0)


def test_spike_event_rejects_negative_time_step() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeEvent(neuron_idx=0, time_step=-1)


def test_spike_event_rejects_non_int() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeEvent(neuron_idx=0.5, time_step=0)  # type: ignore[arg-type]


def test_spike_event_rejects_bool() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeEvent(neuron_idx=True, time_step=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Feature-vector validation
# ---------------------------------------------------------------------------


def test_features_must_be_list_or_tuple() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode("not a vector", num_steps=4)  # type: ignore[arg-type]


def test_features_must_be_non_empty() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode([], num_steps=4)


def test_features_reject_nan() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode([float("nan")], num_steps=4)


def test_features_reject_inf() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode([float("inf")], num_steps=4)


def test_features_reject_negative() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode([-0.1], num_steps=4)


def test_features_reject_above_one() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode([1.1], num_steps=4)


def test_features_reject_bool() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode([True, 0.5], num_steps=4)  # type: ignore[list-item]


def test_features_reject_string() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode(["x"], num_steps=4)  # type: ignore[list-item]


def test_num_steps_must_be_positive() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode([0.5], num_steps=0)


def test_num_steps_rejects_bool() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode([0.5], num_steps=True)  # type: ignore[arg-type]


def test_num_steps_rejects_huge() -> None:
    with pytest.raises(SpykeEncoderError):
        temporal_encode([0.5], num_steps=10_000_000)


def test_seed_rejects_negative() -> None:
    with pytest.raises(SpykeEncoderError):
        rate_encode([0.5], num_steps=4, seed=-1)


def test_seed_rejects_too_large() -> None:
    with pytest.raises(SpykeEncoderError):
        rate_encode([0.5], num_steps=4, seed=1 << 65)


# ---------------------------------------------------------------------------
# Temporal encoding semantics
# ---------------------------------------------------------------------------


def test_temporal_assigns_highest_to_first_step() -> None:
    train = temporal_encode([0.1, 0.9, 0.5], num_steps=3)
    # Sorted rank-order: idx 1 (0.9) at t=0, idx 2 (0.5) at t=1, idx 0 (0.1) at t=2.
    assert train.events == (
        SpikeEvent(neuron_idx=1, time_step=0),
        SpikeEvent(neuron_idx=2, time_step=1),
        SpikeEvent(neuron_idx=0, time_step=2),
    )


def test_temporal_drops_zero_features() -> None:
    train = temporal_encode([0.9, 0.0, 0.5], num_steps=3)
    assert train.events == (
        SpikeEvent(neuron_idx=0, time_step=0),
        SpikeEvent(neuron_idx=2, time_step=1),
    )


def test_temporal_truncates_to_num_steps() -> None:
    train = temporal_encode([0.9, 0.8, 0.7, 0.6], num_steps=2)
    # Only the top 2 ranks fit.
    assert train.events == (
        SpikeEvent(neuron_idx=0, time_step=0),
        SpikeEvent(neuron_idx=1, time_step=1),
    )


def test_temporal_ties_broken_by_index_ascending() -> None:
    train = temporal_encode([0.5, 0.5, 0.5], num_steps=3)
    assert train.events == (
        SpikeEvent(neuron_idx=0, time_step=0),
        SpikeEvent(neuron_idx=1, time_step=1),
        SpikeEvent(neuron_idx=2, time_step=2),
    )


def test_temporal_is_fully_deterministic() -> None:
    a = temporal_encode([0.3, 0.7, 0.5], num_steps=4)
    b = temporal_encode([0.3, 0.7, 0.5], num_steps=4)
    c = temporal_encode([0.3, 0.7, 0.5], num_steps=4)
    assert a == b == c
    assert a.digest == b.digest == c.digest


def test_temporal_uses_no_seed() -> None:
    train = temporal_encode([0.3, 0.7], num_steps=3)
    assert train.seed == 0
    assert train.method is EncodingMethod.TEMPORAL


def test_temporal_all_zero_features_yields_no_events() -> None:
    train = temporal_encode([0.0, 0.0, 0.0], num_steps=3)
    assert train.events == ()
    assert train.spike_count() == 0


# ---------------------------------------------------------------------------
# Rate encoding semantics
# ---------------------------------------------------------------------------


def test_rate_zero_features_yield_no_spikes() -> None:
    train = rate_encode([0.0, 0.0, 0.0], num_steps=8, seed=42)
    assert train.events == ()


def test_rate_one_features_yield_full_spikes() -> None:
    train = rate_encode([1.0, 1.0], num_steps=4, seed=7)
    # Every step, every neuron: u < 1.0 is always True for u in [0, 1).
    assert train.spike_count() == 2 * 4


def test_rate_is_deterministic_three_runs() -> None:
    args = dict(num_steps=16, seed=12345)
    a = rate_encode([0.2, 0.5, 0.8], **args)
    b = rate_encode([0.2, 0.5, 0.8], **args)
    c = rate_encode([0.2, 0.5, 0.8], **args)
    assert a == b == c
    assert a.digest == b.digest == c.digest


def test_rate_different_seeds_diverge() -> None:
    a = rate_encode([0.5, 0.5], num_steps=32, seed=1)
    b = rate_encode([0.5, 0.5], num_steps=32, seed=2)
    assert a != b
    assert a.digest != b.digest


def test_rate_seed_swap_does_not_collide_with_neuron_swap() -> None:
    a = rate_encode([0.4, 0.6], num_steps=16, seed=11)
    b = rate_encode([0.6, 0.4], num_steps=16, seed=11)
    # Swapping inputs produces different events (positions matter).
    assert a.events != b.events


def test_rate_count_in_expected_range_for_balanced_p() -> None:
    train = rate_encode([0.5], num_steps=4096, seed=99)
    n = train.spike_count()
    # 50% spike rate over 4096 trials — wide bound but informative.
    assert 1700 <= n <= 2400


# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------


def test_encode_dispatches_to_temporal() -> None:
    train = encode(
        [0.1, 0.9],
        method=EncodingMethod.TEMPORAL,
        num_steps=3,
    )
    assert train.method is EncodingMethod.TEMPORAL


def test_encode_dispatches_to_rate() -> None:
    train = encode(
        [0.5, 0.5],
        method=EncodingMethod.RATE,
        num_steps=4,
        seed=3,
    )
    assert train.method is EncodingMethod.RATE
    assert train.seed == 3


def test_encode_rejects_invalid_method() -> None:
    with pytest.raises(SpykeEncoderError):
        encode([0.5], method="bogus", num_steps=3)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SpikeTrain invariants
# ---------------------------------------------------------------------------


def test_spike_train_events_are_sorted() -> None:
    train = rate_encode([0.5, 0.5, 0.5], num_steps=8, seed=1)
    keys = [(ev.time_step, ev.neuron_idx) for ev in train.events]
    assert keys == sorted(keys)


def test_spike_train_no_duplicate_events() -> None:
    train = rate_encode([0.5, 0.5, 0.5], num_steps=8, seed=1)
    keys = [(ev.time_step, ev.neuron_idx) for ev in train.events]
    assert len(keys) == len(set(keys))


def test_spike_train_rejects_out_of_range_neuron_idx() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeTrain(
            num_neurons=2,
            num_steps=4,
            method=EncodingMethod.TEMPORAL,
            seed=0,
            events=(SpikeEvent(neuron_idx=2, time_step=0),),
        )


def test_spike_train_rejects_out_of_range_time_step() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeTrain(
            num_neurons=2,
            num_steps=4,
            method=EncodingMethod.TEMPORAL,
            seed=0,
            events=(SpikeEvent(neuron_idx=0, time_step=4),),
        )


def test_spike_train_rejects_unsorted_events() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeTrain(
            num_neurons=2,
            num_steps=4,
            method=EncodingMethod.TEMPORAL,
            seed=0,
            events=(
                SpikeEvent(neuron_idx=0, time_step=1),
                SpikeEvent(neuron_idx=0, time_step=0),
            ),
        )


def test_spike_train_rejects_duplicate_events() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeTrain(
            num_neurons=2,
            num_steps=4,
            method=EncodingMethod.TEMPORAL,
            seed=0,
            events=(
                SpikeEvent(neuron_idx=0, time_step=0),
                SpikeEvent(neuron_idx=0, time_step=0),
            ),
        )


def test_spike_train_rejects_zero_neurons() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeTrain(
            num_neurons=0,
            num_steps=4,
            method=EncodingMethod.TEMPORAL,
            seed=0,
            events=(),
        )


def test_spike_train_rejects_zero_steps() -> None:
    with pytest.raises(SpykeEncoderError):
        SpikeTrain(
            num_neurons=2,
            num_steps=0,
            method=EncodingMethod.TEMPORAL,
            seed=0,
            events=(),
        )


# ---------------------------------------------------------------------------
# Digest determinism
# ---------------------------------------------------------------------------


def test_digest_is_16_hex() -> None:
    train = temporal_encode([0.1, 0.5, 0.9], num_steps=4)
    assert len(train.digest) == 32
    int(train.digest, 16)  # parses as hex


def test_digest_three_run_equality() -> None:
    a = rate_encode([0.2, 0.7], num_steps=16, seed=5)
    b = rate_encode([0.2, 0.7], num_steps=16, seed=5)
    c = rate_encode([0.2, 0.7], num_steps=16, seed=5)
    assert a.digest == b.digest == c.digest


def test_digest_changes_with_method() -> None:
    rate = rate_encode([0.5, 0.5], num_steps=4, seed=0)
    temp = temporal_encode([0.5, 0.5], num_steps=4)
    assert rate.digest != temp.digest


def test_digest_changes_with_seed() -> None:
    a = rate_encode([0.5, 0.5], num_steps=16, seed=1)
    b = rate_encode([0.5, 0.5], num_steps=16, seed=2)
    assert a.digest != b.digest


def test_digest_changes_with_num_steps() -> None:
    a = temporal_encode([0.5, 0.5], num_steps=4)
    b = temporal_encode([0.5, 0.5], num_steps=8)
    assert a.digest != b.digest


# ---------------------------------------------------------------------------
# Dense projection
# ---------------------------------------------------------------------------


def test_to_dense_shape() -> None:
    train = temporal_encode([0.1, 0.9, 0.5], num_steps=3)
    dense = train.to_dense()
    assert len(dense) == 3
    for row in dense:
        assert len(row) == 3


def test_to_dense_marks_spikes_as_one() -> None:
    train = temporal_encode([0.9, 0.0, 0.5], num_steps=3)
    dense = train.to_dense()
    assert dense[0][0] == 1
    assert dense[1][2] == 1
    # Silence everywhere else.
    assert sum(sum(row) for row in dense) == 2


def test_spike_train_to_step_inputs_are_floats() -> None:
    train = temporal_encode([0.9, 0.5], num_steps=2)
    matrix = spike_train_to_step_inputs(train)
    assert all(isinstance(v, float) for row in matrix for v in row)
    assert matrix[0] == (1.0, 0.0)
    assert matrix[1] == (0.0, 1.0)


def test_spike_train_to_step_inputs_rejects_non_train() -> None:
    with pytest.raises(SpykeEncoderError):
        spike_train_to_step_inputs("not a train")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Production seam
# ---------------------------------------------------------------------------


def test_spyketorch_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        spyketorch_intensity_to_latency_factory()


def test_spyketorch_backend_protocol_is_runtime_checkable() -> None:
    assert isinstance(SpykeBackend, type)

    # Anonymous backend that satisfies the Protocol structurally.
    class _Stub:
        def intensity_to_latency(self, features, num_steps) -> SpikeTrain:
            return temporal_encode(features, num_steps=num_steps)

    assert isinstance(_Stub(), SpykeBackend)
