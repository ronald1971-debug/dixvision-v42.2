"""Tests for ``evolution_engine.genetic.strategy_chromosome`` (A-02.1)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from evolution_engine.genetic.strategy_chromosome import (
    DIGEST_HEX_LEN,
    MAX_PARAMETER_NAME_LEN,
    MAX_PARAMETERS_PER_CHROMOSOME,
    MAX_STRATEGY_ID_LEN,
    NEW_PIP_DEPENDENCIES,
    ChromosomeError,
    ParameterKind,
    ParameterSpec,
    StrategyChromosome,
    chromosome_digest,
    clip_to_bounds,
    pack,
    unpack,
)

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "evolution_engine"
    / "genetic"
    / "strategy_chromosome.py"
)

# ---------------------------------------------------------------------------
# Module metadata + AST authority pins
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_is_frozen_tuple() -> None:
    assert NEW_PIP_DEPENDENCIES == ()
    assert isinstance(NEW_PIP_DEPENDENCIES, tuple)


def test_constants_have_expected_shape() -> None:
    assert MAX_PARAMETER_NAME_LEN == 64
    assert MAX_PARAMETERS_PER_CHROMOSOME == 256
    assert MAX_STRATEGY_ID_LEN == 256
    assert DIGEST_HEX_LEN == 16


def test_adapted_from_header_present() -> None:
    src = _MOD_PATH.read_text(encoding="utf-8")
    head = src.splitlines()[:8]
    assert any("ADAPTED FROM" in line and "evotorch" in line for line in head)


def test_no_top_level_io_or_clock_imports() -> None:
    """The chromosome module is OFFLINE pure: no clock / io / random
    imports at any level."""

    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_modules = {
        "time",
        "datetime",
        "asyncio",
        "os",
        "io",
        "pathlib",
        "subprocess",
        "socket",
        "random",
        "secrets",
        "uuid",
        "numpy",
        "evotorch",
        "torch",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules, f"forbidden module imported: {node.module}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, (
                    f"forbidden module imported: {alias.name}"
                )


def test_no_engine_cross_imports() -> None:
    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_prefixes = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "registry",
        "ui",
        "cockpit",
        "dashboard",
        "dashboard_backend",
        "dashboard2026",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith(forbidden_prefixes), (
                f"forbidden cross-engine import: {node.module}"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes), (
                    f"forbidden cross-engine import: {alias.name}"
                )


def test_module_imports_only_stdlib() -> None:
    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            seen.add(node.module.split(".")[0])
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.add(alias.name.split(".")[0])
    allowed = {
        "__future__",
        "hashlib",
        "math",
        "collections",
        "dataclasses",
        "enum",
        "types",
    }
    assert seen <= allowed, f"unexpected imports: {seen - allowed}"


# ---------------------------------------------------------------------------
# ParameterSpec
# ---------------------------------------------------------------------------


def test_parameter_kind_values() -> None:
    assert ParameterKind.CONTINUOUS.value == "CONTINUOUS"
    assert ParameterKind.LOG_CONTINUOUS.value == "LOG_CONTINUOUS"
    assert ParameterKind.INTEGER.value == "INTEGER"


def test_parameter_spec_is_frozen_and_slotted() -> None:
    spec = ParameterSpec("alpha", ParameterKind.CONTINUOUS, 0.0, 1.0)
    assert hasattr(type(spec), "__slots__")
    assert not hasattr(spec, "__dict__")


def test_parameter_spec_continuous_happy() -> None:
    spec = ParameterSpec("alpha", ParameterKind.CONTINUOUS, -1.0, 1.0)
    assert spec.name == "alpha"
    assert spec.kind is ParameterKind.CONTINUOUS
    assert spec.low == -1.0
    assert spec.high == 1.0


def test_parameter_spec_log_continuous_happy() -> None:
    ParameterSpec("lr", ParameterKind.LOG_CONTINUOUS, 1e-6, 1e-1)


def test_parameter_spec_integer_happy() -> None:
    ParameterSpec("window", ParameterKind.INTEGER, 1, 100)


def test_parameter_spec_rejects_empty_name() -> None:
    with pytest.raises(ChromosomeError):
        ParameterSpec("", ParameterKind.CONTINUOUS, 0.0, 1.0)


def test_parameter_spec_rejects_oversize_name() -> None:
    with pytest.raises(ChromosomeError):
        ParameterSpec(
            "x" * (MAX_PARAMETER_NAME_LEN + 1),
            ParameterKind.CONTINUOUS,
            0.0,
            1.0,
        )


def test_parameter_spec_rejects_low_geq_high() -> None:
    with pytest.raises(ChromosomeError):
        ParameterSpec("a", ParameterKind.CONTINUOUS, 1.0, 1.0)
    with pytest.raises(ChromosomeError):
        ParameterSpec("a", ParameterKind.CONTINUOUS, 1.0, 0.0)


def test_parameter_spec_rejects_non_finite_bounds() -> None:
    with pytest.raises(ChromosomeError):
        ParameterSpec("a", ParameterKind.CONTINUOUS, float("nan"), 1.0)
    with pytest.raises(ChromosomeError):
        ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, float("inf"))


def test_parameter_spec_rejects_non_positive_log_low() -> None:
    with pytest.raises(ChromosomeError):
        ParameterSpec("lr", ParameterKind.LOG_CONTINUOUS, 0.0, 1.0)
    with pytest.raises(ChromosomeError):
        ParameterSpec("lr", ParameterKind.LOG_CONTINUOUS, -1.0, 1.0)


def test_parameter_spec_rejects_non_integer_integer_bounds() -> None:
    with pytest.raises(ChromosomeError):
        ParameterSpec("w", ParameterKind.INTEGER, 1.5, 10.0)
    with pytest.raises(ChromosomeError):
        ParameterSpec("w", ParameterKind.INTEGER, 1.0, 10.5)


def test_parameter_spec_rejects_bool_bounds() -> None:
    with pytest.raises(ChromosomeError):
        ParameterSpec("a", ParameterKind.CONTINUOUS, True, 1.0)  # type: ignore[arg-type]


def test_parameter_spec_rejects_non_kind() -> None:
    with pytest.raises(ChromosomeError):
        ParameterSpec("a", "CONTINUOUS", 0.0, 1.0)  # type: ignore[arg-type]


def test_parameter_spec_clip_inside() -> None:
    spec = ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0)
    assert spec.clip(0.5) == 0.5
    assert spec.clip(0.0) == 0.0
    assert spec.clip(1.0) == 1.0


def test_parameter_spec_clip_outside() -> None:
    spec = ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0)
    assert spec.clip(-0.1) == 0.0
    assert spec.clip(1.1) == 1.0


def test_parameter_spec_clip_rejects_non_finite() -> None:
    spec = ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0)
    with pytest.raises(ChromosomeError):
        spec.clip(float("nan"))


# ---------------------------------------------------------------------------
# StrategyChromosome
# ---------------------------------------------------------------------------


def _default_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec("alpha", ParameterKind.CONTINUOUS, 0.0, 1.0),
        ParameterSpec("lr", ParameterKind.LOG_CONTINUOUS, 1e-6, 1e-1),
        ParameterSpec("window", ParameterKind.INTEGER, 1, 100),
    )


def test_chromosome_is_frozen_and_slotted() -> None:
    c = StrategyChromosome(
        strategy_id="strat-001",
        specs=_default_specs(),
        values=(0.5, 1e-3, 50.0),
        version=1,
    )
    assert hasattr(type(c), "__slots__")
    assert not hasattr(c, "__dict__")


def test_chromosome_dimensionality_property() -> None:
    c = StrategyChromosome(
        strategy_id="strat-001",
        specs=_default_specs(),
        values=(0.5, 1e-3, 50.0),
        version=1,
    )
    assert c.dimensionality == 3


def test_chromosome_to_mapping_roundtrip() -> None:
    c = StrategyChromosome(
        strategy_id="strat-001",
        specs=_default_specs(),
        values=(0.5, 1e-3, 50.0),
        version=1,
    )
    m = c.to_mapping()
    assert dict(m) == {"alpha": 0.5, "lr": 1e-3, "window": 50.0}


def test_chromosome_to_mapping_is_immutable() -> None:
    c = StrategyChromosome(
        strategy_id="strat-001",
        specs=_default_specs(),
        values=(0.5, 1e-3, 50.0),
        version=1,
    )
    m = c.to_mapping()
    with pytest.raises(TypeError):
        m["alpha"] = 1.0  # type: ignore[index]


def test_chromosome_rejects_empty_strategy_id() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="",
            specs=_default_specs(),
            values=(0.5, 1e-3, 50.0),
            version=1,
        )


def test_chromosome_rejects_oversize_strategy_id() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="x" * (MAX_STRATEGY_ID_LEN + 1),
            specs=_default_specs(),
            values=(0.5, 1e-3, 50.0),
            version=1,
        )


def test_chromosome_rejects_specs_values_length_mismatch() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=_default_specs(),
            values=(0.5, 1e-3),
            version=1,
        )


def test_chromosome_rejects_empty_specs() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=(),
            values=(),
            version=1,
        )


def test_chromosome_rejects_duplicate_spec_names() -> None:
    specs = (
        ParameterSpec("alpha", ParameterKind.CONTINUOUS, 0.0, 1.0),
        ParameterSpec("alpha", ParameterKind.CONTINUOUS, 0.0, 1.0),
    )
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=specs,
            values=(0.5, 0.5),
            version=1,
        )


def test_chromosome_rejects_negative_version() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=_default_specs(),
            values=(0.5, 1e-3, 50.0),
            version=-1,
        )


def test_chromosome_rejects_bool_version() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=_default_specs(),
            values=(0.5, 1e-3, 50.0),
            version=True,  # type: ignore[arg-type]
        )


def test_chromosome_rejects_value_out_of_bounds() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=_default_specs(),
            values=(1.5, 1e-3, 50.0),
            version=0,
        )


def test_chromosome_rejects_non_finite_value() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=_default_specs(),
            values=(float("nan"), 1e-3, 50.0),
            version=0,
        )


def test_chromosome_rejects_non_integer_for_integer_kind() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=_default_specs(),
            values=(0.5, 1e-3, 50.5),
            version=0,
        )


def test_chromosome_rejects_oversize_dimension() -> None:
    big_specs = tuple(
        ParameterSpec(f"p{i}", ParameterKind.CONTINUOUS, 0.0, 1.0)
        for i in range(MAX_PARAMETERS_PER_CHROMOSOME + 1)
    )
    big_values = tuple(0.5 for _ in big_specs)
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=big_specs,
            values=big_values,
            version=0,
        )


def test_chromosome_rejects_non_str_meta() -> None:
    with pytest.raises(ChromosomeError):
        StrategyChromosome(
            strategy_id="strat-001",
            specs=_default_specs(),
            values=(0.5, 1e-3, 50.0),
            version=0,
            meta={"k": 1},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


def test_pack_continuous_passthrough() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    out = pack(specs, {"a": 0.5})
    assert out == (0.5,)


def test_pack_log_continuous_log10() -> None:
    specs = (ParameterSpec("lr", ParameterKind.LOG_CONTINUOUS, 1e-6, 1e-1),)
    out = pack(specs, {"lr": 1e-3})
    assert out[0] == pytest.approx(-3.0)


def test_pack_integer_passthrough() -> None:
    specs = (ParameterSpec("w", ParameterKind.INTEGER, 1, 100),)
    out = pack(specs, {"w": 50})
    assert out == (50.0,)


def test_pack_rejects_missing_key() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        pack(specs, {})


def test_pack_rejects_out_of_bounds() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        pack(specs, {"a": 1.5})


def test_pack_rejects_non_integer_for_integer_kind() -> None:
    specs = (ParameterSpec("w", ParameterKind.INTEGER, 1, 100),)
    with pytest.raises(ChromosomeError):
        pack(specs, {"w": 50.5})


def test_pack_rejects_non_finite() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        pack(specs, {"a": float("inf")})


def test_pack_rejects_empty_specs() -> None:
    with pytest.raises(ChromosomeError):
        pack((), {})


def test_pack_rejects_non_mapping() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        pack(specs, [("a", 0.5)])  # type: ignore[arg-type]


def test_pack_rejects_non_tuple_specs() -> None:
    with pytest.raises(ChromosomeError):
        pack(
            [ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0)],  # type: ignore[arg-type]
            {"a": 0.5},
        )


# ---------------------------------------------------------------------------
# unpack
# ---------------------------------------------------------------------------


def test_unpack_continuous_passthrough() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    out = unpack(specs, (0.5,))
    assert dict(out) == {"a": 0.5}


def test_unpack_log_continuous_inverse() -> None:
    specs = (ParameterSpec("lr", ParameterKind.LOG_CONTINUOUS, 1e-6, 1e-1),)
    # log10(1e-3) == -3
    out = unpack(specs, (-3.0,))
    assert out["lr"] == pytest.approx(1e-3)


def test_unpack_integer_rounds_half_to_even() -> None:
    specs = (ParameterSpec("w", ParameterKind.INTEGER, 0, 100),)
    # banker's rounding: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2
    assert unpack(specs, (0.5,))["w"] == 0.0
    assert unpack(specs, (1.5,))["w"] == 2.0
    assert unpack(specs, (2.5,))["w"] == 2.0


def test_unpack_clips_out_of_range_continuous() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    assert unpack(specs, (-0.5,))["a"] == 0.0
    assert unpack(specs, (1.5,))["a"] == 1.0


def test_unpack_clips_out_of_range_integer() -> None:
    specs = (ParameterSpec("w", ParameterKind.INTEGER, 1, 10),)
    assert unpack(specs, (-5.0,))["w"] == 1.0
    assert unpack(specs, (50.0,))["w"] == 10.0


def test_unpack_clips_out_of_range_log() -> None:
    specs = (ParameterSpec("lr", ParameterKind.LOG_CONTINUOUS, 1e-3, 1.0),)
    # 10**-10 < low; should clip to low
    assert unpack(specs, (-10.0,))["lr"] == pytest.approx(1e-3)
    # 10**+5 > high; should clip to high
    assert unpack(specs, (5.0,))["lr"] == pytest.approx(1.0)


def test_unpack_rejects_length_mismatch() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        unpack(specs, (0.5, 0.6))


def test_unpack_rejects_non_finite() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        unpack(specs, (float("nan"),))


def test_unpack_rejects_non_tuple_vector() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        unpack(specs, [0.5])  # type: ignore[arg-type]


def test_pack_unpack_roundtrip_continuous_and_integer() -> None:
    specs = (
        ParameterSpec("alpha", ParameterKind.CONTINUOUS, 0.0, 1.0),
        ParameterSpec("window", ParameterKind.INTEGER, 1, 100),
    )
    mapping = {"alpha": 0.123, "window": 42}
    decoded = unpack(specs, pack(specs, mapping))
    assert decoded["alpha"] == pytest.approx(0.123)
    assert decoded["window"] == 42.0


def test_pack_unpack_roundtrip_log_within_tolerance() -> None:
    specs = (ParameterSpec("lr", ParameterKind.LOG_CONTINUOUS, 1e-6, 1e-1),)
    mapping = {"lr": 1.234e-4}
    decoded = unpack(specs, pack(specs, mapping))
    assert math.isclose(decoded["lr"], 1.234e-4, rel_tol=1e-12, abs_tol=0.0)


# ---------------------------------------------------------------------------
# clip_to_bounds
# ---------------------------------------------------------------------------


def test_clip_to_bounds_passthrough_inside() -> None:
    specs = _default_specs()
    out = clip_to_bounds(specs, (0.5, 1e-3, 50.0))
    assert out == (0.5, 1e-3, 50.0)


def test_clip_to_bounds_clips_continuous() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    assert clip_to_bounds(specs, (-0.5,)) == (0.0,)
    assert clip_to_bounds(specs, (1.5,)) == (1.0,)


def test_clip_to_bounds_rounds_and_clips_integer() -> None:
    specs = (ParameterSpec("w", ParameterKind.INTEGER, 1, 10),)
    # 0.7 rounds to 1 (in-range)
    assert clip_to_bounds(specs, (0.7,)) == (1.0,)
    # 12.4 rounds to 12, then clips to 10
    assert clip_to_bounds(specs, (12.4,)) == (10.0,)


def test_clip_to_bounds_rejects_length_mismatch() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        clip_to_bounds(specs, (0.5, 0.6))


def test_clip_to_bounds_rejects_non_finite() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        clip_to_bounds(specs, (float("nan"),))


def test_clip_to_bounds_rejects_non_tuple() -> None:
    specs = (ParameterSpec("a", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    with pytest.raises(ChromosomeError):
        clip_to_bounds(specs, [0.5])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# chromosome_digest — INV-15 byte stability
# ---------------------------------------------------------------------------


def _canonical_chromosome() -> StrategyChromosome:
    return StrategyChromosome(
        strategy_id="strat-001",
        specs=_default_specs(),
        values=(0.5, 1e-3, 50.0),
        version=1,
        meta={"author": "evolution_engine"},
    )


def test_digest_is_16_lower_hex() -> None:
    d = chromosome_digest(_canonical_chromosome())
    assert isinstance(d, str)
    assert len(d) == DIGEST_HEX_LEN
    assert all(c in "0123456789abcdef" for c in d)


def test_digest_is_three_run_byte_identical() -> None:
    d1 = chromosome_digest(_canonical_chromosome())
    d2 = chromosome_digest(_canonical_chromosome())
    d3 = chromosome_digest(_canonical_chromosome())
    assert d1 == d2 == d3


def test_digest_changes_with_strategy_id() -> None:
    base = _canonical_chromosome()
    other = StrategyChromosome(
        strategy_id="strat-002",
        specs=base.specs,
        values=base.values,
        version=base.version,
        meta=base.meta,
    )
    assert chromosome_digest(base) != chromosome_digest(other)


def test_digest_changes_with_value() -> None:
    base = _canonical_chromosome()
    other = StrategyChromosome(
        strategy_id=base.strategy_id,
        specs=base.specs,
        values=(0.4, 1e-3, 50.0),
        version=base.version,
        meta=base.meta,
    )
    assert chromosome_digest(base) != chromosome_digest(other)


def test_digest_changes_with_version() -> None:
    base = _canonical_chromosome()
    other = StrategyChromosome(
        strategy_id=base.strategy_id,
        specs=base.specs,
        values=base.values,
        version=base.version + 1,
        meta=base.meta,
    )
    assert chromosome_digest(base) != chromosome_digest(other)


def test_digest_changes_with_meta() -> None:
    base = _canonical_chromosome()
    other = StrategyChromosome(
        strategy_id=base.strategy_id,
        specs=base.specs,
        values=base.values,
        version=base.version,
        meta={"author": "different"},
    )
    assert chromosome_digest(base) != chromosome_digest(other)


def test_digest_meta_is_sorted_so_key_order_does_not_matter() -> None:
    base = _canonical_chromosome()
    a = StrategyChromosome(
        strategy_id=base.strategy_id,
        specs=base.specs,
        values=base.values,
        version=base.version,
        meta={"a": "1", "b": "2"},
    )
    b = StrategyChromosome(
        strategy_id=base.strategy_id,
        specs=base.specs,
        values=base.values,
        version=base.version,
        meta={"b": "2", "a": "1"},
    )
    assert chromosome_digest(a) == chromosome_digest(b)


def test_digest_rejects_non_chromosome() -> None:
    with pytest.raises(ChromosomeError):
        chromosome_digest({"strategy_id": "strat-001"})  # type: ignore[arg-type]
