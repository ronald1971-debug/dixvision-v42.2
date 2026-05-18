"""Tests for evolution_engine.genetic.crossover (B-02 — DEAP crossover)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from evolution_engine.genetic import crossover as cx
from evolution_engine.genetic.crossover import (
    MAX_META_KEY_LEN,
    MAX_META_KEYS,
    MAX_META_VALUE_LEN,
    NEW_PIP_DEPENDENCIES,
    OPERATOR_BLEND,
    OPERATOR_SBX,
    OPERATOR_TWO_POINT,
    CrossoverOperatorError,
    blend_crossover,
    simulated_binary_crossover,
    two_point_crossover,
)
from evolution_engine.genetic.strategy_chromosome import (
    ParameterKind,
    ParameterSpec,
    StrategyChromosome,
    chromosome_digest,
)

# ---------------------------------------------------------------------------
# Module file path / AST authority pins
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).resolve().parents[1] / "evolution_engine" / "genetic" / "crossover.py"


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _three_kind_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(name="alpha", kind=ParameterKind.CONTINUOUS, low=-1.0, high=1.0),
        ParameterSpec(name="lr", kind=ParameterKind.LOG_CONTINUOUS, low=1e-4, high=1e-1),
        ParameterSpec(name="window", kind=ParameterKind.INTEGER, low=2.0, high=64.0),
    )


def _continuous_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(name="a", kind=ParameterKind.CONTINUOUS, low=-10.0, high=10.0),
        ParameterSpec(name="b", kind=ParameterKind.CONTINUOUS, low=-10.0, high=10.0),
        ParameterSpec(name="c", kind=ParameterKind.CONTINUOUS, low=-10.0, high=10.0),
        ParameterSpec(name="d", kind=ParameterKind.CONTINUOUS, low=-10.0, high=10.0),
    )


def _chromosome(
    values: tuple[float, ...],
    *,
    specs: tuple[ParameterSpec, ...] | None = None,
    version: int = 0,
    strategy_id: str = "strat-A",
) -> StrategyChromosome:
    return StrategyChromosome(
        strategy_id=strategy_id,
        specs=specs if specs is not None else _three_kind_specs(),
        values=values,
        version=version,
    )


@pytest.fixture
def parent_a() -> StrategyChromosome:
    return _chromosome((0.25, 1e-3, 16.0))


@pytest.fixture
def parent_b() -> StrategyChromosome:
    return _chromosome((-0.75, 1e-2, 48.0), version=2)


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------


def _imports(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def test_no_forbidden_imports() -> None:
    imports = _imports(_module_ast())
    forbidden = {
        "deap",
        "nevergrad",
        "numpy",
        "torch",
        "scipy",
        "evotorch",
        "pymoo",
        "random",
        "secrets",
        "os",
        "time",
        "datetime",
        "asyncio",
    }
    for f in forbidden:
        assert not any(imp == f or imp.startswith(f + ".") for imp in imports), (
            f"forbidden import found: {f}"
        )


def test_no_engine_cross_imports() -> None:
    imports = _imports(_module_ast())
    forbidden_prefixes = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "registry",
        "ui",
    )
    for imp in imports:
        for prefix in forbidden_prefixes:
            assert not (imp == prefix or imp.startswith(prefix + ".")), (
                f"engine cross-import found: {imp}"
            )


def test_does_not_construct_patch_proposal() -> None:
    """B27 / B28 / INV-71: this leaf must not construct PatchProposal —
    only the evolution-engine orchestrators do."""

    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name not in {
                "PatchProposal",
                "SignalEvent",
                "GovernanceDecision",
                "ExecutionIntent",
            }, f"forbidden typed-event constructor: {name}"


def _attr_chain_starts_with(node: ast.Attribute, allowed_roots: tuple[str, ...]) -> bool:
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return isinstance(cur, ast.Name) and cur.id in allowed_roots


def test_no_clock_or_io() -> None:
    tree = _module_ast()
    forbidden_attrs = {
        "time_ns",
        "monotonic_ns",
        "monotonic",
        "perf_counter",
        "now",
        "today",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attrs, f"forbidden attribute call: {ast.dump(node)}"


def test_module_is_pure_function_layer() -> None:
    tree = _module_ast()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.Expr, ast.AnnAssign, ast.Assign)):
            continue
        raise AssertionError(f"unexpected top-level node: {ast.dump(node)}")


def test_adapted_from_header_present() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: DEAP/deap" in src
    assert "cxSimulatedBinary" in src
    assert "cxBlend" in src
    assert "cxTwoPoint" in src


def test_new_pip_dependencies_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


# ---------------------------------------------------------------------------
# PRNG helpers (INV-15 building blocks)
# ---------------------------------------------------------------------------


def test_splitmix64_is_deterministic() -> None:
    assert cx._splitmix64(0) == cx._splitmix64(0)
    assert cx._splitmix64(1) != cx._splitmix64(2)


def test_uniform01_in_open_unit_interval() -> None:
    for k in range(50):
        u = cx._uniform01(42, k, 7)
        assert 0.0 < u <= 1.0


def test_uniform01_deterministic() -> None:
    assert cx._uniform01(1, 2, 3) == cx._uniform01(1, 2, 3)


def test_uniform_int_in_range() -> None:
    for k in range(100):
        v = cx._uniform_int(0, 5, k, 17)
        assert 0 <= v <= 5


def test_uniform_int_deterministic() -> None:
    assert cx._uniform_int(0, 9, 1, 2, 3) == cx._uniform_int(0, 9, 1, 2, 3)


def test_uniform_int_rejects_empty_range() -> None:
    with pytest.raises(CrossoverOperatorError):
        cx._uniform_int(5, 0, 1)


# ---------------------------------------------------------------------------
# Encoded-bounds + encode helpers
# ---------------------------------------------------------------------------


def test_encoded_bounds_log_continuous() -> None:
    bounds = cx._encoded_bounds(_three_kind_specs())
    assert bounds[0] == (-1.0, 1.0)
    assert bounds[1] == pytest.approx((math.log10(1e-4), math.log10(1e-1)))
    assert bounds[2] == (2.0, 64.0)


def test_encode_vector_round_trip() -> None:
    specs = _three_kind_specs()
    decoded = (0.5, 1e-3, 32.0)
    enc = cx._encode_vector(specs, decoded)
    back = cx._decode_to_values(specs, enc)
    assert back[0] == pytest.approx(0.5)
    assert back[1] == pytest.approx(1e-3)
    assert back[2] == 32.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_pair_compatible_rejects_strategy_id_mismatch(
    parent_a: StrategyChromosome,
) -> None:
    other = _chromosome((0.0, 1e-3, 4.0), strategy_id="strat-B")
    with pytest.raises(CrossoverOperatorError, match="strategy_id mismatch"):
        simulated_binary_crossover(
            parent_a=parent_a,
            parent_b=other,
            eta=20.0,
            seed=1,
            generation=0,
            individual=0,
        )


def test_validate_pair_compatible_rejects_spec_mismatch(
    parent_a: StrategyChromosome,
) -> None:
    other = _chromosome((0.0, 0.0, 0.0, 0.0), specs=_continuous_specs(), strategy_id="strat-A")
    with pytest.raises(CrossoverOperatorError, match="specs mismatch"):
        simulated_binary_crossover(
            parent_a=parent_a,
            parent_b=other,
            eta=20.0,
            seed=1,
            generation=0,
            individual=0,
        )


def test_rejects_non_chromosome_parent() -> None:
    with pytest.raises(CrossoverOperatorError, match="StrategyChromosome"):
        simulated_binary_crossover(
            parent_a="nope",  # type: ignore[arg-type]
            parent_b=_chromosome((0.0, 1e-3, 4.0)),
            eta=20.0,
            seed=1,
            generation=0,
            individual=0,
        )


@pytest.mark.parametrize("seed", [-1, "0", 1.5, True])
def test_rejects_bad_seed(parent_a, parent_b, seed) -> None:
    with pytest.raises(CrossoverOperatorError):
        simulated_binary_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            eta=20.0,
            seed=seed,
            generation=0,
            individual=0,
        )


@pytest.mark.parametrize("bad_eta", [-1.0, math.inf, math.nan, "20"])
def test_sbx_rejects_bad_eta(parent_a, parent_b, bad_eta) -> None:
    with pytest.raises(CrossoverOperatorError):
        simulated_binary_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            eta=bad_eta,
            seed=1,
            generation=0,
            individual=0,
        )


@pytest.mark.parametrize("bad_alpha", [-0.1, math.inf, math.nan, "0.5"])
def test_blend_rejects_bad_alpha(parent_a, parent_b, bad_alpha) -> None:
    with pytest.raises(CrossoverOperatorError):
        blend_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            alpha=bad_alpha,
            seed=1,
            generation=0,
            individual=0,
        )


def test_extra_meta_rejects_reserved_key(parent_a, parent_b) -> None:
    with pytest.raises(CrossoverOperatorError, match="reserved"):
        simulated_binary_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            eta=20.0,
            seed=1,
            generation=0,
            individual=0,
            extra_meta={"operator": "x"},
        )


def test_extra_meta_rejects_too_many_keys(parent_a, parent_b) -> None:
    extra = {f"k{i}": "v" for i in range(MAX_META_KEYS + 1)}
    with pytest.raises(CrossoverOperatorError, match="keys"):
        blend_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            alpha=0.5,
            seed=1,
            generation=0,
            individual=0,
            extra_meta=extra,
        )


def test_extra_meta_rejects_oversize_value(parent_a, parent_b) -> None:
    with pytest.raises(CrossoverOperatorError, match="value length"):
        two_point_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            seed=1,
            generation=0,
            individual=0,
            extra_meta={"trace": "x" * (MAX_META_VALUE_LEN + 1)},
        )


def test_extra_meta_rejects_oversize_key(parent_a, parent_b) -> None:
    with pytest.raises(CrossoverOperatorError, match="key length"):
        two_point_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            seed=1,
            generation=0,
            individual=0,
            extra_meta={"k" * (MAX_META_KEY_LEN + 1): "v"},
        )


# ---------------------------------------------------------------------------
# SBX (Simulated Binary Crossover)
# ---------------------------------------------------------------------------


def test_sbx_returns_two_children(parent_a, parent_b) -> None:
    c1, c2 = simulated_binary_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        eta=20.0,
        seed=42,
        generation=0,
        individual=0,
    )
    assert isinstance(c1, StrategyChromosome)
    assert isinstance(c2, StrategyChromosome)
    assert c1.strategy_id == parent_a.strategy_id
    assert c1.specs == parent_a.specs
    assert c1.dimensionality == parent_a.dimensionality
    # versions bump per parent
    assert c1.version == parent_a.version + 1
    assert c2.version == parent_b.version + 1


def test_sbx_is_deterministic(parent_a, parent_b) -> None:
    c1_run1, c2_run1 = simulated_binary_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        eta=20.0,
        seed=7,
        generation=3,
        individual=11,
    )
    c1_run2, c2_run2 = simulated_binary_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        eta=20.0,
        seed=7,
        generation=3,
        individual=11,
    )
    c1_run3, c2_run3 = simulated_binary_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        eta=20.0,
        seed=7,
        generation=3,
        individual=11,
    )
    assert chromosome_digest(c1_run1) == chromosome_digest(c1_run2) == chromosome_digest(c1_run3)
    assert chromosome_digest(c2_run1) == chromosome_digest(c2_run2) == chromosome_digest(c2_run3)


def test_sbx_seed_sensitivity(parent_a, parent_b) -> None:
    c1, _ = simulated_binary_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        eta=20.0,
        seed=1,
        generation=0,
        individual=0,
    )
    c1_other, _ = simulated_binary_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        eta=20.0,
        seed=999,
        generation=0,
        individual=0,
    )
    assert chromosome_digest(c1) != chromosome_digest(c1_other)


def test_sbx_meta_keys_sorted(parent_a, parent_b) -> None:
    c1, _ = simulated_binary_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        eta=20.0,
        seed=1,
        generation=0,
        individual=0,
        extra_meta={"trial": "alpha"},
    )
    keys = list(c1.meta.keys())
    assert keys == sorted(keys)
    assert c1.meta["operator"] == OPERATOR_SBX
    assert c1.meta["seed"] == "1"
    assert c1.meta["generation"] == "0"
    assert c1.meta["individual"] == "0"
    assert c1.meta["child"] == "0"
    assert c1.meta["parent_a_digest"] == chromosome_digest(parent_a)
    assert c1.meta["parent_b_digest"] == chromosome_digest(parent_b)
    assert c1.meta["trial"] == "alpha"
    assert "eta" in c1.meta


def test_sbx_children_are_feasible(parent_a, parent_b) -> None:
    c1, c2 = simulated_binary_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        eta=20.0,
        seed=12345,
        generation=0,
        individual=0,
    )
    for child in (c1, c2):
        for spec, value in zip(child.specs, child.values, strict=True):
            assert spec.low <= value <= spec.high
            if spec.kind is ParameterKind.INTEGER:
                assert int(value) == value


def test_sbx_identical_parents_yield_identical_children() -> None:
    """SBX algebra: when y1 == y2 every child equals the parent value
    regardless of beta_q (both linear combinations collapse to y)."""

    parent = _chromosome((0.0, 1e-3, 16.0))
    c1, c2 = simulated_binary_crossover(
        parent_a=parent,
        parent_b=parent,
        eta=20.0,
        seed=1,
        generation=0,
        individual=0,
    )
    # CONTINUOUS / LOG_CONTINUOUS collapse exactly; INTEGER passes through round()
    assert c1.values[0] == pytest.approx(0.0, abs=1e-12)
    assert c2.values[0] == pytest.approx(0.0, abs=1e-12)
    assert c1.values[1] == pytest.approx(1e-3, rel=1e-9)
    assert c2.values[1] == pytest.approx(1e-3, rel=1e-9)
    assert c1.values[2] == 16.0
    assert c2.values[2] == 16.0


def test_sbx_eta_zero_is_arithmetic_recombination() -> None:
    """eta=0 → exponent=1 → beta_q is linear in u → mean over many
    children equals the parent midpoint."""

    parent_a = _chromosome(
        (-5.0,), specs=(ParameterSpec("x", ParameterKind.CONTINUOUS, -10.0, 10.0),)
    )
    parent_b = _chromosome(
        (5.0,), specs=(ParameterSpec("x", ParameterKind.CONTINUOUS, -10.0, 10.0),)
    )
    s = 0.0
    n = 200
    for i in range(n):
        c1, c2 = simulated_binary_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            eta=0.0,
            seed=1,
            generation=0,
            individual=i,
        )
        s += c1.values[0] + c2.values[0]
    mean = s / (2 * n)
    # parent midpoint is 0.0; expect mean within ~0.5 with 200 samples
    assert abs(mean) < 0.5


# ---------------------------------------------------------------------------
# Blend (BLX-alpha)
# ---------------------------------------------------------------------------


def test_blend_returns_two_children(parent_a, parent_b) -> None:
    c1, c2 = blend_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        alpha=0.5,
        seed=1,
        generation=0,
        individual=0,
    )
    assert isinstance(c1, StrategyChromosome)
    assert isinstance(c2, StrategyChromosome)
    assert c1.dimensionality == parent_a.dimensionality


def test_blend_is_deterministic(parent_a, parent_b) -> None:
    c1_a, c2_a = blend_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        alpha=0.5,
        seed=5,
        generation=2,
        individual=4,
    )
    c1_b, c2_b = blend_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        alpha=0.5,
        seed=5,
        generation=2,
        individual=4,
    )
    assert chromosome_digest(c1_a) == chromosome_digest(c1_b)
    assert chromosome_digest(c2_a) == chromosome_digest(c2_b)


def test_blend_alpha_zero_is_arithmetic_recombination() -> None:
    """alpha=0 → gamma = u, child1 = (1-u)*y1 + u*y2, every dim is
    a strict convex combination of the parents (inside [min, max])."""

    parent_a = _chromosome(
        (-1.0,), specs=(ParameterSpec("x", ParameterKind.CONTINUOUS, -10.0, 10.0),)
    )
    parent_b = _chromosome(
        (5.0,), specs=(ParameterSpec("x", ParameterKind.CONTINUOUS, -10.0, 10.0),)
    )
    for i in range(50):
        c1, c2 = blend_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            alpha=0.0,
            seed=1,
            generation=0,
            individual=i,
        )
        for c in (c1, c2):
            assert -1.0 <= c.values[0] <= 5.0


def test_blend_alpha_extends_range() -> None:
    """alpha=0.5 → children may sample from [-0.5*d below min, 0.5*d
    above max], but still clipped to spec bounds."""

    parent_a = _chromosome(
        (0.0,), specs=(ParameterSpec("x", ParameterKind.CONTINUOUS, -10.0, 10.0),)
    )
    parent_b = _chromosome(
        (4.0,), specs=(ParameterSpec("x", ParameterKind.CONTINUOUS, -10.0, 10.0),)
    )
    # alpha=0.5 with parents [0,4]: range is [-2, 6]
    saw_below_min_parent = False
    saw_above_max_parent = False
    for i in range(200):
        c1, _ = blend_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            alpha=0.5,
            seed=1,
            generation=0,
            individual=i,
        )
        if c1.values[0] < 0.0:
            saw_below_min_parent = True
        if c1.values[0] > 4.0:
            saw_above_max_parent = True
        assert -10.0 <= c1.values[0] <= 10.0  # always clipped to spec
    assert saw_below_min_parent
    assert saw_above_max_parent


def test_blend_children_are_feasible(parent_a, parent_b) -> None:
    for i in range(20):
        c1, c2 = blend_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            alpha=0.3,
            seed=99,
            generation=0,
            individual=i,
        )
        for child in (c1, c2):
            for spec, value in zip(child.specs, child.values, strict=True):
                assert spec.low <= value <= spec.high
                if spec.kind is ParameterKind.INTEGER:
                    assert int(value) == value


def test_blend_identical_parents_yield_identical_children() -> None:
    parent = _chromosome((0.5, 1e-2, 32.0))
    c1, c2 = blend_crossover(
        parent_a=parent,
        parent_b=parent,
        alpha=0.5,
        seed=1,
        generation=0,
        individual=0,
    )
    # identical parents collapse for every alpha (algebraic identity)
    for child in (c1, c2):
        assert child.values[0] == pytest.approx(0.5, abs=1e-12)
        assert child.values[1] == pytest.approx(1e-2, rel=1e-9)
        assert child.values[2] == 32.0


def test_blend_meta_includes_alpha(parent_a, parent_b) -> None:
    c1, _ = blend_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        alpha=0.25,
        seed=1,
        generation=0,
        individual=0,
    )
    assert c1.meta["operator"] == OPERATOR_BLEND
    assert "alpha" in c1.meta


# ---------------------------------------------------------------------------
# Two-point crossover
# ---------------------------------------------------------------------------


def test_two_point_returns_two_children(parent_a, parent_b) -> None:
    c1, c2 = two_point_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        seed=1,
        generation=0,
        individual=0,
    )
    assert isinstance(c1, StrategyChromosome)
    assert isinstance(c2, StrategyChromosome)


def test_two_point_is_deterministic(parent_a, parent_b) -> None:
    c1_a, c2_a = two_point_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        seed=10,
        generation=5,
        individual=2,
    )
    c1_b, c2_b = two_point_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        seed=10,
        generation=5,
        individual=2,
    )
    assert chromosome_digest(c1_a) == chromosome_digest(c1_b)
    assert chromosome_digest(c2_a) == chromosome_digest(c2_b)
    assert c1_a.meta["cut_lo"] == c1_b.meta["cut_lo"]
    assert c1_a.meta["cut_hi"] == c1_b.meta["cut_hi"]


def test_two_point_cuts_are_ordered(parent_a, parent_b) -> None:
    for i in range(50):
        c1, _ = two_point_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            seed=42,
            generation=0,
            individual=i,
        )
        lo = int(c1.meta["cut_lo"])
        hi = int(c1.meta["cut_hi"])
        assert 0 <= lo <= hi <= parent_a.dimensionality


def test_two_point_swaps_segment_only() -> None:
    """Outside [lo, hi) the child equals its parent verbatim; inside
    [lo, hi) it equals the *other* parent."""

    specs = _continuous_specs()
    a = _chromosome((1.0, 2.0, 3.0, 4.0), specs=specs)
    b = _chromosome((-1.0, -2.0, -3.0, -4.0), specs=specs)
    c1, c2 = two_point_crossover(
        parent_a=a,
        parent_b=b,
        seed=1,
        generation=0,
        individual=0,
    )
    lo = int(c1.meta["cut_lo"])
    hi = int(c1.meta["cut_hi"])
    for i in range(len(specs)):
        if lo <= i < hi:
            assert c1.values[i] == b.values[i]
            assert c2.values[i] == a.values[i]
        else:
            assert c1.values[i] == a.values[i]
            assert c2.values[i] == b.values[i]


def test_two_point_handles_dimension_one() -> None:
    """When n=1 the cut range collapses; children must remain
    feasible (one of two no-op cases: [0,0], [0,1], or [1,1])."""

    single_spec = (ParameterSpec("x", ParameterKind.CONTINUOUS, 0.0, 1.0),)
    a = _chromosome((0.25,), specs=single_spec)
    b = _chromosome((0.75,), specs=single_spec)
    for i in range(20):
        c1, c2 = two_point_crossover(
            parent_a=a,
            parent_b=b,
            seed=1,
            generation=0,
            individual=i,
        )
        for child in (c1, c2):
            assert child.dimensionality == 1
            assert 0.0 <= child.values[0] <= 1.0


def test_two_point_meta_includes_cuts(parent_a, parent_b) -> None:
    c1, _ = two_point_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        seed=1,
        generation=0,
        individual=0,
    )
    assert c1.meta["operator"] == OPERATOR_TWO_POINT
    assert "cut_lo" in c1.meta
    assert "cut_hi" in c1.meta


def test_two_point_preserves_integer_kind_exactly() -> None:
    """Two-point operates on decoded values, so INTEGER kinds are
    transplanted bit-for-bit (no float-encode/round trip)."""

    specs = (
        ParameterSpec("window", ParameterKind.INTEGER, 2.0, 1024.0),
        ParameterSpec("depth", ParameterKind.INTEGER, 1.0, 32.0),
    )
    a = _chromosome((100.0, 8.0), specs=specs)
    b = _chromosome((300.0, 16.0), specs=specs)
    c1, c2 = two_point_crossover(
        parent_a=a,
        parent_b=b,
        seed=1,
        generation=0,
        individual=0,
    )
    for child in (c1, c2):
        for value in child.values:
            assert int(value) == value


def test_two_point_children_are_feasible(parent_a, parent_b) -> None:
    for i in range(20):
        c1, c2 = two_point_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            seed=7,
            generation=0,
            individual=i,
        )
        for child in (c1, c2):
            for spec, value in zip(child.specs, child.values, strict=True):
                assert spec.low <= value <= spec.high


# ---------------------------------------------------------------------------
# Shared INV-15 properties across all three operators
# ---------------------------------------------------------------------------


def test_three_run_byte_identical_replay(parent_a, parent_b) -> None:
    digests: list[tuple[str, str]] = []
    for _ in range(3):
        c1_sbx, c2_sbx = simulated_binary_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            eta=20.0,
            seed=2024,
            generation=4,
            individual=8,
        )
        c1_bl, c2_bl = blend_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            alpha=0.5,
            seed=2024,
            generation=4,
            individual=8,
        )
        c1_tp, c2_tp = two_point_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            seed=2024,
            generation=4,
            individual=8,
        )
        digests.append(
            (
                chromosome_digest(c1_sbx) + chromosome_digest(c2_sbx),
                chromosome_digest(c1_bl)
                + chromosome_digest(c2_bl)
                + chromosome_digest(c1_tp)
                + chromosome_digest(c2_tp),
            )
        )
    assert digests[0] == digests[1] == digests[2]


def test_extra_meta_key_order_invariance(parent_a, parent_b) -> None:
    """INV-15: extra_meta input dict-insertion-order must NOT affect
    the offspring digest."""

    forward = {"a": "1", "b": "2", "c": "3"}
    reverse = {"c": "3", "b": "2", "a": "1"}
    c1_fwd, _ = blend_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        alpha=0.3,
        seed=1,
        generation=0,
        individual=0,
        extra_meta=forward,
    )
    c1_rev, _ = blend_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        alpha=0.3,
        seed=1,
        generation=0,
        individual=0,
        extra_meta=reverse,
    )
    assert chromosome_digest(c1_fwd) == chromosome_digest(c1_rev)


def test_version_bumps_per_parent_side(parent_a, parent_b) -> None:
    """Each child inherits version from its respective parent side
    (parent_a -> c1, parent_b -> c2). Ensures version monotonicity
    on the receiving lineage."""

    for op_call, kwargs in [
        (simulated_binary_crossover, {"eta": 20.0}),
        (blend_crossover, {"alpha": 0.5}),
        (two_point_crossover, {}),
    ]:
        c1, c2 = op_call(
            parent_a=parent_a,
            parent_b=parent_b,
            seed=1,
            generation=0,
            individual=0,
            **kwargs,  # type: ignore[arg-type]
        )
        assert c1.version == parent_a.version + 1
        assert c2.version == parent_b.version + 1


def test_public_surface() -> None:
    expected = {
        "MAX_META_KEYS",
        "MAX_META_KEY_LEN",
        "MAX_META_VALUE_LEN",
        "NEW_PIP_DEPENDENCIES",
        "OPERATOR_BLEND",
        "OPERATOR_SBX",
        "OPERATOR_TWO_POINT",
        "CrossoverOperatorError",
        "blend_crossover",
        "simulated_binary_crossover",
        "two_point_crossover",
    }
    assert set(cx.__all__) == expected
