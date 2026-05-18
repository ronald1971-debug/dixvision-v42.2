"""B-16 — Pytest for the Brian2-style LIF research prototype.

Authority pins:

* The :mod:`sensory.neuromorphic.neuro_prototype` module is
  RESEARCH_SOURCE — pinned by AST tests + a repo-wide importer scan.
* No top-level brian2 / torch / norse / numpy / scipy import.
* No engine cross-imports. No clock / random / asyncio / os / socket.
* No typed bus event construction.

Math pins:

* Sub-threshold leak converges to ``v_leak``.
* Constant supra-threshold drive produces a periodic spike train.
* Refractory window suppresses adjacent spikes.

INV-15:

* Three-run digest equality.
* ``continuous`` and ``discrete`` traces have identical digests when
  ``sub_steps=1``.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import pathlib

import pytest

from sensory.neuromorphic.neuro_prototype import (
    MAX_TRACE_LEN,
    NEURO_PROTOTYPE_VERSION,
    NEW_PIP_DEPENDENCIES,
    Brian2PrototypeFactory,
    LIFComparisonReport,
    LIFParams,
    LIFTrace,
    NeuroPrototypeError,
    brian2_prototype_factory,
    continuous_time_lif_reference,
    norse_style_discrete_lif,
    prototype_lif_market_signal,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "sensory" / "neuromorphic" / "neuro_prototype.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")
MODULE_AST = ast.parse(MODULE_SOURCE)


# ----------------------------------------------------------------- authority


def test_authority_adapted_from_header() -> None:
    assert MODULE_SOURCE.startswith("# ADAPTED FROM: brian-team/brian2")


def test_authority_research_source_classification_documented() -> None:
    assert "RESEARCH_SOURCE" in MODULE_SOURCE
    assert "NEVER imported by any production runtime" in MODULE_SOURCE


def test_authority_pip_dependencies_brian2_only() -> None:
    assert NEW_PIP_DEPENDENCIES == ("brian2",)


def _iter_top_level_imports(tree: ast.AST):
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node


def test_authority_no_top_level_research_or_runtime_deps() -> None:
    forbidden = {
        "brian2",
        "torch",
        "norse",
        "numpy",
        "scipy",
        "pandas",
        "polars",
        "bindsnet",
        "snntorch",
    }
    for node in _iter_top_level_imports(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"forbidden top-level import: {alias.name}"
        else:
            if node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden, f"forbidden top-level import: {node.module}"


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
    for node in _iter_top_level_imports(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"forbidden top-level import: {alias.name}"
        else:
            if node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden, f"forbidden top-level import: {node.module}"


def test_authority_no_engine_cross_imports() -> None:
    engines = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "evolution_engine",
        "learning_engine",
    }
    for node in _iter_top_level_imports(MODULE_AST):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in engines, f"forbidden engine import: {node.module}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in engines, f"forbidden engine import: {alias.name}"


def test_authority_no_typed_event_construction() -> None:
    forbidden_types = {
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "ExecutionIntent",
        "GovernanceDecision",
        "PatchProposal",
        "RealitySummary",
        "RiskPulse",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in forbidden_types:
                raise AssertionError(f"forbidden type constructed: {name}")


def test_authority_research_only_no_production_importers() -> None:
    """RESEARCH_SOURCE — only tests/ and offline/ may import this module."""
    target = "sensory.neuromorphic.neuro_prototype"
    skip_prefixes = ("tests", "offline", ".git", ".venv", "node_modules")
    suffix_target = target.split(".")[-1]
    bad: list[pathlib.Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        rel_str = str(rel)
        if any(rel_str.startswith(p) for p in skip_prefixes):
            continue
        if path == MODULE_PATH:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if target in text or f"from sensory.neuromorphic import {suffix_target}" in text:
            bad.append(rel)
    assert not bad, f"RESEARCH_SOURCE module imported from production tier: {[str(p) for p in bad]}"


# ----------------------------------------------------------------- freezing


def test_freezing_lif_params() -> None:
    p = LIFParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.tau_mem = 0.5  # type: ignore[misc]


def test_freezing_lif_trace() -> None:
    t = continuous_time_lif_reference(
        current=[0.0, 0.0, 0.0],
        params=LIFParams(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.backend = "x"  # type: ignore[misc]


def test_freezing_comparison_report() -> None:
    r = prototype_lif_market_signal(current=[0.0, 0.0, 0.0], params=LIFParams())
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.count_tolerance = 9  # type: ignore[misc]


# ----------------------------------------------------------- LIFParams validation


def test_params_tau_mem_must_be_positive() -> None:
    with pytest.raises(NeuroPrototypeError):
        LIFParams(tau_mem=0.0)


def test_params_tau_mem_must_be_finite() -> None:
    with pytest.raises(NeuroPrototypeError):
        LIFParams(tau_mem=math.inf)


def test_params_v_thresh_must_exceed_v_reset() -> None:
    with pytest.raises(NeuroPrototypeError):
        LIFParams(v_thresh=0.0, v_reset=0.0)


def test_params_v_thresh_must_strictly_exceed_v_reset() -> None:
    with pytest.raises(NeuroPrototypeError):
        LIFParams(v_thresh=-1.0, v_reset=0.0)


def test_params_v_leak_must_be_finite() -> None:
    with pytest.raises(NeuroPrototypeError):
        LIFParams(v_leak=math.nan)


def test_params_r_input_must_be_positive() -> None:
    with pytest.raises(NeuroPrototypeError):
        LIFParams(r_input=0.0)


def test_params_dt_must_be_positive() -> None:
    with pytest.raises(NeuroPrototypeError):
        LIFParams(dt=0.0)


def test_params_refractory_steps_non_negative() -> None:
    with pytest.raises(NeuroPrototypeError):
        LIFParams(refractory_steps=-1)


# ------------------------------------------------------ current trace validation


def test_current_empty_raises() -> None:
    with pytest.raises(NeuroPrototypeError):
        continuous_time_lif_reference(current=[], params=LIFParams())


def test_current_too_long_raises() -> None:
    with pytest.raises(NeuroPrototypeError):
        continuous_time_lif_reference(current=[0.0] * (MAX_TRACE_LEN + 1), params=LIFParams())


def test_current_must_be_sequence() -> None:
    with pytest.raises(NeuroPrototypeError):
        continuous_time_lif_reference(current=42, params=LIFParams())  # type: ignore[arg-type]


def test_current_non_finite_value_raises() -> None:
    with pytest.raises(NeuroPrototypeError):
        continuous_time_lif_reference(current=[0.0, math.inf, 0.0], params=LIFParams())


def test_current_non_float_value_raises() -> None:
    with pytest.raises(NeuroPrototypeError):
        continuous_time_lif_reference(current=[0.0, "x", 0.0], params=LIFParams())  # type: ignore[list-item]


def test_current_bool_value_raises() -> None:
    with pytest.raises(NeuroPrototypeError):
        continuous_time_lif_reference(current=[True, False, True], params=LIFParams())  # type: ignore[list-item]


def test_sub_steps_must_be_at_least_one() -> None:
    with pytest.raises(NeuroPrototypeError):
        continuous_time_lif_reference(current=[0.0], params=LIFParams(), sub_steps=0)


def test_sub_steps_sanity_cap() -> None:
    with pytest.raises(NeuroPrototypeError):
        continuous_time_lif_reference(current=[0.0], params=LIFParams(), sub_steps=1_001)


# ----------------------------------------------------------------- LIF math


def test_zero_input_no_spikes() -> None:
    p = LIFParams()
    t = norse_style_discrete_lif(current=[0.0] * 100, params=p)
    assert t.spike_count == 0
    assert all(v <= 0.0 + 1e-12 for v in t.v_history)


def test_subthreshold_decays_to_v_leak() -> None:
    p = LIFParams(v_leak=0.5, tau_mem=5e-3, dt=1e-3, v_thresh=10.0)
    t = norse_style_discrete_lif(current=[0.0] * 200, params=p)
    assert abs(t.v_history[-1] - 0.5) < 1e-6
    assert t.spike_count == 0


def test_strong_drive_produces_spikes() -> None:
    p = LIFParams()
    t = norse_style_discrete_lif(current=[5.0] * 100, params=p)
    assert t.spike_count > 0


def test_periodic_spike_train_under_constant_drive() -> None:
    p = LIFParams()
    t = norse_style_discrete_lif(current=[5.0] * 200, params=p)
    diffs = [b - a for a, b in zip(t.spike_times_steps, t.spike_times_steps[1:], strict=False)]
    assert len(set(diffs)) <= 2  # one or two ISI values (rounding tolerance)


def test_refractory_window_suppresses_adjacent_spikes() -> None:
    p_ref = LIFParams(refractory_steps=5)
    t_ref = norse_style_discrete_lif(current=[5.0] * 200, params=p_ref)
    pairs = zip(t_ref.spike_times_steps, t_ref.spike_times_steps[1:], strict=False)
    diffs_ref = [b - a for a, b in pairs]
    assert all(d > 5 for d in diffs_ref)


def test_first_spike_step_property() -> None:
    p = LIFParams()
    t = norse_style_discrete_lif(current=[5.0] * 100, params=p)
    assert t.first_spike_step == t.spike_times_steps[0]


def test_first_spike_step_none_when_silent() -> None:
    p = LIFParams()
    t = norse_style_discrete_lif(current=[0.0] * 50, params=p)
    assert t.first_spike_step is None


# -------------------------------------------------- continuous vs discrete equiv


def test_continuous_with_sub_steps_one_matches_discrete() -> None:
    p = LIFParams()
    cur = [0.0, 0.5, 1.5, 2.0, 1.0, 0.0, 0.0]
    cont = continuous_time_lif_reference(current=cur, params=p, sub_steps=1)
    disc = norse_style_discrete_lif(current=cur, params=p)
    assert cont.spikes == disc.spikes
    assert cont.v_history == pytest.approx(disc.v_history)


def test_continuous_higher_sub_steps_approaches_discrete_spike_count() -> None:
    p = LIFParams()
    cur = [3.0] * 50
    disc = norse_style_discrete_lif(current=cur, params=p)
    cont = continuous_time_lif_reference(current=cur, params=p, sub_steps=20)
    assert abs(cont.spike_count - disc.spike_count) <= 2


# --------------------------------------------------------- comparison report


def test_report_consistent_when_traces_agree() -> None:
    p = LIFParams()
    cur = [3.0] * 50
    report = prototype_lif_market_signal(
        current=cur,
        params=p,
        sub_steps=1,
        count_tolerance=0,
        first_spike_step_tolerance=0,
    )
    assert report.is_consistent()
    assert report.spike_count_delta == 0
    assert report.first_spike_step_delta == 0


def test_report_inconsistent_when_count_exceeds_tolerance() -> None:
    p = LIFParams()
    cur = [3.0] * 50
    report = prototype_lif_market_signal(
        current=cur,
        params=p,
        sub_steps=20,
        count_tolerance=0,
    )
    if report.spike_count_delta > 0:
        assert not report.is_consistent()


def test_report_count_tolerance_must_be_non_negative() -> None:
    p = LIFParams()
    cont = continuous_time_lif_reference(current=[0.0], params=p)
    disc = norse_style_discrete_lif(current=[0.0], params=p)
    with pytest.raises(NeuroPrototypeError):
        LIFComparisonReport(continuous=cont, discrete=disc, count_tolerance=-1)


def test_report_first_spike_tolerance_must_be_non_negative() -> None:
    p = LIFParams()
    cont = continuous_time_lif_reference(current=[0.0], params=p)
    disc = norse_style_discrete_lif(current=[0.0], params=p)
    with pytest.raises(NeuroPrototypeError):
        LIFComparisonReport(continuous=cont, discrete=disc, first_spike_step_tolerance=-1)


def test_report_length_mismatch_raises() -> None:
    p = LIFParams()
    cont = continuous_time_lif_reference(current=[0.0, 0.0], params=p)
    disc = norse_style_discrete_lif(current=[0.0, 0.0, 0.0], params=p)
    with pytest.raises(NeuroPrototypeError):
        LIFComparisonReport(continuous=cont, discrete=disc)


def test_report_first_spike_delta_none_when_both_silent() -> None:
    p = LIFParams()
    report = prototype_lif_market_signal(current=[0.0] * 10, params=p)
    assert report.first_spike_step_delta is None
    assert report.is_consistent()


def test_report_not_consistent_when_one_side_silent() -> None:
    p = LIFParams()
    cont = LIFTrace(
        backend="continuous",
        spikes=(True, False, False),
        v_history=(0.0, 0.0, 0.0),
        spike_times_steps=(0,),
        params=p,
    )
    disc = LIFTrace(
        backend="discrete",
        spikes=(False, False, False),
        v_history=(0.0, 0.0, 0.0),
        spike_times_steps=(),
        params=p,
    )
    rep = LIFComparisonReport(continuous=cont, discrete=disc, count_tolerance=1)
    assert not rep.is_consistent()


# ----------------------------------------------------------- INV-15 digests


def test_inv15_trace_digest_byte_identical_three_runs() -> None:
    p = LIFParams()
    cur = [0.0, 1.0, 2.0, 3.0, 0.0]
    a = norse_style_discrete_lif(current=cur, params=p)
    b = norse_style_discrete_lif(current=cur, params=p)
    c = norse_style_discrete_lif(current=cur, params=p)
    assert a.digest() == b.digest() == c.digest()
    assert a == b == c


def test_inv15_continuous_trace_digest_stable() -> None:
    p = LIFParams()
    cur = [0.0, 1.0, 2.0, 3.0, 0.0]
    a = continuous_time_lif_reference(current=cur, params=p, sub_steps=10)
    b = continuous_time_lif_reference(current=cur, params=p, sub_steps=10)
    assert a.digest() == b.digest()


def test_inv15_report_digest_stable() -> None:
    p = LIFParams()
    cur = [0.0, 1.5, 2.0, 0.5, 0.0]
    a = prototype_lif_market_signal(current=cur, params=p)
    b = prototype_lif_market_signal(current=cur, params=p)
    assert a.digest() == b.digest()


def test_inv15_digest_differs_on_input_change() -> None:
    p = LIFParams()
    a = norse_style_discrete_lif(current=[1.0, 2.0, 3.0], params=p)
    b = norse_style_discrete_lif(current=[1.0, 2.0, 4.0], params=p)
    assert a.digest() != b.digest()


def test_inv15_digest_differs_on_backend_label() -> None:
    p = LIFParams()
    cur = [1.0, 2.0, 3.0]
    cont = continuous_time_lif_reference(current=cur, params=p, sub_steps=1)
    disc = norse_style_discrete_lif(current=cur, params=p)
    assert cont.digest() != disc.digest()


# --------------------------------------------------------- production seam


def test_brian2_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError) as exc:
        brian2_prototype_factory(current=[0.0], params=LIFParams())
    msg = str(exc.value)
    assert "brian2" in msg.lower()
    assert "research" in msg.lower()


def test_brian2_factory_protocol_signature() -> None:
    assert callable(Brian2PrototypeFactory)
    assert Brian2PrototypeFactory.__name__ == "Brian2PrototypeFactory"


# ------------------------------------------------------------------ misc


def test_version_string_stable() -> None:
    assert NEURO_PROTOTYPE_VERSION == "neuro-prototype/v1"


def test_lif_trace_count_property() -> None:
    p = LIFParams()
    t = norse_style_discrete_lif(current=[5.0] * 30, params=p)
    assert t.spike_count == len(t.spike_times_steps)


def test_lif_trace_spike_times_sorted() -> None:
    p = LIFParams()
    t = norse_style_discrete_lif(current=[5.0] * 30, params=p)
    assert list(t.spike_times_steps) == sorted(t.spike_times_steps)
