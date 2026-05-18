"""Tests for ``simulation/strategy_arena/kill_underperformers.py`` (A-03.2).

Covers:

* module metadata + AST authority pins (no DEAP / clock / engine
  cross-imports / random / os);
* :func:`build_demotion_recommendations` argument validation;
* tournament-result → demotion-recommendation translation semantics;
* INV-15 byte-identical replay (3 runs over same arena output produce
  identical recommendation tuples);
* OFFLINE_ONLY tier (DEAP not in ``sys.modules`` after import);
* :class:`DemotionRecommendation` shape conformance + frozen + slotted;
* B28 authority symmetry: simulation tier does NOT build PatchProposal.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib
import sys

import pytest

from core.contracts.simulation import RealitySummary
from simulation.strategy_arena import kill_underperformers as kill_module
from simulation.strategy_arena.arena import (
    Arena,
    ArenaConfig,
    Contestant,
    TournamentBracket,
    TournamentResult,
)
from simulation.strategy_arena.kill_underperformers import (
    DEMOTION_KIND,
    DEMOTION_TOUCHPOINT,
    MAX_KILL_BATCH,
    MAX_RATIONALE_LEN,
    NEW_PIP_DEPENDENCIES,
    PROPOSAL_SOURCE,
    DemotionRecommendation,
    KillUnderperformersInputError,
    build_demotion_recommendations,
)

KILL_PATH = pathlib.Path(kill_module.__file__)
KILL_TREE = ast.parse(KILL_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_summary(
    *,
    scenario_id: str = "scn-1",
    n: int = 4,
    pnl_mean: float = 0.0,
    drawdown: float = 0.0,
) -> RealitySummary:
    return RealitySummary(
        scenario_id=scenario_id,
        n_realities=n,
        pnl_mean_usd=pnl_mean,
        pnl_median_usd=pnl_mean,
        pnl_p05_usd=pnl_mean - 1.0,
        pnl_p95_usd=pnl_mean + 1.0,
        win_rate=0.5,
        max_drawdown_usd=drawdown,
    )


def _make_contestant(
    sid: str,
    pnl_mean: float = 0.0,
    drawdown: float = 0.0,
) -> Contestant:
    return Contestant(
        strategy_id=sid,
        summary=_make_summary(pnl_mean=pnl_mean, drawdown=drawdown),
    )


def _make_result(
    *,
    n: int = 6,
    survivor_count: int = 3,
    seed: int = 42,
    ts_ns: int = 1_700_000_000_000_000_000,
    arena_id: str = "arena-test",
    elitism_count: int = 1,
) -> TournamentResult:
    contestants = [
        _make_contestant(
            f"strat-{i:03d}",
            pnl_mean=float(i),
            drawdown=0.0,
        )
        for i in range(n)
    ]
    config = ArenaConfig(
        arena_id=arena_id,
        tournament_size=2,
        n_winners=max(1, survivor_count - elitism_count),
        elitism_count=elitism_count,
        drawdown_weight=0.0,
    )
    arena = Arena()
    return arena.run(
        contestants=contestants,
        config=config,
        seed=seed,
        ts_ns=ts_ns,
    )


# ----------------------------------------------------------------------
# Module metadata (8)
# ----------------------------------------------------------------------


def test_new_pip_dependencies_empty():
    assert NEW_PIP_DEPENDENCIES == ()


def test_new_pip_dependencies_is_tuple():
    assert isinstance(NEW_PIP_DEPENDENCIES, tuple)


def test_proposal_source_stable():
    assert PROPOSAL_SOURCE == "simulation.strategy_arena.kill_underperformers"


def test_demotion_kind_stable():
    assert DEMOTION_KIND == "STRATEGY_DEMOTED"


def test_demotion_touchpoint_stable():
    assert DEMOTION_TOUCHPOINT == "strategy_registry.demoted"


def test_max_kill_batch_finite_positive():
    assert isinstance(MAX_KILL_BATCH, int)
    assert MAX_KILL_BATCH > 0


def test_max_rationale_len_finite_positive():
    assert isinstance(MAX_RATIONALE_LEN, int)
    assert MAX_RATIONALE_LEN > 0


def test_module_has_adapted_from_header():
    src = KILL_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: deap/algorithms.py" in src


# ----------------------------------------------------------------------
# AST authority pins (12)
# ----------------------------------------------------------------------


def _imported_module_names() -> set[str]:
    names: set[str] = set()
    for node in ast.walk(KILL_TREE):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_no_deap_import():
    assert "deap" not in _imported_module_names()


def test_no_numpy_import():
    assert "numpy" not in _imported_module_names()


def test_no_torch_import():
    assert "torch" not in _imported_module_names()


def test_no_random_import():
    assert "random" not in _imported_module_names()


def test_no_os_import():
    assert "os" not in _imported_module_names()


def test_no_time_import():
    assert "time" not in _imported_module_names()


def test_no_datetime_import():
    assert "datetime" not in _imported_module_names()


def test_no_engine_cross_imports():
    forbidden = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
    }
    assert _imported_module_names().isdisjoint(forbidden)


def test_no_logging_or_print():
    src = KILL_PATH.read_text(encoding="utf-8")
    assert "import logging" not in src
    assert "\nprint(" not in src


def test_no_clock_text():
    src = KILL_PATH.read_text(encoding="utf-8")
    for token in (
        "time.time(",
        "time.monotonic(",
        "time.perf_counter(",
        "time.time_ns(",
        "datetime.now(",
        "datetime.utcnow(",
    ):
        assert token not in src, f"forbidden clock token: {token}"


def test_b28_authority_symmetry_no_patchproposal():
    """B28: simulation tier MUST NOT construct PatchProposal.

    Mentions of the symbol in docstrings/comments are allowed (they
    explain the architectural constraint). Construction calls and
    imports are not.
    """
    # No import-from core.contracts.learning at AST level.
    for node in ast.walk(KILL_TREE):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "core.contracts.learning", (
                "kill_underperformers must NOT import from "
                "core.contracts.learning (B28 / HARDEN-06)"
            )
        # No PatchProposal(...) call expressions.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                assert func.id != "PatchProposal", (
                    "kill_underperformers must NOT construct PatchProposal"
                )
            if isinstance(func, ast.Attribute):
                assert func.attr != "PatchProposal", (
                    "kill_underperformers must NOT construct PatchProposal"
                )


def test_no_evolution_engine_import():
    """A-03.2 stays in simulation tier — never reach into evolution_engine."""
    assert "evolution_engine" not in _imported_module_names()


# ----------------------------------------------------------------------
# build_demotion_recommendations argument validation (12)
# ----------------------------------------------------------------------


def test_result_must_be_tournament_result():
    with pytest.raises(KillUnderperformersInputError):
        build_demotion_recommendations(result="not-a-result", ts_ns=1)  # type: ignore[arg-type]


def test_result_cannot_be_none():
    with pytest.raises(KillUnderperformersInputError):
        build_demotion_recommendations(result=None, ts_ns=1)  # type: ignore[arg-type]


def test_ts_ns_must_be_int():
    result = _make_result()
    with pytest.raises(KillUnderperformersInputError):
        build_demotion_recommendations(result=result, ts_ns="1")  # type: ignore[arg-type]


def test_ts_ns_cannot_be_bool():
    result = _make_result()
    with pytest.raises(KillUnderperformersInputError):
        build_demotion_recommendations(result=result, ts_ns=True)  # type: ignore[arg-type]


def test_ts_ns_cannot_be_negative():
    result = _make_result()
    with pytest.raises(KillUnderperformersInputError):
        build_demotion_recommendations(result=result, ts_ns=-1)


def test_ts_ns_zero_is_allowed():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=0)
    for r in recs:
        assert r.ts_ns == 0


def test_extra_meta_must_be_mapping_or_none():
    result = _make_result()
    with pytest.raises(KillUnderperformersInputError):
        build_demotion_recommendations(
            result=result,
            ts_ns=1,
            extra_meta=[("k", "v")],  # type: ignore[arg-type]
        )


def test_extra_meta_keys_must_be_str():
    result = _make_result()
    with pytest.raises(KillUnderperformersInputError):
        build_demotion_recommendations(
            result=result,
            ts_ns=1,
            extra_meta={1: "v"},  # type: ignore[dict-item]
        )


def test_extra_meta_values_must_be_str():
    result = _make_result()
    with pytest.raises(KillUnderperformersInputError):
        build_demotion_recommendations(
            result=result,
            ts_ns=1,
            extra_meta={"k": 1},  # type: ignore[dict-item]
        )


def test_extra_meta_keys_cannot_be_empty():
    result = _make_result()
    with pytest.raises(KillUnderperformersInputError):
        build_demotion_recommendations(result=result, ts_ns=1, extra_meta={"": "v"})


def test_extra_meta_cannot_collide_with_reserved():
    result = _make_result()
    for key in ("kind", "arena_id", "arena_digest", "pnl_mean_usd", "max_drawdown_usd"):
        with pytest.raises(KillUnderperformersInputError):
            build_demotion_recommendations(result=result, ts_ns=1, extra_meta={key: "x"})


def test_extra_meta_none_is_allowed():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1, extra_meta=None)
    assert isinstance(recs, tuple)


# ----------------------------------------------------------------------
# Translation semantics (12)
# ----------------------------------------------------------------------


def test_returns_tuple():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    assert isinstance(recs, tuple)


def test_every_entry_is_demotion_recommendation():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert isinstance(r, DemotionRecommendation)


def test_eliminated_count_matches_contestants_minus_survivors():
    result = _make_result(n=6, survivor_count=3)
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    assert len(recs) == len(result.contestants) - len(result.survivors)


def test_no_recommendation_for_any_survivor():
    result = _make_result(n=6, survivor_count=3)
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    survivors = set(result.survivors)
    for r in recs:
        assert r.strategy_id not in survivors


def test_every_eliminated_id_appears_exactly_once():
    result = _make_result(n=6, survivor_count=3)
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    ids = [r.strategy_id for r in recs]
    assert len(ids) == len(set(ids))
    survivors = set(result.survivors)
    eliminated = {c.strategy_id for c in result.contestants} - survivors
    assert set(ids) == eliminated


def test_recommendation_order_matches_contestant_order():
    result = _make_result(n=6, survivor_count=3)
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    survivors = set(result.survivors)
    expected = [c.strategy_id for c in result.contestants if c.strategy_id not in survivors]
    actual = [r.strategy_id for r in recs]
    assert actual == expected


def test_source_string_is_proposal_source():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.source == PROPOSAL_SOURCE


def test_kind_field_is_strategy_demoted():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.kind == DEMOTION_KIND


def test_touchpoint_field_is_demotion_touchpoint():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.touchpoint == DEMOTION_TOUCHPOINT


def test_arena_fields_propagate():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.arena_id == result.arena_id
        assert r.arena_digest == result.arena_digest


def test_pnl_and_drawdown_propagate():
    contestants = [
        _make_contestant("strat-a", pnl_mean=-2.5, drawdown=10.0),
        _make_contestant("strat-b", pnl_mean=5.0, drawdown=1.0),
    ]
    config = ArenaConfig(
        arena_id="arena-pnl",
        tournament_size=2,
        n_winners=1,
        elitism_count=0,
        drawdown_weight=0.0,
    )
    arena = Arena()
    result = arena.run(contestants=contestants, config=config, seed=1, ts_ns=1)
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    by_id = {r.strategy_id: r for r in recs}
    if "strat-a" in by_id:
        assert by_id["strat-a"].pnl_mean_usd == -2.5
        assert by_id["strat-a"].max_drawdown_usd == 10.0


def test_extra_meta_propagates():
    result = _make_result()
    recs = build_demotion_recommendations(
        result=result,
        ts_ns=1,
        extra_meta={"reason": "low-fitness", "round": "3"},
    )
    for r in recs:
        meta_dict = dict(r.meta)
        assert meta_dict["reason"] == "low-fitness"
        assert meta_dict["round"] == "3"


# ----------------------------------------------------------------------
# Output shape (8)
# ----------------------------------------------------------------------


def test_recommendation_id_format():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.recommendation_id.startswith("demote-")
        # 16 hex chars after the prefix (BLAKE2b-8)
        suffix = r.recommendation_id[len("demote-") :]
        assert len(suffix) == 16
        int(suffix, 16)  # hex check


def test_recommendation_id_uniqueness_within_call():
    result = _make_result(n=6, survivor_count=3)
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    ids = [r.recommendation_id for r in recs]
    assert len(ids) == len(set(ids))


def test_ts_ns_propagates():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1234567890)
    for r in recs:
        assert r.ts_ns == 1234567890


def test_rationale_within_cap():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert len(r.rationale) <= MAX_RATIONALE_LEN


def test_rationale_non_empty():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.rationale


def test_meta_is_tuple():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1, extra_meta={"a": "1"})
    for r in recs:
        assert isinstance(r.meta, tuple)


def test_meta_empty_when_extra_none():
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.meta == ()


def test_recommendation_is_frozen_and_slotted():
    fields = dataclasses.fields(DemotionRecommendation)
    assert fields  # non-empty
    rec = DemotionRecommendation(
        ts_ns=1,
        recommendation_id="demote-aaaaaaaaaaaaaaaa",
        source=PROPOSAL_SOURCE,
        kind=DEMOTION_KIND,
        arena_id="a",
        arena_digest="0" * 16,
        strategy_id="s",
        touchpoint=DEMOTION_TOUCHPOINT,
        pnl_mean_usd=0.0,
        max_drawdown_usd=0.0,
        rationale="x",
        meta=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.ts_ns = 2  # type: ignore[misc]
    # __slots__ → no __dict__
    assert not hasattr(rec, "__dict__")


# ----------------------------------------------------------------------
# DemotionRecommendation post-init validation (6)
# ----------------------------------------------------------------------


def _valid_kwargs() -> dict:
    return dict(
        ts_ns=1,
        recommendation_id="demote-aaaaaaaaaaaaaaaa",
        source=PROPOSAL_SOURCE,
        kind=DEMOTION_KIND,
        arena_id="a",
        arena_digest="0" * 16,
        strategy_id="s",
        touchpoint=DEMOTION_TOUCHPOINT,
        pnl_mean_usd=0.0,
        max_drawdown_usd=0.0,
        rationale="x",
        meta=(),
    )


def test_post_init_rejects_negative_ts_ns():
    kwargs = _valid_kwargs()
    kwargs["ts_ns"] = -1
    with pytest.raises(KillUnderperformersInputError):
        DemotionRecommendation(**kwargs)


def test_post_init_rejects_empty_arena_id():
    kwargs = _valid_kwargs()
    kwargs["arena_id"] = ""
    with pytest.raises(KillUnderperformersInputError):
        DemotionRecommendation(**kwargs)


def test_post_init_rejects_int_pnl():
    kwargs = _valid_kwargs()
    kwargs["pnl_mean_usd"] = 0
    with pytest.raises(KillUnderperformersInputError):
        DemotionRecommendation(**kwargs)


def test_post_init_rejects_meta_list():
    kwargs = _valid_kwargs()
    kwargs["meta"] = [("k", "v")]
    with pytest.raises(KillUnderperformersInputError):
        DemotionRecommendation(**kwargs)


def test_post_init_rejects_meta_bad_pair_shape():
    kwargs = _valid_kwargs()
    kwargs["meta"] = (("k",),)
    with pytest.raises(KillUnderperformersInputError):
        DemotionRecommendation(**kwargs)


def test_post_init_rejects_duplicate_meta_keys():
    kwargs = _valid_kwargs()
    kwargs["meta"] = (("k", "1"), ("k", "2"))
    with pytest.raises(KillUnderperformersInputError):
        DemotionRecommendation(**kwargs)


# ----------------------------------------------------------------------
# INV-15 byte-identical replay (6)
# ----------------------------------------------------------------------


def test_three_runs_byte_identical():
    result = _make_result()
    a = build_demotion_recommendations(result=result, ts_ns=1)
    b = build_demotion_recommendations(result=result, ts_ns=1)
    c = build_demotion_recommendations(result=result, ts_ns=1)
    assert a == b == c


def test_same_arena_different_ts_ns_different_recommendation_id():
    result = _make_result()
    a = build_demotion_recommendations(result=result, ts_ns=1)
    b = build_demotion_recommendations(result=result, ts_ns=2)
    if a and b:
        assert a[0].recommendation_id != b[0].recommendation_id


def test_same_arena_different_ts_ns_same_target():
    """Eliminated set is a function of the arena, not ``ts_ns``."""
    result = _make_result()
    a = build_demotion_recommendations(result=result, ts_ns=1)
    b = build_demotion_recommendations(result=result, ts_ns=2)
    a_ids = [r.strategy_id for r in a]
    b_ids = [r.strategy_id for r in b]
    assert a_ids == b_ids


def test_extra_meta_key_order_does_not_matter():
    """Reorder extra_meta keys → identical recommendation tuple."""
    result = _make_result()
    a = build_demotion_recommendations(result=result, ts_ns=1, extra_meta={"a": "1", "b": "2"})
    b = build_demotion_recommendations(result=result, ts_ns=1, extra_meta={"b": "2", "a": "1"})
    assert a == b


def test_different_arena_digest_different_recommendation_id():
    """Different arena outputs must produce different ids."""
    r1 = _make_result(seed=1)
    r2 = _make_result(seed=2)
    a = build_demotion_recommendations(result=r1, ts_ns=1)
    b = build_demotion_recommendations(result=r2, ts_ns=1)
    if a and b:
        a_ids = {r.recommendation_id for r in a}
        b_ids = {r.recommendation_id for r in b}
        assert a_ids != b_ids


def test_recommendation_id_hex_and_short():
    """Recommendation id is 23 chars total (`demote-` + 16 hex)."""
    result = _make_result()
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert len(r.recommendation_id) == len("demote-") + 16


# ----------------------------------------------------------------------
# Frozen + slotted contract (1)
# ----------------------------------------------------------------------


def test_kill_underperformers_input_error_subclass():
    assert issubclass(KillUnderperformersInputError, ValueError)


# ----------------------------------------------------------------------
# OFFLINE_ONLY tier import audit (1)
# ----------------------------------------------------------------------


def test_module_imports_clean_without_deap():
    # The kill_underperformers module has been imported by this test
    # file's top-level imports. Confirm that did NOT pull deap into
    # sys.modules.
    assert "deap" not in sys.modules


# ----------------------------------------------------------------------
# Boundary cases (3)
# ----------------------------------------------------------------------


def test_no_eliminations_returns_empty():
    """When all contestants survive, no recommendations are emitted."""
    contestants = [
        _make_contestant("strat-001", pnl_mean=1.0),
        _make_contestant("strat-002", pnl_mean=2.0),
    ]
    bracket = TournamentBracket(
        bracket_id=0,
        candidate_ids=("strat-001", "strat-002"),
        winner_id="strat-002",
        winner_fitness=2.0,
    )
    result = TournamentResult(
        arena_id="arena-empty",
        ts_ns=1,
        seed=1,
        contestants=tuple(contestants),
        elites=("strat-002", "strat-001"),
        tournament_winners=("strat-002",),
        survivors=("strat-002", "strat-001"),
        brackets=(bracket,),
        arena_digest="0" * 16,
    )
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    assert recs == ()


def test_eliminated_strict_partition():
    """Eliminated set = contestants \\ survivors, exactly."""
    contestants = [_make_contestant(f"strat-{i:03d}", pnl_mean=float(i)) for i in range(5)]
    config = ArenaConfig(
        arena_id="arena-strict-part",
        tournament_size=2,
        n_winners=1,
        elitism_count=1,
        drawdown_weight=0.0,
    )
    arena = Arena()
    result = arena.run(
        contestants=contestants,
        config=config,
        seed=1,
        ts_ns=1,
    )
    recs = build_demotion_recommendations(result=result, ts_ns=1)
    assert len(recs) == len(result.contestants) - len(result.survivors)
    survivors = set(result.survivors)
    for r in recs:
        assert r.strategy_id not in survivors


def test_callable_signature_is_keyword_only():
    """Public function must take only keyword arguments — INV-08 ergonomics."""
    sig = inspect.signature(build_demotion_recommendations)
    for param in sig.parameters.values():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"build_demotion_recommendations.{param.name} must be keyword-only"
        )
