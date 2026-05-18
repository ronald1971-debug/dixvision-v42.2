"""Tests for ``simulation/strategy_arena/promotion_engine.py`` (A-03.3).

Covers:

* module metadata + AST authority pins (no DEAP / clock / engine
  cross-imports / random / os);
* :func:`build_promotion_recommendations` argument validation;
* tournament-result → promotion-recommendation translation semantics;
* role classification (ELITE / TOURNAMENT_WINNER / ELITE_AND_WINNER);
* INV-15 byte-identical replay (3 runs over same arena output produce
  identical recommendation tuples);
* OFFLINE_ONLY tier (DEAP not in ``sys.modules`` after import);
* :class:`PromotionRecommendation` shape conformance + frozen + slotted;
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
from simulation.strategy_arena import promotion_engine as promote_module
from simulation.strategy_arena.arena import (
    Arena,
    ArenaConfig,
    Contestant,
    TournamentBracket,
    TournamentResult,
)
from simulation.strategy_arena.promotion_engine import (
    MAX_PROMOTE_BATCH,
    MAX_RATIONALE_LEN,
    NEW_PIP_DEPENDENCIES,
    PROMOTION_KIND,
    PROMOTION_TOUCHPOINT,
    PROPOSAL_SOURCE,
    ROLE_BOTH,
    ROLE_ELITE,
    ROLE_WINNER,
    PromotionEngineInputError,
    PromotionRecommendation,
    build_promotion_recommendations,
)

PROMOTE_PATH = pathlib.Path(promote_module.__file__)
PROMOTE_TREE = ast.parse(PROMOTE_PATH.read_text(encoding="utf-8"))


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
# Module metadata (10)
# ----------------------------------------------------------------------


def test_new_pip_dependencies_empty():
    assert NEW_PIP_DEPENDENCIES == ()


def test_new_pip_dependencies_is_tuple():
    assert isinstance(NEW_PIP_DEPENDENCIES, tuple)


def test_proposal_source_stable():
    assert PROPOSAL_SOURCE == "simulation.strategy_arena.promotion_engine"


def test_promotion_kind_stable():
    assert PROMOTION_KIND == "STRATEGY_PROMOTED"


def test_promotion_touchpoint_stable():
    assert PROMOTION_TOUCHPOINT == "strategy_registry.promoted"


def test_role_constants_stable():
    assert ROLE_ELITE == "ELITE"
    assert ROLE_WINNER == "TOURNAMENT_WINNER"
    assert ROLE_BOTH == "ELITE_AND_WINNER"


def test_max_promote_batch_finite_positive():
    assert isinstance(MAX_PROMOTE_BATCH, int)
    assert MAX_PROMOTE_BATCH > 0


def test_max_rationale_len_finite_positive():
    assert isinstance(MAX_RATIONALE_LEN, int)
    assert MAX_RATIONALE_LEN > 0


def test_module_has_adapted_from_header():
    src = PROMOTE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: deap/algorithms.py" in src


def test_module_references_governance_approval_gate():
    """Spec line 881: 'Promotion must pass through governance approval gate'."""
    src = PROMOTE_PATH.read_text(encoding="utf-8")
    assert "governance approval gate" in src


# ----------------------------------------------------------------------
# AST authority pins (12)
# ----------------------------------------------------------------------


def _imported_module_names() -> set[str]:
    names: set[str] = set()
    for node in ast.walk(PROMOTE_TREE):
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
    src = PROMOTE_PATH.read_text(encoding="utf-8")
    assert "import logging" not in src
    assert "\nprint(" not in src


def test_no_clock_text():
    src = PROMOTE_PATH.read_text(encoding="utf-8")
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
    for node in ast.walk(PROMOTE_TREE):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "core.contracts.learning", (
                "promotion_engine must NOT import from core.contracts.learning (B28 / HARDEN-06)"
            )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                assert func.id != "PatchProposal", (
                    "promotion_engine must NOT construct PatchProposal"
                )
            if isinstance(func, ast.Attribute):
                assert func.attr != "PatchProposal", (
                    "promotion_engine must NOT construct PatchProposal"
                )


def test_no_evolution_engine_import():
    """A-03.3 stays in simulation tier — never reach into evolution_engine."""
    assert "evolution_engine" not in _imported_module_names()


# ----------------------------------------------------------------------
# build_promotion_recommendations argument validation (12)
# ----------------------------------------------------------------------


def test_result_must_be_tournament_result():
    with pytest.raises(PromotionEngineInputError):
        build_promotion_recommendations(result="not-a-result", ts_ns=1)  # type: ignore[arg-type]


def test_result_cannot_be_none():
    with pytest.raises(PromotionEngineInputError):
        build_promotion_recommendations(result=None, ts_ns=1)  # type: ignore[arg-type]


def test_ts_ns_must_be_int():
    result = _make_result()
    with pytest.raises(PromotionEngineInputError):
        build_promotion_recommendations(result=result, ts_ns="1")  # type: ignore[arg-type]


def test_ts_ns_cannot_be_bool():
    result = _make_result()
    with pytest.raises(PromotionEngineInputError):
        build_promotion_recommendations(result=result, ts_ns=True)  # type: ignore[arg-type]


def test_ts_ns_cannot_be_negative():
    result = _make_result()
    with pytest.raises(PromotionEngineInputError):
        build_promotion_recommendations(result=result, ts_ns=-1)


def test_ts_ns_zero_is_allowed():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=0)
    for r in recs:
        assert r.ts_ns == 0


def test_extra_meta_must_be_mapping_or_none():
    result = _make_result()
    with pytest.raises(PromotionEngineInputError):
        build_promotion_recommendations(
            result=result,
            ts_ns=1,
            extra_meta=[("k", "v")],  # type: ignore[arg-type]
        )


def test_extra_meta_keys_must_be_str():
    result = _make_result()
    with pytest.raises(PromotionEngineInputError):
        build_promotion_recommendations(
            result=result,
            ts_ns=1,
            extra_meta={1: "v"},  # type: ignore[dict-item]
        )


def test_extra_meta_values_must_be_str():
    result = _make_result()
    with pytest.raises(PromotionEngineInputError):
        build_promotion_recommendations(
            result=result,
            ts_ns=1,
            extra_meta={"k": 1},  # type: ignore[dict-item]
        )


def test_extra_meta_keys_cannot_be_empty():
    result = _make_result()
    with pytest.raises(PromotionEngineInputError):
        build_promotion_recommendations(result=result, ts_ns=1, extra_meta={"": "v"})


def test_extra_meta_cannot_collide_with_reserved():
    result = _make_result()
    for key in (
        "kind",
        "role",
        "arena_id",
        "arena_digest",
        "pnl_mean_usd",
        "max_drawdown_usd",
    ):
        with pytest.raises(PromotionEngineInputError):
            build_promotion_recommendations(result=result, ts_ns=1, extra_meta={key: "x"})


def test_extra_meta_none_is_allowed():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1, extra_meta=None)
    assert isinstance(recs, tuple)


# ----------------------------------------------------------------------
# Translation semantics (12)
# ----------------------------------------------------------------------


def test_returns_tuple():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    assert isinstance(recs, tuple)


def test_every_entry_is_promotion_recommendation():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert isinstance(r, PromotionRecommendation)


def test_recommendation_count_matches_survivors():
    result = _make_result(n=6, survivor_count=3)
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    assert len(recs) == len(result.survivors)


def test_no_recommendation_for_eliminated():
    result = _make_result(n=6, survivor_count=3)
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    survivors = set(result.survivors)
    eliminated = {c.strategy_id for c in result.contestants} - survivors
    rec_ids = {r.strategy_id for r in recs}
    assert rec_ids.isdisjoint(eliminated)


def test_every_survivor_id_appears_exactly_once():
    result = _make_result(n=6, survivor_count=3)
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    ids = [r.strategy_id for r in recs]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(result.survivors)


def test_recommendation_order_matches_survivors_order():
    result = _make_result(n=6, survivor_count=3)
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    assert [r.strategy_id for r in recs] == list(result.survivors)


def test_source_string_is_proposal_source():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.source == PROPOSAL_SOURCE


def test_kind_field_is_strategy_promoted():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.kind == PROMOTION_KIND


def test_touchpoint_field_is_promotion_touchpoint():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.touchpoint == PROMOTION_TOUCHPOINT


def test_arena_fields_propagate():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.arena_id == result.arena_id
        assert r.arena_digest == result.arena_digest


def test_pnl_and_drawdown_propagate():
    contestants = [
        _make_contestant("strat-a", pnl_mean=5.0, drawdown=1.0),
        _make_contestant("strat-b", pnl_mean=-2.5, drawdown=10.0),
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
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    by_id = {r.strategy_id: r for r in recs}
    if "strat-a" in by_id:
        assert by_id["strat-a"].pnl_mean_usd == 5.0
        assert by_id["strat-a"].max_drawdown_usd == 1.0


def test_extra_meta_propagates():
    result = _make_result()
    recs = build_promotion_recommendations(
        result=result,
        ts_ns=1,
        extra_meta={"reason": "high-fitness", "round": "3"},
    )
    for r in recs:
        meta_dict = dict(r.meta)
        assert meta_dict["reason"] == "high-fitness"
        assert meta_dict["round"] == "3"


# ----------------------------------------------------------------------
# Role classification (5)
# ----------------------------------------------------------------------


def test_role_is_one_of_three_constants():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.role in {ROLE_ELITE, ROLE_WINNER, ROLE_BOTH}


def test_pure_elite_role():
    """Survivor in elites but not in tournament_winners → ELITE."""
    contestants = [_make_contestant(f"strat-{i:03d}", pnl_mean=float(i)) for i in range(4)]
    config = ArenaConfig(
        arena_id="arena-elite",
        tournament_size=2,
        n_winners=1,
        elitism_count=2,
        drawdown_weight=0.0,
    )
    arena = Arena()
    result = arena.run(contestants=contestants, config=config, seed=99, ts_ns=1)
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    by_id = {r.strategy_id: r for r in recs}
    elites_only = set(result.elites) - set(result.tournament_winners)
    for sid in elites_only:
        if sid in by_id:
            assert by_id[sid].role == ROLE_ELITE


def test_pure_winner_role():
    """Survivor in tournament_winners but not in elites → TOURNAMENT_WINNER."""
    result = _make_result(n=8, survivor_count=4)
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    by_id = {r.strategy_id: r for r in recs}
    winners_only = set(result.tournament_winners) - set(result.elites)
    for sid in winners_only:
        assert by_id[sid].role == ROLE_WINNER


def test_both_role_when_in_both_sets():
    """Survivor in both elites and tournament_winners → ELITE_AND_WINNER."""
    result = _make_result(n=8, survivor_count=4)
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    by_id = {r.strategy_id: r for r in recs}
    in_both = set(result.elites) & set(result.tournament_winners)
    for sid in in_both:
        assert by_id[sid].role == ROLE_BOTH


def test_role_appears_in_rationale():
    """Rationale string includes the role tag for traceability."""
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert f"role={r.role}" in r.rationale


# ----------------------------------------------------------------------
# Output shape (8)
# ----------------------------------------------------------------------


def test_recommendation_id_format():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.recommendation_id.startswith("promote-")
        suffix = r.recommendation_id[len("promote-") :]
        assert len(suffix) == 16
        int(suffix, 16)  # hex check


def test_recommendation_id_uniqueness_within_call():
    result = _make_result(n=6, survivor_count=3)
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    ids = [r.recommendation_id for r in recs]
    assert len(ids) == len(set(ids))


def test_ts_ns_propagates():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1234567890)
    for r in recs:
        assert r.ts_ns == 1234567890


def test_rationale_within_cap():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert len(r.rationale) <= MAX_RATIONALE_LEN


def test_rationale_non_empty():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.rationale


def test_meta_is_tuple():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1, extra_meta={"a": "1"})
    for r in recs:
        assert isinstance(r.meta, tuple)


def test_meta_empty_when_extra_none():
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert r.meta == ()


def test_recommendation_is_frozen_and_slotted():
    fields = dataclasses.fields(PromotionRecommendation)
    assert fields  # non-empty
    rec = PromotionRecommendation(
        ts_ns=1,
        recommendation_id="promote-aaaaaaaaaaaaaaaa",
        source=PROPOSAL_SOURCE,
        kind=PROMOTION_KIND,
        arena_id="a",
        arena_digest="0" * 16,
        strategy_id="s",
        role=ROLE_ELITE,
        touchpoint=PROMOTION_TOUCHPOINT,
        pnl_mean_usd=0.0,
        max_drawdown_usd=0.0,
        rationale="x",
        meta=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.ts_ns = 2  # type: ignore[misc]
    assert not hasattr(rec, "__dict__")


# ----------------------------------------------------------------------
# PromotionRecommendation post-init validation (7)
# ----------------------------------------------------------------------


def _valid_kwargs() -> dict:
    return dict(
        ts_ns=1,
        recommendation_id="promote-aaaaaaaaaaaaaaaa",
        source=PROPOSAL_SOURCE,
        kind=PROMOTION_KIND,
        arena_id="a",
        arena_digest="0" * 16,
        strategy_id="s",
        role=ROLE_ELITE,
        touchpoint=PROMOTION_TOUCHPOINT,
        pnl_mean_usd=0.0,
        max_drawdown_usd=0.0,
        rationale="x",
        meta=(),
    )


def test_post_init_rejects_negative_ts_ns():
    kwargs = _valid_kwargs()
    kwargs["ts_ns"] = -1
    with pytest.raises(PromotionEngineInputError):
        PromotionRecommendation(**kwargs)


def test_post_init_rejects_empty_arena_id():
    kwargs = _valid_kwargs()
    kwargs["arena_id"] = ""
    with pytest.raises(PromotionEngineInputError):
        PromotionRecommendation(**kwargs)


def test_post_init_rejects_unknown_role():
    kwargs = _valid_kwargs()
    kwargs["role"] = "OUTSIDER"
    with pytest.raises(PromotionEngineInputError):
        PromotionRecommendation(**kwargs)


def test_post_init_rejects_int_pnl():
    kwargs = _valid_kwargs()
    kwargs["pnl_mean_usd"] = 0
    with pytest.raises(PromotionEngineInputError):
        PromotionRecommendation(**kwargs)


def test_post_init_rejects_meta_list():
    kwargs = _valid_kwargs()
    kwargs["meta"] = [("k", "v")]
    with pytest.raises(PromotionEngineInputError):
        PromotionRecommendation(**kwargs)


def test_post_init_rejects_meta_bad_pair_shape():
    kwargs = _valid_kwargs()
    kwargs["meta"] = (("k",),)
    with pytest.raises(PromotionEngineInputError):
        PromotionRecommendation(**kwargs)


def test_post_init_rejects_duplicate_meta_keys():
    kwargs = _valid_kwargs()
    kwargs["meta"] = (("k", "1"), ("k", "2"))
    with pytest.raises(PromotionEngineInputError):
        PromotionRecommendation(**kwargs)


# ----------------------------------------------------------------------
# INV-15 byte-identical replay (6)
# ----------------------------------------------------------------------


def test_three_runs_byte_identical():
    result = _make_result()
    a = build_promotion_recommendations(result=result, ts_ns=1)
    b = build_promotion_recommendations(result=result, ts_ns=1)
    c = build_promotion_recommendations(result=result, ts_ns=1)
    assert a == b == c


def test_same_arena_different_ts_ns_different_recommendation_id():
    result = _make_result()
    a = build_promotion_recommendations(result=result, ts_ns=1)
    b = build_promotion_recommendations(result=result, ts_ns=2)
    if a and b:
        assert a[0].recommendation_id != b[0].recommendation_id


def test_same_arena_different_ts_ns_same_target():
    """Surviving set is a function of the arena, not ``ts_ns``."""
    result = _make_result()
    a = build_promotion_recommendations(result=result, ts_ns=1)
    b = build_promotion_recommendations(result=result, ts_ns=2)
    a_ids = [r.strategy_id for r in a]
    b_ids = [r.strategy_id for r in b]
    assert a_ids == b_ids


def test_extra_meta_key_order_does_not_matter():
    """Reorder extra_meta keys → identical recommendation tuple."""
    result = _make_result()
    a = build_promotion_recommendations(result=result, ts_ns=1, extra_meta={"a": "1", "b": "2"})
    b = build_promotion_recommendations(result=result, ts_ns=1, extra_meta={"b": "2", "a": "1"})
    assert a == b


def test_different_arena_digest_different_recommendation_id():
    """Different arena outputs must produce different ids."""
    r1 = _make_result(seed=1)
    r2 = _make_result(seed=2)
    a = build_promotion_recommendations(result=r1, ts_ns=1)
    b = build_promotion_recommendations(result=r2, ts_ns=1)
    if a and b:
        a_ids = {r.recommendation_id for r in a}
        b_ids = {r.recommendation_id for r in b}
        assert a_ids != b_ids


def test_recommendation_id_hex_and_short():
    """Recommendation id is 24 chars total (`promote-` + 16 hex)."""
    result = _make_result()
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    for r in recs:
        assert len(r.recommendation_id) == len("promote-") + 16


# ----------------------------------------------------------------------
# Frozen + slotted contract (1)
# ----------------------------------------------------------------------


def test_promotion_engine_input_error_subclass():
    assert issubclass(PromotionEngineInputError, ValueError)


# ----------------------------------------------------------------------
# OFFLINE_ONLY tier import audit (1)
# ----------------------------------------------------------------------


def test_module_imports_clean_without_deap():
    # The promotion_engine module has been imported by this test
    # file's top-level imports. Confirm that did NOT pull deap into
    # sys.modules.
    assert "deap" not in sys.modules


# ----------------------------------------------------------------------
# Boundary cases (3)
# ----------------------------------------------------------------------


def test_no_survivors_returns_empty():
    """A hand-crafted TournamentResult with empty survivors → empty tuple."""
    contestants = [_make_contestant("strat-a", pnl_mean=1.0)]
    bracket = TournamentBracket(
        bracket_id=0,
        candidate_ids=("strat-a",),
        winner_id="strat-a",
        winner_fitness=1.0,
    )
    result = TournamentResult(
        arena_id="arena-empty",
        ts_ns=1,
        seed=1,
        contestants=tuple(contestants),
        elites=(),
        tournament_winners=(),
        survivors=(),
        brackets=(bracket,),
        arena_digest="0" * 16,
    )
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    assert recs == ()


def test_all_survive_when_n_winners_equals_population():
    """All contestants survive → one recommendation per contestant."""
    contestants = [_make_contestant(f"strat-{i:03d}", pnl_mean=float(i)) for i in range(3)]
    config = ArenaConfig(
        arena_id="arena-all-survive",
        tournament_size=2,
        n_winners=3,
        elitism_count=0,
        drawdown_weight=0.0,
    )
    arena = Arena()
    result = arena.run(contestants=contestants, config=config, seed=1, ts_ns=1)
    recs = build_promotion_recommendations(result=result, ts_ns=1)
    assert len(recs) == len(result.survivors)
    assert {r.strategy_id for r in recs} <= {c.strategy_id for c in contestants}


def test_callable_signature_is_keyword_only():
    """Public function must take only keyword arguments — INV-08 ergonomics."""
    sig = inspect.signature(build_promotion_recommendations)
    for param in sig.parameters.values():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"build_promotion_recommendations.{param.name} must be keyword-only"
        )
