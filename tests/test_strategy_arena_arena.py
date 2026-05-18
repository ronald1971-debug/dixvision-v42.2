"""Tests for ``simulation/strategy_arena/arena.py`` (A-03.1).

Covers DEAP-adapted tournament selection + elitism, INV-15 byte-
identical replay, OFFLINE_ONLY tier compliance, AST authority pins
(no DEAP / clock / engine cross-imports), frozen+slotted dataclass
validation, and tie-break determinism.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import sys

import pytest

from core.contracts.simulation import RealitySummary
from simulation.strategy_arena import arena as arena_module
from simulation.strategy_arena.arena import (
    DEFAULT_DRAWDOWN_WEIGHT,
    MAX_ARENA_ID_LEN,
    MAX_CONTESTANTS,
    MAX_ELITISM_COUNT,
    MAX_N_WINNERS,
    MAX_TOURNAMENT_SIZE,
    MIN_CONTESTANTS,
    MIN_TOURNAMENT_SIZE,
    NEW_PIP_DEPENDENCIES,
    PROPOSAL_SOURCE,
    Arena,
    ArenaConfig,
    ArenaConfigError,
    ArenaInputError,
    Contestant,
    TournamentBracket,
    TournamentResult,
)

# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def _make_summary(
    sid: str = "scenario-1",
    *,
    mean: float = 0.0,
    dd: float = 0.0,
    n: int = 10,
) -> RealitySummary:
    return RealitySummary(
        scenario_id=sid,
        n_realities=n,
        pnl_mean_usd=float(mean),
        pnl_median_usd=float(mean),
        pnl_p05_usd=float(mean) - 1.0,
        pnl_p95_usd=float(mean) + 1.0,
        win_rate=0.5,
        max_drawdown_usd=float(dd),
    )


def _make_contestants(n: int = 8) -> list[Contestant]:
    return [
        Contestant(
            strategy_id=f"strat-{i:03d}",
            summary=_make_summary(f"s{i}", mean=float(i * 10), dd=5.0),
        )
        for i in range(n)
    ]


def _module_source() -> str:
    src_path = pathlib.Path(inspect.getsourcefile(arena_module) or "")
    return src_path.read_text(encoding="utf-8")


def _module_ast() -> ast.Module:
    return ast.parse(_module_source())


def _iter_imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module)
    return out


# ----------------------------------------------------------------------
# Module metadata (8)
# ----------------------------------------------------------------------


def test_new_pip_dependencies_is_empty_tuple():
    assert NEW_PIP_DEPENDENCIES == ()


def test_proposal_source_is_stable_dotted_path():
    assert PROPOSAL_SOURCE == "simulation.strategy_arena.arena"


def test_min_contestants_is_two():
    assert MIN_CONTESTANTS == 2


def test_max_contestants_is_capped():
    assert 0 < MAX_CONTESTANTS <= 4096


def test_min_tournament_size_is_two():
    assert MIN_TOURNAMENT_SIZE == 2


def test_max_tournament_size_is_capped():
    assert MIN_TOURNAMENT_SIZE < MAX_TOURNAMENT_SIZE <= 256


def test_max_n_winners_and_elitism_capped():
    assert 0 < MAX_N_WINNERS <= 4096
    assert 0 < MAX_ELITISM_COUNT <= 1024


def test_default_drawdown_weight_is_zero():
    assert DEFAULT_DRAWDOWN_WEIGHT == 0.0


def test_max_arena_id_len_is_capped():
    assert 0 < MAX_ARENA_ID_LEN <= 1024


def test_adapted_from_header_present():
    src = _module_source()
    assert "# ADAPTED FROM: deap/tools/selection.py" in src
    assert "# ADAPTED FROM: deap/algorithms.py" in src


# ----------------------------------------------------------------------
# AST authority pins (10)
# ----------------------------------------------------------------------


def test_ast_no_deap_import():
    for name in _iter_imports(_module_ast()):
        assert "deap" not in name.lower(), f"forbidden deap import: {name}"


def test_ast_no_numpy_import():
    for name in _iter_imports(_module_ast()):
        assert name != "numpy" and not name.startswith("numpy.")


def test_ast_no_torch_import():
    for name in _iter_imports(_module_ast()):
        assert name != "torch" and not name.startswith("torch.")


def test_ast_no_clock_imports():
    for name in _iter_imports(_module_ast()):
        assert name not in {"time", "datetime", "calendar"}


def test_ast_no_random_imports():
    for name in _iter_imports(_module_ast()):
        assert name != "random" and name != "secrets"


def test_ast_no_os_or_io_imports():
    for name in _iter_imports(_module_ast()):
        assert name not in {"os", "io", "pathlib", "sys"}


def test_ast_no_engine_cross_imports():
    forbidden_prefixes = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "registry",
    )
    for name in _iter_imports(_module_ast()):
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), f"forbidden engine import: {name}"


def test_ast_no_clock_text():
    src = _module_source()
    forbidden = ("time.time", "time.monotonic", "time.perf_counter", "datetime.")
    for token in forbidden:
        assert token not in src, f"clock-text leak: {token}"


def test_ast_no_logging_or_print():
    src = _module_source()
    assert "logging" not in src
    assert "print(" not in src


def test_imports_are_stdlib_only_plus_core_contracts():
    allowed_prefixes = (
        "__future__",
        "dataclasses",
        "hashlib",
        "collections",
        "typing",
        "core.contracts",
    )
    for name in _iter_imports(_module_ast()):
        assert any(name == p or name.startswith(p) for p in allowed_prefixes), (
            f"unexpected import: {name}"
        )


# ----------------------------------------------------------------------
# ArenaConfig validation (15)
# ----------------------------------------------------------------------


def test_config_is_frozen_and_slotted():
    assert ArenaConfig.__dataclass_params__.frozen
    assert ArenaConfig.__slots__ == (
        "arena_id",
        "tournament_size",
        "n_winners",
        "elitism_count",
        "drawdown_weight",
    )


def test_config_happy_path():
    cfg = ArenaConfig(
        arena_id="alpha",
        tournament_size=3,
        n_winners=5,
        elitism_count=1,
        drawdown_weight=0.25,
    )
    assert cfg.arena_id == "alpha"
    assert cfg.tournament_size == 3
    assert cfg.n_winners == 5
    assert cfg.elitism_count == 1
    assert cfg.drawdown_weight == 0.25


def test_config_default_drawdown_weight():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    assert cfg.drawdown_weight == DEFAULT_DRAWDOWN_WEIGHT


def test_config_rejects_non_str_arena_id():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(arena_id=123, tournament_size=2, n_winners=1)  # type: ignore[arg-type]


def test_config_rejects_empty_arena_id():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(arena_id="", tournament_size=2, n_winners=1)


def test_config_rejects_oversized_arena_id():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(
            arena_id="x" * (MAX_ARENA_ID_LEN + 1),
            tournament_size=2,
            n_winners=1,
        )


def test_config_rejects_tournament_size_below_min():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(
            arena_id="a",
            tournament_size=MIN_TOURNAMENT_SIZE - 1,
            n_winners=1,
        )


def test_config_rejects_tournament_size_above_max():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(
            arena_id="a",
            tournament_size=MAX_TOURNAMENT_SIZE + 1,
            n_winners=1,
        )


def test_config_rejects_n_winners_zero():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(arena_id="a", tournament_size=2, n_winners=0)


def test_config_rejects_n_winners_above_max():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(arena_id="a", tournament_size=2, n_winners=MAX_N_WINNERS + 1)


def test_config_rejects_negative_elitism():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(arena_id="a", tournament_size=2, n_winners=1, elitism_count=-1)


def test_config_rejects_elitism_above_max():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(
            arena_id="a",
            tournament_size=2,
            n_winners=1,
            elitism_count=MAX_ELITISM_COUNT + 1,
        )


def test_config_rejects_negative_drawdown_weight():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(
            arena_id="a",
            tournament_size=2,
            n_winners=1,
            drawdown_weight=-1.0,
        )


def test_config_rejects_non_finite_drawdown_weight():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(
            arena_id="a",
            tournament_size=2,
            n_winners=1,
            drawdown_weight=float("inf"),
        )


def test_config_rejects_bool_tournament_size():
    with pytest.raises(ArenaConfigError):
        ArenaConfig(
            arena_id="a",
            tournament_size=True,
            n_winners=1,  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------
# Contestant validation (5)
# ----------------------------------------------------------------------


def test_contestant_is_frozen_and_slotted():
    assert Contestant.__dataclass_params__.frozen
    assert Contestant.__slots__ == ("strategy_id", "summary")


def test_contestant_happy_path():
    c = Contestant(strategy_id="s1", summary=_make_summary())
    assert c.strategy_id == "s1"


def test_contestant_rejects_empty_id():
    with pytest.raises(ArenaInputError):
        Contestant(strategy_id="", summary=_make_summary())


def test_contestant_rejects_non_str_id():
    with pytest.raises(ArenaInputError):
        Contestant(strategy_id=42, summary=_make_summary())  # type: ignore[arg-type]


def test_contestant_rejects_non_summary_payload():
    with pytest.raises(ArenaInputError):
        Contestant(strategy_id="s1", summary={"oops": 1})  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# TournamentBracket / TournamentResult validation (8)
# ----------------------------------------------------------------------


def test_bracket_is_frozen_and_slotted():
    assert TournamentBracket.__dataclass_params__.frozen
    assert TournamentBracket.__slots__ == (
        "bracket_id",
        "candidate_ids",
        "winner_id",
        "winner_fitness",
    )


def test_bracket_rejects_winner_outside_candidates():
    with pytest.raises(ArenaInputError):
        TournamentBracket(
            bracket_id=0,
            candidate_ids=("a", "b"),
            winner_id="c",
            winner_fitness=1.0,
        )


def test_bracket_rejects_negative_id():
    with pytest.raises(ArenaInputError):
        TournamentBracket(
            bracket_id=-1,
            candidate_ids=("a",),
            winner_id="a",
            winner_fitness=0.0,
        )


def test_bracket_rejects_empty_candidates():
    with pytest.raises(ArenaInputError):
        TournamentBracket(
            bracket_id=0,
            candidate_ids=(),
            winner_id="a",
            winner_fitness=0.0,
        )


def test_bracket_rejects_non_finite_fitness():
    with pytest.raises(ArenaInputError):
        TournamentBracket(
            bracket_id=0,
            candidate_ids=("a",),
            winner_id="a",
            winner_fitness=float("nan"),
        )


def test_result_is_frozen_and_slotted():
    assert TournamentResult.__dataclass_params__.frozen
    assert "arena_digest" in TournamentResult.__slots__


def test_result_rejects_bad_digest_length():
    contestants = tuple(_make_contestants(2))
    with pytest.raises(ArenaInputError):
        TournamentResult(
            arena_id="a",
            ts_ns=1,
            seed=1,
            contestants=contestants,
            elites=(),
            tournament_winners=(),
            survivors=(),
            brackets=(),
            arena_digest="abc",
        )


def test_result_rejects_non_hex_digest():
    contestants = tuple(_make_contestants(2))
    with pytest.raises(ArenaInputError):
        TournamentResult(
            arena_id="a",
            ts_ns=1,
            seed=1,
            contestants=contestants,
            elites=(),
            tournament_winners=(),
            survivors=(),
            brackets=(),
            arena_digest="zzzzzzzzzzzzzzzz",
        )


def test_result_rejects_unknown_survivor():
    contestants = tuple(_make_contestants(2))
    with pytest.raises(ArenaInputError):
        TournamentResult(
            arena_id="a",
            ts_ns=1,
            seed=1,
            contestants=contestants,
            elites=(),
            tournament_winners=(),
            survivors=("ghost",),
            brackets=(),
            arena_digest="0" * 16,
        )


# ----------------------------------------------------------------------
# Arena.run argument validation (10)
# ----------------------------------------------------------------------


def test_arena_run_rejects_non_arenaconfig():
    with pytest.raises(ArenaInputError):
        Arena().run(
            contestants=_make_contestants(),
            config="not-a-config",  # type: ignore[arg-type]
            seed=1,
            ts_ns=1,
        )


def test_arena_run_rejects_too_few_contestants():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    with pytest.raises(ArenaInputError):
        Arena().run(
            contestants=_make_contestants(1),
            config=cfg,
            seed=1,
            ts_ns=1,
        )


def test_arena_run_rejects_duplicate_strategy_ids():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    contestants = _make_contestants(2)
    contestants.append(Contestant(strategy_id=contestants[0].strategy_id, summary=_make_summary()))
    with pytest.raises(ArenaInputError):
        Arena().run(contestants=contestants, config=cfg, seed=1, ts_ns=1)


def test_arena_run_rejects_non_contestant_entry():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    with pytest.raises(ArenaInputError):
        Arena().run(
            contestants=[_make_contestants(1)[0], "not-a-contestant"],  # type: ignore[list-item]
            config=cfg,
            seed=1,
            ts_ns=1,
        )


def test_arena_run_rejects_non_int_seed():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    with pytest.raises(ArenaInputError):
        Arena().run(
            contestants=_make_contestants(2),
            config=cfg,
            seed="bad",  # type: ignore[arg-type]
            ts_ns=1,
        )


def test_arena_run_rejects_negative_seed():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    with pytest.raises(ArenaInputError):
        Arena().run(
            contestants=_make_contestants(2),
            config=cfg,
            seed=-1,
            ts_ns=1,
        )


def test_arena_run_rejects_bool_seed():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    with pytest.raises(ArenaInputError):
        Arena().run(
            contestants=_make_contestants(2),
            config=cfg,
            seed=True,  # type: ignore[arg-type]
            ts_ns=1,
        )


def test_arena_run_rejects_negative_ts_ns():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    with pytest.raises(ArenaInputError):
        Arena().run(
            contestants=_make_contestants(2),
            config=cfg,
            seed=1,
            ts_ns=-1,
        )


def test_arena_run_rejects_elitism_above_count():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1, elitism_count=99)
    with pytest.raises(ArenaInputError):
        Arena().run(
            contestants=_make_contestants(3),
            config=cfg,
            seed=1,
            ts_ns=1,
        )


def test_arena_run_rejects_non_sequence_contestants():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    with pytest.raises(ArenaInputError):
        Arena().run(
            contestants={c.strategy_id: c for c in _make_contestants(2)},  # type: ignore[arg-type]
            config=cfg,
            seed=1,
            ts_ns=1,
        )


# ----------------------------------------------------------------------
# Tournament + elitism semantics (12)
# ----------------------------------------------------------------------


def test_arena_returns_tournament_result_shape():
    cfg = ArenaConfig(arena_id="a", tournament_size=3, n_winners=4, elitism_count=2)
    res = Arena().run(
        contestants=_make_contestants(8),
        config=cfg,
        seed=42,
        ts_ns=1,
    )
    assert isinstance(res, TournamentResult)
    assert res.arena_id == "a"
    assert res.ts_ns == 1
    assert res.seed == 42
    assert len(res.elites) == 2
    assert len(res.tournament_winners) == 4
    assert len(res.brackets) == 4


def test_arena_elites_are_top_n_by_pnl_mean():
    contestants = _make_contestants(8)
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1, elitism_count=3)
    res = Arena().run(contestants=contestants, config=cfg, seed=1, ts_ns=1)
    assert res.elites == ("strat-007", "strat-006", "strat-005")


def test_arena_elite_zero_yields_no_elites():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=2, elitism_count=0)
    res = Arena().run(
        contestants=_make_contestants(4),
        config=cfg,
        seed=1,
        ts_ns=1,
    )
    assert res.elites == ()


def test_arena_brackets_have_tournament_size_candidates():
    cfg = ArenaConfig(arena_id="a", tournament_size=4, n_winners=3)
    res = Arena().run(
        contestants=_make_contestants(8),
        config=cfg,
        seed=99,
        ts_ns=1,
    )
    for b in res.brackets:
        assert len(b.candidate_ids) == 4


def test_arena_winner_is_best_in_each_bracket():
    contestants = _make_contestants(8)
    cfg = ArenaConfig(arena_id="a", tournament_size=3, n_winners=5)
    res = Arena().run(contestants=contestants, config=cfg, seed=7, ts_ns=1)
    by_id = {c.strategy_id: c for c in contestants}
    for bracket in res.brackets:
        bracket_pnls = [by_id[cid].summary.pnl_mean_usd for cid in bracket.candidate_ids]
        assert by_id[bracket.winner_id].summary.pnl_mean_usd == max(bracket_pnls)


def test_arena_survivors_dedup_preserves_first_occurrence():
    cfg = ArenaConfig(arena_id="a", tournament_size=8, n_winners=3, elitism_count=2)
    res = Arena().run(
        contestants=_make_contestants(8),
        config=cfg,
        seed=1,
        ts_ns=1,
    )
    seen: set[str] = set()
    for s in res.survivors:
        assert s not in seen
        seen.add(s)
    assert set(res.survivors) <= set(res.elites) | set(res.tournament_winners)


def test_arena_elite_always_in_survivors():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=2, elitism_count=2)
    res = Arena().run(
        contestants=_make_contestants(8),
        config=cfg,
        seed=1,
        ts_ns=1,
    )
    for elite in res.elites:
        assert elite in res.survivors


def test_arena_tournament_with_replacement_allows_duplicate_winners():
    contestants = _make_contestants(2)
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=10)
    res = Arena().run(contestants=contestants, config=cfg, seed=3, ts_ns=1)
    assert len(res.tournament_winners) == 10
    assert len(res.survivors) <= 2


def test_arena_drawdown_weight_can_change_ordering():
    high_pnl_high_dd = Contestant(
        strategy_id="s-high",
        summary=_make_summary("s1", mean=100.0, dd=200.0),
    )
    low_pnl_low_dd = Contestant(
        strategy_id="s-low",
        summary=_make_summary("s2", mean=50.0, dd=1.0),
    )
    contestants = [high_pnl_high_dd, low_pnl_low_dd]

    cfg_no_pen = ArenaConfig(
        arena_id="a",
        tournament_size=2,
        n_winners=1,
        elitism_count=1,
    )
    cfg_with_pen = ArenaConfig(
        arena_id="a",
        tournament_size=2,
        n_winners=1,
        elitism_count=1,
        drawdown_weight=1.0,
    )
    r1 = Arena().run(contestants=contestants, config=cfg_no_pen, seed=1, ts_ns=1)
    r2 = Arena().run(contestants=contestants, config=cfg_with_pen, seed=1, ts_ns=1)
    assert r1.elites == ("s-high",)
    assert r2.elites == ("s-low",)


def test_arena_handles_min_size_input():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    res = Arena().run(
        contestants=_make_contestants(MIN_CONTESTANTS),
        config=cfg,
        seed=1,
        ts_ns=1,
    )
    assert len(res.contestants) == MIN_CONTESTANTS


def test_arena_tournament_winners_have_finite_fitness():
    cfg = ArenaConfig(arena_id="a", tournament_size=3, n_winners=4)
    res = Arena().run(
        contestants=_make_contestants(8),
        config=cfg,
        seed=1,
        ts_ns=1,
    )
    for b in res.brackets:
        assert b.winner_fitness == b.winner_fitness  # not NaN
        assert b.winner_fitness not in (float("inf"), float("-inf"))


def test_arena_winners_match_brackets_order():
    cfg = ArenaConfig(arena_id="a", tournament_size=3, n_winners=5)
    res = Arena().run(
        contestants=_make_contestants(8),
        config=cfg,
        seed=11,
        ts_ns=1,
    )
    bracket_winners = tuple(b.winner_id for b in res.brackets)
    assert bracket_winners == res.tournament_winners


# ----------------------------------------------------------------------
# Tie-break determinism (3)
# ----------------------------------------------------------------------


def test_tie_break_by_strategy_id_ascending():
    contestants = [
        Contestant(strategy_id="strat-c", summary=_make_summary(mean=10.0)),
        Contestant(strategy_id="strat-a", summary=_make_summary(mean=10.0)),
        Contestant(strategy_id="strat-b", summary=_make_summary(mean=10.0)),
    ]
    cfg = ArenaConfig(arena_id="a", tournament_size=3, n_winners=1, elitism_count=3)
    res = Arena().run(contestants=contestants, config=cfg, seed=1, ts_ns=1)
    assert res.elites == ("strat-a", "strat-b", "strat-c")


def test_tie_break_stable_across_input_shuffles():
    perm_a = [
        Contestant(strategy_id="x1", summary=_make_summary(mean=5.0)),
        Contestant(strategy_id="x2", summary=_make_summary(mean=5.0)),
        Contestant(strategy_id="x3", summary=_make_summary(mean=5.0)),
    ]
    perm_b = list(reversed(perm_a))
    cfg = ArenaConfig(arena_id="a", tournament_size=3, n_winners=1, elitism_count=3)
    r1 = Arena().run(contestants=perm_a, config=cfg, seed=42, ts_ns=1)
    r2 = Arena().run(contestants=perm_b, config=cfg, seed=42, ts_ns=1)
    assert r1.elites == r2.elites
    assert r1.tournament_winners == r2.tournament_winners


def test_constant_evaluator_yields_alphabetic_winner():
    contestants = [
        Contestant(strategy_id=f"strat-{c}", summary=_make_summary(mean=1.0)) for c in "edcba"
    ]
    cfg = ArenaConfig(arena_id="a", tournament_size=5, n_winners=3)
    res = Arena().run(contestants=contestants, config=cfg, seed=1, ts_ns=1)
    for b in res.brackets:
        assert b.winner_id == sorted(b.candidate_ids)[0]


# ----------------------------------------------------------------------
# INV-15 byte-identical replay (5)
# ----------------------------------------------------------------------


def test_inv15_three_run_identical_digest():
    cfg = ArenaConfig(arena_id="alpha", tournament_size=3, n_winners=4, elitism_count=2)
    contestants = _make_contestants(8)
    digests = []
    for _ in range(3):
        res = Arena().run(contestants=contestants, config=cfg, seed=99, ts_ns=42)
        digests.append(res.arena_digest)
    assert len(set(digests)) == 1


def test_inv15_three_run_identical_brackets():
    cfg = ArenaConfig(arena_id="alpha", tournament_size=3, n_winners=4, elitism_count=2)
    contestants = _make_contestants(8)
    runs = [Arena().run(contestants=contestants, config=cfg, seed=99, ts_ns=42) for _ in range(3)]
    for r in runs[1:]:
        assert r.brackets == runs[0].brackets


def test_inv15_three_run_identical_survivors():
    cfg = ArenaConfig(arena_id="alpha", tournament_size=3, n_winners=4, elitism_count=2)
    contestants = _make_contestants(8)
    runs = [Arena().run(contestants=contestants, config=cfg, seed=99, ts_ns=42) for _ in range(3)]
    for r in runs[1:]:
        assert r.survivors == runs[0].survivors


def test_inv15_different_seed_yields_different_digest():
    cfg = ArenaConfig(arena_id="alpha", tournament_size=3, n_winners=4)
    contestants = _make_contestants(8)
    r1 = Arena().run(contestants=contestants, config=cfg, seed=1, ts_ns=42)
    r2 = Arena().run(contestants=contestants, config=cfg, seed=2, ts_ns=42)
    assert r1.arena_digest != r2.arena_digest


def test_inv15_different_ts_ns_yields_different_digest():
    cfg = ArenaConfig(arena_id="alpha", tournament_size=3, n_winners=4)
    contestants = _make_contestants(8)
    r1 = Arena().run(contestants=contestants, config=cfg, seed=1, ts_ns=1)
    r2 = Arena().run(contestants=contestants, config=cfg, seed=1, ts_ns=2)
    assert r1.arena_digest != r2.arena_digest


# ----------------------------------------------------------------------
# Output shape audits (4)
# ----------------------------------------------------------------------


def test_result_digest_is_16_hex_chars():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=2)
    res = Arena().run(
        contestants=_make_contestants(4),
        config=cfg,
        seed=1,
        ts_ns=1,
    )
    assert len(res.arena_digest) == 16
    int(res.arena_digest, 16)  # must parse


def test_result_contestants_match_input_count():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=2)
    contestants = _make_contestants(6)
    res = Arena().run(contestants=contestants, config=cfg, seed=1, ts_ns=1)
    assert len(res.contestants) == len(contestants)


def test_result_arena_id_propagates_from_config():
    cfg = ArenaConfig(arena_id="bravo", tournament_size=2, n_winners=1)
    res = Arena().run(
        contestants=_make_contestants(2),
        config=cfg,
        seed=1,
        ts_ns=1,
    )
    assert res.arena_id == "bravo"


def test_result_seed_and_ts_ns_propagate():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1)
    res = Arena().run(
        contestants=_make_contestants(2),
        config=cfg,
        seed=12345,
        ts_ns=999,
    )
    assert res.seed == 12345
    assert res.ts_ns == 999


# ----------------------------------------------------------------------
# OFFLINE_ONLY tier import audit (1)
# ----------------------------------------------------------------------


def test_module_imports_clean_without_deap():
    # The arena module has already been imported by this test file's
    # top-level imports. Confirm that side-effect did NOT pull deap
    # into sys.modules. Reloading is unsafe — it would break
    # ``isinstance`` checks against the existing Contestant class.
    assert "deap" not in sys.modules


# ----------------------------------------------------------------------
# Boundary cases (3)
# ----------------------------------------------------------------------


def test_arena_handles_single_winner_no_elite():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1, elitism_count=0)
    res = Arena().run(
        contestants=_make_contestants(2),
        config=cfg,
        seed=1,
        ts_ns=1,
    )
    assert res.elites == ()
    assert len(res.tournament_winners) == 1
    assert res.survivors == res.tournament_winners


def test_arena_handles_full_elitism():
    cfg = ArenaConfig(arena_id="a", tournament_size=2, n_winners=1, elitism_count=4)
    res = Arena().run(
        contestants=_make_contestants(4),
        config=cfg,
        seed=1,
        ts_ns=1,
    )
    assert len(res.elites) == 4
    assert set(res.elites) == {f"strat-{i:03d}" for i in range(4)}


def test_arena_high_dimensional_input():
    n = 64
    cfg = ArenaConfig(arena_id="a", tournament_size=8, n_winners=20, elitism_count=4)
    res = Arena().run(
        contestants=_make_contestants(n),
        config=cfg,
        seed=7,
        ts_ns=1,
    )
    assert len(res.contestants) == n
    assert len(res.tournament_winners) == 20
    assert len(res.elites) == 4
