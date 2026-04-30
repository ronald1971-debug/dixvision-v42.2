"""Wave-Stress-Tests — adversarial coverage of StrategyRegistry FSM.

The :class:`StrategyRegistry` (PR #113, Wave-04.6 PR-D) is the
governance-side approval lifecycle for every strategy that may be
dispatched. Its FSM (DRAFT → VALIDATING → APPROVED → RETIRED) is the
last gate before LIVE / CANARY / AUTO modes can run a strategy. A
silent edge violation is an INV-56 (Triad Lock) hazard. This file
hammers the FSM from every angle:

* Random sequences of legal transitions never diverge across replays
  (INV-15: ledger-replay equals live).
* Random sequences containing illegal edges leave the registry +
  ledger byte-identical to the pre-attempt state (PR #113 fix on
  ledger-first ordering).
* Replay is idempotent across many invocations.
* Many strategies in flight at once never cross-contaminate each other.

INV-15 — every test uses a seeded :class:`random.Random` so failures
are deterministic and bisectable.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from core.contracts.strategy_registry import (
    LEGAL_LIFECYCLE_TRANSITIONS,
    StrategyLifecycle,
    StrategyLifecycleError,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.strategy_registry import StrategyRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> tuple[StrategyRegistry, LedgerAuthorityWriter]:
    ledger = LedgerAuthorityWriter()
    return StrategyRegistry(ledger=ledger), ledger


def _legal_targets(state: StrategyLifecycle) -> tuple[StrategyLifecycle, ...]:
    return tuple(LEGAL_LIFECYCLE_TRANSITIONS[state])


def _all_states() -> tuple[StrategyLifecycle, ...]:
    return tuple(LEGAL_LIFECYCLE_TRANSITIONS.keys())


def _random_legal_path(
    rng: random.Random,
    *,
    max_hops: int,
) -> list[StrategyLifecycle]:
    """Return a random legal lifecycle path starting at DRAFT."""
    path: list[StrategyLifecycle] = [StrategyLifecycle.DRAFT]
    cur = StrategyLifecycle.DRAFT
    for _ in range(max_hops):
        targets = _legal_targets(cur)
        if not targets:
            break
        nxt = rng.choice(targets)
        path.append(nxt)
        cur = nxt
        if cur is StrategyLifecycle.RETIRED:
            break
    return path


# ---------------------------------------------------------------------------
# 1. Random legal-path replay equivalence (INV-15)
# ---------------------------------------------------------------------------


def test_random_legal_paths_replay_identically() -> None:
    """For 50 random strategies on random legal paths, replay == live."""
    rng = random.Random(20260420)
    live, ledger = _fresh()

    for n in range(50):
        sid = f"strat-{n}"
        path = _random_legal_path(rng, max_hops=4)
        ts_ns = (n + 1) * 100
        live.register_draft(strategy_id=sid, ts_ns=ts_ns)
        for i, target in enumerate(path[1:], start=1):
            live.transition(
                strategy_id=sid,
                new_lifecycle=target,
                ts_ns=ts_ns + i,
                reason=f"r{i}",
            )

    replay = StrategyRegistry(ledger=LedgerAuthorityWriter())
    replay.replay_from_ledger(ledger.read())

    assert len(replay) == len(live)
    for sid in [f"strat-{n}" for n in range(50)]:
        assert replay.get(sid) == live.get(sid), (
            f"replay diverged on {sid}"
        )


def test_random_legal_paths_double_replay_is_idempotent() -> None:
    """Replaying twice produces the same registry state as once."""
    rng = random.Random(424242)
    _, ledger = _fresh_with_random_path(rng, hops=3, count=20)

    once = StrategyRegistry(ledger=LedgerAuthorityWriter())
    once.replay_from_ledger(ledger.read())

    twice = StrategyRegistry(ledger=LedgerAuthorityWriter())
    twice.replay_from_ledger(ledger.read())
    twice.replay_from_ledger(ledger.read())

    assert len(once) == len(twice)
    for state in _all_states():
        for r in once.all_in(state):
            assert twice.get(r.strategy_id) == r


def _fresh_with_random_path(
    rng: random.Random, *, hops: int, count: int
) -> tuple[StrategyRegistry, LedgerAuthorityWriter]:
    live, ledger = _fresh()
    for n in range(count):
        sid = f"strat-{n}"
        path = _random_legal_path(rng, max_hops=hops)
        ts_ns = (n + 1) * 100
        live.register_draft(strategy_id=sid, ts_ns=ts_ns)
        for i, target in enumerate(path[1:], start=1):
            live.transition(
                strategy_id=sid,
                new_lifecycle=target,
                ts_ns=ts_ns + i,
                reason=f"r{i}",
            )
    return live, ledger


# ---------------------------------------------------------------------------
# 2. Illegal-edge fuzz — invariant: state + ledger length unchanged
# ---------------------------------------------------------------------------


def test_random_illegal_transitions_do_not_mutate_state() -> None:
    """For every (prev, new) edge that is *not* in the legal table, the
    raise must leave both the in-memory record and the ledger row count
    untouched (PR #113 fix on ledger-first ordering)."""
    illegal_edges: list[tuple[StrategyLifecycle, StrategyLifecycle]] = []
    for prev in _all_states():
        for new in _all_states():
            if new not in LEGAL_LIFECYCLE_TRANSITIONS[prev]:
                if prev is new:
                    continue  # trivially "illegal" but uninteresting
                illegal_edges.append((prev, new))

    assert illegal_edges, "table has no illegal edges to test"

    for prev, new in illegal_edges:
        reg, ledger = _fresh()
        sid = "s1"
        # Walk legally to ``prev`` first — DRAFT is start state.
        reg.register_draft(strategy_id=sid, ts_ns=1)
        legal_walk = _walk_to(prev)
        for i, intermediate in enumerate(legal_walk, start=2):
            reg.transition(
                strategy_id=sid,
                new_lifecycle=intermediate,
                ts_ns=i,
                reason=f"setup-{intermediate.value}",
            )
        rows_before = len(ledger.read())
        record_before = reg.get(sid)

        try:
            reg.transition(
                strategy_id=sid,
                new_lifecycle=new,
                ts_ns=999,
                reason="illegal-attempt",
            )
            raise AssertionError(
                f"illegal {prev.value} → {new.value} did not raise"
            )
        except StrategyLifecycleError:
            pass

        # State + ledger must be byte-identical to pre-attempt.
        assert reg.get(sid) == record_before, (
            f"state mutated on illegal {prev.value} → {new.value}"
        )
        assert len(ledger.read()) == rows_before, (
            f"ledger row appended on illegal {prev.value} → {new.value}"
        )


def _walk_to(target: StrategyLifecycle) -> Sequence[StrategyLifecycle]:
    """Return the canonical legal walk from DRAFT to ``target``."""
    if target is StrategyLifecycle.DRAFT:
        return ()
    if target is StrategyLifecycle.VALIDATING:
        return (StrategyLifecycle.VALIDATING,)
    if target is StrategyLifecycle.APPROVED:
        return (StrategyLifecycle.VALIDATING, StrategyLifecycle.APPROVED)
    if target is StrategyLifecycle.RETIRED:
        return (StrategyLifecycle.RETIRED,)
    raise AssertionError(f"unknown target: {target}")


# ---------------------------------------------------------------------------
# 3. Multi-strategy isolation
# ---------------------------------------------------------------------------


def test_many_strategies_in_flight_do_not_cross_contaminate() -> None:
    """100 strategies on different paths must end up in their declared
    states, with no registry leak between strategy_ids."""
    rng = random.Random(2026)
    reg, _ = _fresh()
    expected: dict[str, StrategyLifecycle] = {}

    for n in range(100):
        sid = f"strat-{n}"
        path = _random_legal_path(rng, max_hops=4)
        ts_ns = (n + 1) * 100
        reg.register_draft(strategy_id=sid, ts_ns=ts_ns)
        for i, target in enumerate(path[1:], start=1):
            reg.transition(
                strategy_id=sid,
                new_lifecycle=target,
                ts_ns=ts_ns + i,
                reason=f"r{i}",
            )
        expected[sid] = path[-1]

    for sid, want in expected.items():
        rec = reg.get(sid)
        assert rec is not None, f"missing {sid}"
        assert rec.lifecycle is want, (
            f"{sid}: lifecycle drift — want {want}, got {rec.lifecycle}"
        )


def test_strategies_filter_returns_only_lifecycle_match() -> None:
    """For each StrategyLifecycle, ``records_in`` returns the right set."""
    rng = random.Random(7777)
    reg, _ = _fresh()

    for n in range(60):
        sid = f"strat-{n}"
        path = _random_legal_path(rng, max_hops=4)
        ts_ns = (n + 1) * 100
        reg.register_draft(strategy_id=sid, ts_ns=ts_ns)
        for i, target in enumerate(path[1:], start=1):
            reg.transition(
                strategy_id=sid,
                new_lifecycle=target,
                ts_ns=ts_ns + i,
                reason=f"r{i}",
            )

    # Build the canonical view by walking every state with all_in().
    seen: dict[str, StrategyLifecycle] = {}
    for state in _all_states():
        for r in reg.all_in(state):
            assert r.lifecycle is state, (
                f"all_in({state}) returned {r.strategy_id} in {r.lifecycle}"
            )
            assert r.strategy_id not in seen, (
                f"{r.strategy_id} returned by two states"
            )
            seen[r.strategy_id] = state
    assert len(seen) == len(reg), (
        "all_in walk did not cover every registered strategy"
    )


# ---------------------------------------------------------------------------
# 4. Reason / strategy_id input fuzz
# ---------------------------------------------------------------------------


def test_register_draft_rejects_empty_strategy_id_under_fuzz() -> None:
    """Empty ``strategy_id`` is always rejected, regardless of other args."""
    reg, _ = _fresh()
    rng = random.Random(1)
    for _ in range(20):
        params = {f"p{i}": str(rng.random()) for i in range(rng.randint(0, 3))}
        try:
            reg.register_draft(strategy_id="", ts_ns=rng.randint(1, 1_000_000), parameters=params)
        except ValueError:
            continue
        raise AssertionError("empty strategy_id was not rejected")


def test_transition_rejects_empty_reason_under_fuzz() -> None:
    """Empty ``reason`` is always rejected on every legal edge."""
    rng = random.Random(2)
    for prev in _all_states():
        for new in _legal_targets(prev):
            reg, _ = _fresh()
            sid = f"s-{prev.value}-{new.value}"
            reg.register_draft(strategy_id=sid, ts_ns=1)
            for i, intermediate in enumerate(_walk_to(prev), start=2):
                reg.transition(
                    strategy_id=sid,
                    new_lifecycle=intermediate,
                    ts_ns=i,
                    reason="setup",
                )
            try:
                reg.transition(
                    strategy_id=sid,
                    new_lifecycle=new,
                    ts_ns=rng.randint(100, 1_000_000),
                    reason="",
                )
            except ValueError:
                continue
            raise AssertionError(
                f"empty reason accepted on {prev.value} → {new.value}"
            )


def test_transition_unknown_strategy_always_raises_keyerror() -> None:
    """Random unknown strategy ids must always KeyError, never silently
    succeed or attribute-error."""
    rng = random.Random(3)
    for _ in range(40):
        reg, _ = _fresh()
        bogus = f"missing-{rng.randint(0, 1_000_000)}"
        try:
            reg.transition(
                strategy_id=bogus,
                new_lifecycle=StrategyLifecycle.VALIDATING,
                ts_ns=rng.randint(1, 1_000_000),
                reason="nope",
            )
        except KeyError:
            continue
        raise AssertionError(f"transition on {bogus} did not raise KeyError")


# ---------------------------------------------------------------------------
# 5. Version monotonicity invariant
# ---------------------------------------------------------------------------


def test_version_monotonically_increases_on_each_transition() -> None:
    """``StrategyRecord.version`` increases by exactly 1 per transition."""
    rng = random.Random(99)
    reg, _ = _fresh()
    for n in range(30):
        sid = f"strat-{n}"
        path = _random_legal_path(rng, max_hops=4)
        ts_ns = (n + 1) * 100
        reg.register_draft(strategy_id=sid, ts_ns=ts_ns)
        first = reg.get(sid)
        assert first is not None and first.version == 1
        prev_version = 1
        for i, target in enumerate(path[1:], start=1):
            reg.transition(
                strategy_id=sid,
                new_lifecycle=target,
                ts_ns=ts_ns + i,
                reason=f"r{i}",
            )
            cur = reg.get(sid)
            assert cur is not None
            assert cur.version == prev_version + 1, (
                f"{sid}: version skipped — {prev_version} → {cur.version}"
            )
            prev_version = cur.version
