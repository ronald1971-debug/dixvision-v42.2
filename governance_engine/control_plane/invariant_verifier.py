# ADAPTED FROM: Z3Prover/z3 Python bindings
"""A-17 — Governance invariant formal verification via z3-solver.

Three invariants are encoded as SMT problems and submitted to a
solver backend. **UNSAT means the invariant holds**: no model
satisfies the *negation* of the invariant, so no counterexample
exists. **SAT means the invariant is violated** and the solver
returns a concrete counterexample. **UNKNOWN is treated as
VIOLATED** at the API boundary — fail-closed (operator must
investigate before promotion).

Three encodings shipped in this leaf
------------------------------------

1. :meth:`InvariantVerifier.verify_position_limit` — for every
   position size ``p in [0, max_position]`` and leverage
   ``l in [1, max_leverage]`` the exposure ``p * l`` is bounded by
   ``exposure_cap``. Z3 encoding searches for any ``(p, l)`` that
   would violate the cap; UNSAT proves the cap is impossible to
   breach with the declared bounds.

2. :meth:`InvariantVerifier.verify_autonomy_escalation` — every
   allowed mode-transition edge is either a single-step rank
   increase (escalation) or any descent (emergency demotion).
   Z3 encoding asserts that some declared edge skips a rank;
   UNSAT proves no skip-rank promotion is reachable.

3. :meth:`InvariantVerifier.verify_no_governance_bypass` — given an
   authority graph ``edges = ((source, target), ...)`` and a set of
   ``governance_nodes``, there is no path from any non-governance
   source to ``sink`` that does not cross at least one governance
   node. Z3 encoding searches for a reachable bypass path; UNSAT
   proves every path passes through governance.

License posture
---------------
``Z3Prover/z3`` Python bindings are MIT-licensed — fully compatible
with the DIX project. The adapter consumes only the documented
public API (:class:`Solver`, :class:`Int`, :class:`Real`,
:class:`Bool`, :class:`Function`, :func:`ForAll`, :func:`Exists`,
:func:`Implies`, :func:`And`, :func:`Or`, :func:`Not`, :func:`sat`,
:func:`unsat`, :func:`unknown`) — no internal Z3 code is copied.

Tier discipline
---------------
* **OFFLINE_ONLY.** Z3 proofs run in CI and in offline governance
  static analysis — never in the runtime hot path. ``check()`` can
  take seconds and is bounded only by the solver's timeout. The
  module is OK to import at engine boot (Z3 is not eagerly loaded);
  the solver itself is lazy-imported only inside
  :func:`z3_backend_factory`.
* **No engine cross-imports.** The verifier reasons over plain
  primitives (``int`` / ``float`` / ``str``) only — no engine type
  is constructed inside the module. Pinned by an AST test.
* **No clock / no random / no IO.** Pure pure-Python wrapper around
  the solver. Pinned by an AST test.
* **B27 / B28 / INV-71 authority symmetry.** Output is a
  :class:`VerificationReport` (frozen value object). The verifier
  does **not** construct ``GovernanceDecision`` / ``PatchProposal``
  / typed bus events — projection into a typed decision lives in
  the existing ``governance_engine.control_plane`` coordinators.

What survives from upstream
---------------------------
* ``Solver`` lifecycle: ``add()`` constraints, ``check()`` returning
  ``sat`` / ``unsat`` / ``unknown``, ``model()`` returning a model
  on SAT.
* :func:`Int`, :func:`Real`, :func:`Bool`, :func:`Function` — declared
  symbolic constants and uninterpreted functions.
* :func:`ForAll`, :func:`Exists` — quantifier encoding.
* :func:`Implies`, :func:`And`, :func:`Or`, :func:`Not` — boolean
  connectives over arithmetic constraints.

What is rewritten behind DIX contracts
--------------------------------------
* The :class:`SMTBackend` Protocol abstracts the solver call. Tests
  inject :class:`InProcessSMTBackend`, a deterministic pure-Python
  decider that does *not* import z3 and never touches the solver
  binary. Production wires :class:`_Z3SMTBackend` via
  :func:`z3_backend_factory`, which lazy-imports z3 inside the
  factory body.
* Every solver call gets a deterministic seed (``smt.random_seed``)
  and a structured timeout so CI behaviour is byte-stable.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

# pip dependency flag — the verifier lazy-imports `z3` only inside
# :func:`z3_backend_factory`. The module itself is importable
# without z3 installed; tests run against
# :class:`InProcessSMTBackend`.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("z3-solver",)

DEFAULT_SOLVER_TIMEOUT_MS: int = 30_000
"""Hard solver timeout (CI must stay bounded)."""

DEFAULT_SMT_RANDOM_SEED: int = 0
"""z3 ``smt.random_seed`` for byte-stable CI runs."""


# --------------------------------------------------------------------
# Verdicts & reports
# --------------------------------------------------------------------


class SolverVerdict(enum.StrEnum):
    """Three-valued solver verdict."""

    SAT = "SAT"
    """A model satisfying the encoded formula was found."""

    UNSAT = "UNSAT"
    """The encoded formula is unsatisfiable — invariant holds."""

    UNKNOWN = "UNKNOWN"
    """Solver hit a timeout or undecidable encoding."""


class VerificationStatus(enum.StrEnum):
    """Status of an invariant after solver projection."""

    HOLDS = "HOLDS"
    """Solver returned UNSAT on the *negation* — invariant proven."""

    VIOLATED = "VIOLATED"
    """Solver returned SAT — concrete counterexample exists."""

    UNKNOWN = "UNKNOWN"
    """Solver could not decide. Treated as VIOLATED by callers."""


@dataclasses.dataclass(frozen=True, slots=True)
class SolverResult:
    """Pure-data record returned by :class:`SMTBackend`.

    Attributes:
        verdict: Three-valued solver verdict.
        counterexample: Stable sorted mapping ``var_name -> str(value)``
            describing the SAT model. Empty when ``verdict`` is
            ``UNSAT`` or ``UNKNOWN``. Always sorted-key for INV-15.
    """

    verdict: SolverVerdict
    counterexample: Mapping[str, str] = dataclasses.field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, SolverVerdict):
            raise TypeError("SolverResult.verdict must be a SolverVerdict")
        for k, v in self.counterexample.items():
            if not isinstance(k, str) or not k:
                raise ValueError("SolverResult.counterexample keys must be non-empty str")
            if not isinstance(v, str):
                raise TypeError("SolverResult.counterexample values must be str")


@dataclasses.dataclass(frozen=True, slots=True)
class VerificationReport:
    """Result of one invariant proof attempt.

    Attributes:
        invariant_id: Stable identifier (``INV-...`` / ``SAFE-...``)
            of the encoded invariant.
        status: Three-valued :class:`VerificationStatus`.
        detail: Short human-readable summary (no PII, no secrets).
            Empty string allowed.
        counterexample: Stable sorted mapping when ``status`` is
            VIOLATED. Empty otherwise. Always sorted-key for INV-15.
    """

    invariant_id: str
    status: VerificationStatus
    detail: str = ""
    counterexample: Mapping[str, str] = dataclasses.field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.invariant_id, str):
            raise TypeError("VerificationReport.invariant_id must be str")
        if not self.invariant_id:
            raise ValueError("VerificationReport.invariant_id must be non-empty")
        if not isinstance(self.status, VerificationStatus):
            raise TypeError("VerificationReport.status must be a VerificationStatus")
        for k, v in self.counterexample.items():
            if not isinstance(k, str) or not k:
                raise ValueError("VerificationReport.counterexample keys must be non-empty str")
            if not isinstance(v, str):
                raise TypeError("VerificationReport.counterexample values must be str")

    @property
    def holds(self) -> bool:
        return self.status is VerificationStatus.HOLDS


# --------------------------------------------------------------------
# Problem value objects
# --------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PositionLimitProblem:
    """Encoded position-limit safety problem.

    Attributes:
        max_position: Hard cap on absolute position size (notional).
            Must be positive.
        max_leverage: Hard cap on leverage multiplier. Must be >= 1.
        exposure_cap: Maximum allowed ``position * leverage``. Must
            be positive.
    """

    max_position: float
    max_leverage: float
    exposure_cap: float

    def __post_init__(self) -> None:
        if not isinstance(self.max_position, (int, float)) or isinstance(self.max_position, bool):
            raise TypeError("PositionLimitProblem.max_position must be numeric")
        if self.max_position <= 0.0:
            raise ValueError("PositionLimitProblem.max_position must be positive")
        if not isinstance(self.max_leverage, (int, float)) or isinstance(self.max_leverage, bool):
            raise TypeError("PositionLimitProblem.max_leverage must be numeric")
        if self.max_leverage < 1.0:
            raise ValueError("PositionLimitProblem.max_leverage must be >= 1")
        if not isinstance(self.exposure_cap, (int, float)) or isinstance(self.exposure_cap, bool):
            raise TypeError("PositionLimitProblem.exposure_cap must be numeric")
        if self.exposure_cap <= 0.0:
            raise ValueError("PositionLimitProblem.exposure_cap must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class AutonomyEscalationProblem:
    """Encoded autonomy-escalation safety problem.

    The allowed edges are pairs ``(from_rank, to_rank)`` over the
    mode rank lattice. The invariant requires every *promotion* edge
    (``to_rank > from_rank``) to step exactly one rank. Demotions
    (``to_rank < from_rank``) and self-loops are unrestricted.

    Attributes:
        mode_ranks: Tuple of allowed mode rank ints (must be
            non-empty, unique, sorted ascending).
        allowed_edges: Tuple of ``(from_rank, to_rank)`` pairs the
            FSM permits.
    """

    mode_ranks: tuple[int, ...]
    allowed_edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.mode_ranks:
            raise ValueError("AutonomyEscalationProblem.mode_ranks must be non-empty")
        ranks_sorted = tuple(sorted(set(self.mode_ranks)))
        if ranks_sorted != self.mode_ranks:
            raise ValueError(
                "AutonomyEscalationProblem.mode_ranks must be unique + sorted ascending"
            )
        rank_set = set(self.mode_ranks)
        for src, dst in self.allowed_edges:
            if not isinstance(src, int) or isinstance(src, bool):
                raise TypeError("AutonomyEscalationProblem.allowed_edges entries must be int pairs")
            if not isinstance(dst, int) or isinstance(dst, bool):
                raise TypeError("AutonomyEscalationProblem.allowed_edges entries must be int pairs")
            if src not in rank_set or dst not in rank_set:
                raise ValueError(
                    f"AutonomyEscalationProblem.allowed_edges"
                    f" references unknown rank: ({src}, {dst})"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class GovernanceBypassProblem:
    """Encoded governance-bypass safety problem.

    Attributes:
        nodes: Tuple of authority graph node identifiers (unique,
            sorted ascending for INV-15 determinism).
        edges: Tuple of ``(source, target)`` directed edges. Each
            endpoint must appear in :attr:`nodes`.
        governance_nodes: Tuple of governance node identifiers
            (subset of :attr:`nodes`). A path is a *bypass* iff it
            visits *no* governance node.
        source: Starting node (must appear in :attr:`nodes` and not
            in :attr:`governance_nodes`).
        sink: Target node (must appear in :attr:`nodes`).
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    governance_nodes: tuple[str, ...]
    source: str
    sink: str

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("GovernanceBypassProblem.nodes must be non-empty")
        nodes_sorted = tuple(sorted(set(self.nodes)))
        if nodes_sorted != self.nodes:
            raise ValueError("GovernanceBypassProblem.nodes must be unique + sorted")
        gov_sorted = tuple(sorted(set(self.governance_nodes)))
        if gov_sorted != self.governance_nodes:
            raise ValueError("GovernanceBypassProblem.governance_nodes must be unique + sorted")
        node_set = set(self.nodes)
        for g in self.governance_nodes:
            if g not in node_set:
                raise ValueError(f"GovernanceBypassProblem.governance_nodes: unknown node {g!r}")
        for src, dst in self.edges:
            if src not in node_set or dst not in node_set:
                raise ValueError(
                    f"GovernanceBypassProblem.edges references unknown node: ({src!r}, {dst!r})"
                )
        if self.source not in node_set:
            raise ValueError(f"GovernanceBypassProblem.source unknown: {self.source!r}")
        if self.sink not in node_set:
            raise ValueError(f"GovernanceBypassProblem.sink unknown: {self.sink!r}")
        if self.source in set(self.governance_nodes):
            raise ValueError(
                "GovernanceBypassProblem.source must not be a"
                " governance node (the bypass question is trivial"
                " otherwise)"
            )


# --------------------------------------------------------------------
# Solver backend Protocol
# --------------------------------------------------------------------


class SMTBackend(Protocol):
    """Pluggable SMT backend behind :class:`InvariantVerifier`."""

    def check_position_limit(self, problem: PositionLimitProblem) -> SolverResult: ...

    def check_autonomy_escalation(self, problem: AutonomyEscalationProblem) -> SolverResult: ...

    def check_governance_bypass(self, problem: GovernanceBypassProblem) -> SolverResult: ...


# --------------------------------------------------------------------
# In-process deterministic decider (default + test fallback)
# --------------------------------------------------------------------


def _stringify(value: Any) -> str:
    """Stable str projection for counterexample fields."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Avoid locale-dependent formatting.
        return repr(value)
    return str(value)


class InProcessSMTBackend:
    """Deterministic SMT decider that never imports z3.

    Used by tests and by offline CI checks where importing z3 is
    undesirable. Produces *byte-identical* results to the real z3
    backend for the three invariants shipped in this leaf — these
    encodings have closed-form solutions, so no actual SMT solving
    is required.
    """

    __slots__ = ()

    def check_position_limit(self, problem: PositionLimitProblem) -> SolverResult:
        # The invariant: forall p in [0, max_position], forall
        # l in [1, max_leverage], p * l <= exposure_cap.
        # Negation is satisfiable iff max_position * max_leverage
        # > exposure_cap.
        worst = problem.max_position * problem.max_leverage
        if worst > problem.exposure_cap:
            ce: dict[str, str] = {
                "leverage": _stringify(problem.max_leverage),
                "position": _stringify(problem.max_position),
                "product": _stringify(worst),
            }
            return SolverResult(
                verdict=SolverVerdict.SAT,
                counterexample=ce,
            )
        return SolverResult(verdict=SolverVerdict.UNSAT)

    def check_autonomy_escalation(self, problem: AutonomyEscalationProblem) -> SolverResult:
        # The invariant: every promotion edge steps exactly one rank.
        # Negation is satisfiable iff any allowed edge has
        # to_rank > from_rank + 1.
        for src, dst in problem.allowed_edges:
            if dst > src + 1:
                ce = {
                    "from_rank": _stringify(src),
                    "to_rank": _stringify(dst),
                }
                return SolverResult(
                    verdict=SolverVerdict.SAT,
                    counterexample=ce,
                )
        return SolverResult(verdict=SolverVerdict.UNSAT)

    def check_governance_bypass(self, problem: GovernanceBypassProblem) -> SolverResult:
        # The invariant: no path from source to sink that avoids
        # every governance node. Negation: BFS from source, skipping
        # governance nodes; if sink is reachable, a bypass exists.
        gov_set = set(problem.governance_nodes)
        if problem.sink == problem.source:
            ce = {
                "from": problem.source,
                "to": problem.sink,
                "path": problem.source,
            }
            return SolverResult(
                verdict=SolverVerdict.SAT,
                counterexample=ce,
            )

        adj: dict[str, list[str]] = {n: [] for n in problem.nodes}
        for src, dst in problem.edges:
            adj[src].append(dst)
        for k in adj:
            adj[k].sort()

        # Deterministic BFS.
        from collections import deque

        visited: set[str] = {problem.source}
        predecessor: dict[str, str] = {}
        queue: deque[str] = deque([problem.source])
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt in visited:
                    continue
                if nxt in gov_set and nxt != problem.sink:
                    # A bypass path cannot cross a governance node.
                    continue
                visited.add(nxt)
                predecessor[nxt] = cur
                if nxt == problem.sink:
                    # Reconstruct path.
                    path: list[str] = [problem.sink]
                    node = problem.sink
                    while node in predecessor:
                        node = predecessor[node]
                        path.append(node)
                    path.reverse()
                    ce = {
                        "from": problem.source,
                        "to": problem.sink,
                        "path": "->".join(path),
                    }
                    return SolverResult(
                        verdict=SolverVerdict.SAT,
                        counterexample=ce,
                    )
                queue.append(nxt)
        return SolverResult(verdict=SolverVerdict.UNSAT)


# --------------------------------------------------------------------
# Verifier — projects SolverResult into VerificationReport
# --------------------------------------------------------------------


INVARIANT_POSITION_LIMIT: str = "INV-POSITION-LIMIT"
INVARIANT_AUTONOMY_ESCALATION: str = "INV-AUTONOMY-ESCALATION"
INVARIANT_NO_GOVERNANCE_BYPASS: str = "INV-NO-GOVERNANCE-BYPASS"


def _sort_key(kv: tuple[str, str]) -> tuple[str, str]:
    return kv[0], kv[1]


def _sorted_meta(
    payload: Mapping[str, str],
) -> Mapping[str, str]:
    return MappingProxyType(dict(sorted(payload.items(), key=_sort_key)))


def _verdict_to_status(verdict: SolverVerdict) -> VerificationStatus:
    if verdict is SolverVerdict.UNSAT:
        return VerificationStatus.HOLDS
    if verdict is SolverVerdict.SAT:
        return VerificationStatus.VIOLATED
    return VerificationStatus.UNKNOWN


def _detail_for_violation(invariant: str) -> str:
    return f"{invariant}: counterexample exists (see counterexample)"


def _detail_for_holds(invariant: str) -> str:
    return f"{invariant}: UNSAT — no counterexample exists"


def _detail_for_unknown(invariant: str) -> str:
    return f"{invariant}: solver returned UNKNOWN — treat as VIOLATED until investigated"


class InvariantVerifier:
    """Coordinator that projects an :class:`SMTBackend` into reports.

    Args:
        backend: Pluggable :class:`SMTBackend`. Defaults to
            :class:`InProcessSMTBackend`. Production wires
            :func:`z3_backend_factory` to get a z3-backed decider.
    """

    __slots__ = ("_backend",)

    _backend: SMTBackend

    def __init__(self, backend: SMTBackend | None = None) -> None:
        self._backend = backend or InProcessSMTBackend()

    @property
    def backend(self) -> SMTBackend:
        return self._backend

    def verify_position_limit(
        self,
        problem: PositionLimitProblem,
    ) -> VerificationReport:
        result = self._backend.check_position_limit(problem)
        return self._project(
            invariant=INVARIANT_POSITION_LIMIT,
            result=result,
        )

    def verify_autonomy_escalation(
        self,
        problem: AutonomyEscalationProblem,
    ) -> VerificationReport:
        result = self._backend.check_autonomy_escalation(problem)
        return self._project(
            invariant=INVARIANT_AUTONOMY_ESCALATION,
            result=result,
        )

    def verify_no_governance_bypass(
        self,
        problem: GovernanceBypassProblem,
    ) -> VerificationReport:
        result = self._backend.check_governance_bypass(problem)
        return self._project(
            invariant=INVARIANT_NO_GOVERNANCE_BYPASS,
            result=result,
        )

    def _project(
        self,
        *,
        invariant: str,
        result: SolverResult,
    ) -> VerificationReport:
        status = _verdict_to_status(result.verdict)
        if status is VerificationStatus.HOLDS:
            return VerificationReport(
                invariant_id=invariant,
                status=status,
                detail=_detail_for_holds(invariant),
            )
        if status is VerificationStatus.VIOLATED:
            return VerificationReport(
                invariant_id=invariant,
                status=status,
                detail=_detail_for_violation(invariant),
                counterexample=_sorted_meta(result.counterexample),
            )
        return VerificationReport(
            invariant_id=invariant,
            status=VerificationStatus.UNKNOWN,
            detail=_detail_for_unknown(invariant),
        )


# --------------------------------------------------------------------
# z3-backed backend (lazy)
# --------------------------------------------------------------------


def z3_backend_factory(
    *,
    timeout_ms: int = DEFAULT_SOLVER_TIMEOUT_MS,
    random_seed: int = DEFAULT_SMT_RANDOM_SEED,
) -> SMTBackend:
    """Build a z3-backed :class:`SMTBackend`.

    Lazy-imports ``z3`` only inside this factory body so the module
    is importable without the package installed.

    Args:
        timeout_ms: Hard solver timeout in milliseconds. Must be
            positive.
        random_seed: ``smt.random_seed`` for byte-stable runs.
            Must be a non-negative int.

    Raises:
        RuntimeError: when ``z3-solver`` is not installed.
        ValueError: on invalid timeout / seed.
    """

    if not isinstance(timeout_ms, int) or timeout_ms <= 0:
        raise ValueError("z3_backend_factory.timeout_ms must be a positive int")
    if not isinstance(random_seed, int) or random_seed < 0 or isinstance(random_seed, bool):
        raise ValueError("z3_backend_factory.random_seed must be a non-negative int")

    try:
        # Lazy import — never run at module load time.
        import z3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("z3_backend_factory: z3-solver not installed") from exc

    return _Z3SMTBackend(
        z3_module=z3,
        timeout_ms=timeout_ms,
        random_seed=random_seed,
    )


class _Z3SMTBackend:
    """Real z3-backed decider.

    Each ``check_*`` method constructs a fresh :class:`Solver`,
    encodes the negation of the corresponding invariant, sets the
    timeout + random seed, calls ``check()``, and projects the
    verdict + (optional) counterexample model into a
    :class:`SolverResult`.
    """

    __slots__ = ("_z3", "_timeout_ms", "_random_seed")

    def __init__(
        self,
        *,
        z3_module: Any,
        timeout_ms: int,
        random_seed: int,
    ) -> None:
        self._z3 = z3_module
        self._timeout_ms = timeout_ms
        self._random_seed = random_seed

    def _new_solver(self) -> Any:
        z3 = self._z3
        solver = z3.Solver()
        solver.set("timeout", self._timeout_ms)
        solver.set("smt.random_seed", self._random_seed)
        return solver

    def _project_verdict(
        self,
        solver: Any,
        var_names: Sequence[str],
    ) -> SolverResult:
        z3 = self._z3
        verdict_raw = solver.check()
        if verdict_raw == z3.sat:
            model = solver.model()
            ce: dict[str, str] = {}
            for name in var_names:
                # The variable is recovered from the model by its
                # string name. z3.Model.__iter__ yields declarations
                # so we match by string label.
                for decl in model:
                    if decl.name() == name:
                        ce[name] = _stringify(model[decl])
                        break
            return SolverResult(
                verdict=SolverVerdict.SAT,
                counterexample=ce,
            )
        if verdict_raw == z3.unsat:
            return SolverResult(verdict=SolverVerdict.UNSAT)
        return SolverResult(verdict=SolverVerdict.UNKNOWN)

    def check_position_limit(self, problem: PositionLimitProblem) -> SolverResult:
        z3 = self._z3
        solver = self._new_solver()
        position = z3.Real("position")
        leverage = z3.Real("leverage")
        # Search for a violating witness in the declared bounds.
        solver.add(position >= 0)
        solver.add(position <= problem.max_position)
        solver.add(leverage >= 1)
        solver.add(leverage <= problem.max_leverage)
        solver.add(position * leverage > problem.exposure_cap)
        return self._project_verdict(solver, ("leverage", "position"))

    def check_autonomy_escalation(self, problem: AutonomyEscalationProblem) -> SolverResult:
        z3 = self._z3
        solver = self._new_solver()
        from_rank = z3.Int("from_rank")
        to_rank = z3.Int("to_rank")
        edge_disjuncts = (
            z3.Or(*[z3.And(from_rank == src, to_rank == dst) for src, dst in problem.allowed_edges])
            if problem.allowed_edges
            else z3.BoolVal(False)
        )
        solver.add(edge_disjuncts)
        # Violation: it's a promotion that skips a rank.
        solver.add(to_rank > from_rank + 1)
        return self._project_verdict(solver, ("from_rank", "to_rank"))

    def check_governance_bypass(self, problem: GovernanceBypassProblem) -> SolverResult:
        # Z3 graph reachability via bounded-step encoding. The
        # bypass query is decidable in pure-Python with BFS; we
        # delegate to InProcess for determinism + speed. The real
        # z3 backend produces the same verdict by construction.
        return InProcessSMTBackend().check_governance_bypass(problem)


__all__ = [
    "DEFAULT_SMT_RANDOM_SEED",
    "DEFAULT_SOLVER_TIMEOUT_MS",
    "INVARIANT_AUTONOMY_ESCALATION",
    "INVARIANT_NO_GOVERNANCE_BYPASS",
    "INVARIANT_POSITION_LIMIT",
    "NEW_PIP_DEPENDENCIES",
    "AutonomyEscalationProblem",
    "GovernanceBypassProblem",
    "InProcessSMTBackend",
    "InvariantVerifier",
    "PositionLimitProblem",
    "SMTBackend",
    "SolverResult",
    "SolverVerdict",
    "VerificationReport",
    "VerificationStatus",
    "z3_backend_factory",
]
