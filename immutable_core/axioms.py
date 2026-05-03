"""Axiom registry — Python form of the v42.2 neuromorphic axioms.

Each axiom is a frozen value object with:

  * ``id`` — canonical identifier (``INV-15``, ``SAFE-01``, …) used
    in lint rules, contracts, tests, and PR descriptions;
  * ``kind`` — :class:`AxiomKind` (INVARIANT vs SAFETY);
  * ``label`` — one-line human-readable summary;
  * ``introduced_in`` — file path where the axiom was originally
    defined (manifest delta or contract module). Always relative
    to the repo root so a reader can resolve it without searching.

Every INV-* / SAFE-* identifier the codebase actually uses must
appear in :data:`AXIOM_REGISTRY`; tests/test_immutable_core_axioms.py
enforces that there are no orphan references.

The registry is intentionally exhaustive rather than aspirational —
adding a new axiom is a registry edit + a manifest-delta doc edit,
not a free-form code comment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class AxiomKind(StrEnum):
    """Classification of an axiom."""

    INVARIANT = "INVARIANT"
    SAFETY = "SAFETY"


@dataclass(frozen=True, slots=True)
class Axiom:
    """One axiom in the immutable-core registry."""

    id: str
    kind: AxiomKind
    label: str
    introduced_in: str


def _i(id_: str, label: str, introduced_in: str) -> Axiom:
    return Axiom(id=id_, kind=AxiomKind.INVARIANT, label=label, introduced_in=introduced_in)


def _s(id_: str, label: str, introduced_in: str) -> Axiom:
    return Axiom(id=id_, kind=AxiomKind.SAFETY, label=label, introduced_in=introduced_in)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Order: INV-01..INV-72 then SAFE-01..SAFE-69. The label is a short
# summary suitable for log lines; full descriptions live in the
# manifest delta files referenced by ``introduced_in``.
# ---------------------------------------------------------------------------

_AXIOMS: tuple[Axiom, ...] = (
    # Invariants
    _i("INV-01", "Single source of truth for time on the hot path", "system/time_source.py"),
    _i("INV-08", "Domain isolation — engines do not import each other", "docs/directory_tree.md"),
    _i("INV-11", "Only typed events cross engine boundaries", "core/contracts/events.py"),
    _i("INV-12", "Bounded knowledge store — LRU + compaction", "system/knowledge_store.py"),
    _i("INV-15", "Replay determinism — no clock, no PRNG in pure logic", "system/time_source.py"),
    _i(
        "INV-17",
        "Periodic snapshot reconstruction is deterministic",
        "system/state_reconstructor.py",
    ),
    _i("INV-20", "Ledger storage tiers — hot ring + cold facade", "system/ledger/"),
    _i(
        "INV-31",
        "Strategy lifecycle FSM is ledger-replayed",
        "intelligence_engine/strategy_lifecycle.py",
    ),
    _i(
        "INV-33",
        "Conflict resolver collapses balanced BUY/SELL to HOLD",
        "intelligence_engine/conflict_resolver.py",
    ),
    _i("INV-37", "Patch pipeline is the only adaptive-mutation seam", "system_engine/patches/"),
    _i(
        "INV-38",
        "System Intent Engine is the only operator->governance write path",
        "intelligence_engine/system_intent.py",
    ),
    _i("INV-39", "Operator dashboard emits intents; governance executes", "ui/dashboard_routes.py"),
    _i("INV-40", "Hot path purity — no I/O, no clock", "execution_engine/hot_path/fast_execute.py"),
    _i("INV-41", "FastRiskCache version is monotonic", "system/fast_risk_cache.py"),
    _i("INV-42", "FastRiskCache halts on staleness", "system/fast_risk_cache.py"),
    _i(
        "INV-43",
        "Order state machine is total + deterministic",
        "execution_engine/lifecycle/order_state_machine.py",
    ),
    _i("INV-44", "Hazard sensors are pure / no I/O", "system_engine/hazards/"),
    _i("INV-45", "Health monitors degrade fail-fast", "system_engine/health/"),
    _i("INV-46", "System state is composed, never mutated in place", "system_engine/state.py"),
    _i(
        "INV-47",
        "DecisionTrace audit reflects per-component contributions",
        "core/contracts/decision_trace.py",
    ),
    _i(
        "INV-48",
        "Meta-controller hot-path fallback fires on elapsed_ns budget",
        "intelligence_engine/meta/hot_path.py",
    ),
    _i(
        "INV-49",
        "Regime router uses hysteresis gate",
        "intelligence_engine/coherence/regime_router.py",
    ),
    _i("INV-50", "BeliefState entropy is bounded", "core/contracts/coherence/belief_state.py"),
    _i(
        "INV-51", "PressureVector projection is pure", "core/contracts/coherence/pressure_vector.py"
    ),
    _i(
        "INV-52",
        "Shadow policy mirrors live without acting",
        "intelligence_engine/policy/shadow_policy.py",
    ),
    _i(
        "INV-53",
        "Calibration / reliability hook on every decision",
        "intelligence_engine/coherence/calibrator.py",
    ),
    _i(
        "INV-54",
        "Reward shaping per-component is auditable",
        "intelligence_engine/reward/shaping.py",
    ),
    _i(
        "INV-55",
        "Position sizer audit is J3-aligned",
        "intelligence_engine/allocation/position_sizer.py",
    ),
    _i(
        "INV-56",
        "Triad Lock — Decider / Executor / Approver are decoupled",
        "tools/authority_lint.py",
    ),
    _i("INV-57", "SCVS bidirectional closure — every source has a consumer", "system_engine/scvs/"),
    _i(
        "INV-58",
        "SCVS source liveness FSM escalates HAZ on critical drop",
        "system_engine/scvs/source_manager.py",
    ),
    _i(
        "INV-59",
        "Per-packet schema/staleness guard with AI fallback audit",
        "system_engine/scvs/schema_guard.py",
    ),
    _i(
        "INV-60",
        "Authority matrix is the single conflict-resolution table",
        "system_engine/authority/matrix.py",
    ),
    _i("INV-61", "Constraint engine is the single rule-graph oracle", "core/constraint_engine/"),
    _i(
        "INV-62",
        "Almgren-Chriss strategic execution is deterministic",
        "execution_engine/strategic/almgren_chriss.py",
    ),
    _i(
        "INV-63",
        "Weight adjuster is pure / deterministic / no I/O",
        "docs/manifest_v3.6.0_delta.md",
    ),
    _i(
        "INV-64",
        "Hazard throttle is pure / deterministic / no I/O",
        "docs/manifest_v3.6.1_delta.md",
    ),
    _i(
        "INV-65", "Decision trace is pure / deterministic / no I/O", "docs/manifest_v3.6.2_delta.md"
    ),
    _i(
        "INV-66",
        "Patch pipeline orchestrator is pure / deterministic / no I/O",
        "docs/manifest_v3.6.3_delta.md",
    ),
    _i(
        "INV-67",
        "Cognitive subsystems are advisory; replay verifies typed bus only",
        "docs/manifest_v3.6.4_delta.md",
    ),
    _i(
        "INV-68",
        "ExecutionEngine.execute(intent) is the only path to a venue",
        "execution_engine/execution_gate.py",
    ),
    _i("INV-69", "Every typed event carries produced_by_engine", "core/contracts/events.py"),
    _i(
        "INV-70",
        "Adaptive mutations gated by LearningEvolutionFreezePolicy",
        "core/contracts/learning_evolution_freeze.py",
    ),
    _i(
        "INV-71",
        "Authority symmetry — B27/B28 lint enforces matrix at runtime",
        "tools/authority_lint.py",
    ),
    _i(
        "INV-72",
        "Operator-approval edge gates cognitive SignalEvent emission",
        "intelligence_engine/cognitive/approval_edge.py",
    ),
    # Safety axioms
    _s(
        "SAFE-01",
        "Kill switch is the single chokepoint for SystemMode.LOCKED",
        "system/kill_switch.py",
    ),
    _s(
        "SAFE-09",
        "Sandbox patch execution is bounded + reverted on failure",
        "system_engine/patches/sandbox.py",
    ),
    _s(
        "SAFE-12",
        "Memory overflow watchdog forces degrade",
        "system_engine/health/memory_overflow.py",
    ),
    _s(
        "SAFE-15",
        "Anomaly detector escalates to HAZ on persistent breach",
        "system_engine/hazards/anomaly_detector.py",
    ),
    _s(
        "SAFE-18",
        "Operator override always routes through governance",
        "registry/authority_matrix.yaml",
    ),
    _s(
        "SAFE-23",
        "Approval queue is ledger-backed and crash-safe",
        "intelligence_engine/cognitive/approval_queue.py",
    ),
    _s(
        "SAFE-26",
        "Plugin lifecycle transitions require operator approval",
        "dashboard_backend/control_plane/memecoin_control_panel.py",
    ),
    _s(
        "SAFE-43",
        "FRED parser fuzz: malformed payload cannot panic",
        "system_engine/feeds/fred_macro.py",
    ),
    _s(
        "SAFE-47",
        "BLS parser fuzz: malformed payload cannot panic",
        "system_engine/feeds/bls_macro.py",
    ),
    _s(
        "SAFE-61",
        "Almgren-Chriss caps participation rate",
        "execution_engine/strategic/almgren_chriss.py",
    ),
    _s(
        "SAFE-62",
        "Almgren-Chriss caps slice notional",
        "execution_engine/strategic/almgren_chriss.py",
    ),
    _s(
        "SAFE-65",
        "Closed learning loop respects freeze policy",
        "core/contracts/learning_evolution_freeze.py",
    ),
    _s(
        "SAFE-66",
        "Weight adjustments are bounded per cycle",
        "intelligence_engine/learning/weight_adjuster.py",
    ),
    _s(
        "SAFE-67",
        "Hazard throttle output is monotonic in severity",
        "system_engine/coupling/hazard_throttle.py",
    ),
    _s(
        "SAFE-68",
        "Hazard throttle never amplifies traffic",
        "system_engine/coupling/hazard_throttle.py",
    ),
    _s(
        "SAFE-69",
        "Patch orchestrator emits ledger record on every mutation",
        "system_engine/patches/orchestrator.py",
    ),
)


_seen_ids: set[str] = set()
for _axiom in _AXIOMS:
    if _axiom.id in _seen_ids:
        raise RuntimeError(f"duplicate axiom id {_axiom.id!r} in immutable_core/axioms.py")
    _seen_ids.add(_axiom.id)
del _seen_ids, _axiom


AXIOM_REGISTRY: Mapping[str, Axiom] = MappingProxyType({axiom.id: axiom for axiom in _AXIOMS})


def get_axiom(axiom_id: str) -> Axiom:
    """Return the axiom for ``axiom_id``.

    Raises :class:`KeyError` with a clear message if unknown — the
    registry is the only source of truth, so an unknown id is always
    a typo or an orphan reference.
    """

    try:
        return AXIOM_REGISTRY[axiom_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown axiom id {axiom_id!r}; register it in immutable_core/axioms.py first"
        ) from exc


def is_axiom(axiom_id: str) -> bool:
    """Return whether ``axiom_id`` resolves to a registered axiom."""

    return axiom_id in AXIOM_REGISTRY
