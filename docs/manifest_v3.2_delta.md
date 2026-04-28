# DIX VISION v42.2 — MANIFEST v3.2 DELTA

> Additive delta over v3.1 (`docs/manifest_v3.1_delta.md`, PR #36).
> Captures the 7 stress-stabilization findings approved under operator
> decisions I1 / I2 / I3 / I4 / I5 / I6 / I7.
>
> v3.2 is **additive only**. Build Compiler Spec §1.0–§1.1 freeze rules
> apply: no engine renames, no domain collapses, no module removals,
> no event-type explosion. Every v3.2 node sits inside an existing
> engine boundary.
>
> Resolution rule: **v3.2 wins over v3.1 wins over v3** when in conflict.

---

## 0. WHY v3.2 EXISTS

v3.1 closed the 8 *strategic* gaps (Intent / Opponent / Reflexive sim /
Genetics / Regret / Debate / Time hierarchy / Dynamic identity).
Stress-testing v3.1 against real-world load surfaced 7 *resilience*
gaps that, if left unaddressed, would let the system pass replay tests
but break under tick-frequency stress:

| # | Gap | Layer | Failure mode if ignored |
|---|---|---|---|
| I1 | Meta-controller has no `O(1)` degradation path | Meta-Controller policy | Latency spike in `position_sizer` / `execution_policy` starves the hot path; T1 budget violated under load |
| I2 | `regime_router` reads Belief State without hysteresis | Indira regime detection | Regime flapping → strategy oscillation → confidence instability |
| I3 | Pressure Vector `uncertainty` does not include cross-signal entropy | Coherence | System looks safe while internally conflicting (5 BUY + 5 SELL at high individual confidence) |
| I4 | Agents emit `SignalEvent` without structured metadata | Agents (Phase 10.8) | Agents become "expensive plugins with memory" — no structural advantage over plugins |
| I5 | Simulation outputs only `StrategyRanking` | Simulation (Phase 10.1) | Wastes simulation depth; meta-controller can't downweight regime-fragile strategies |
| I6 | Archetypes have no lifecycle (decay / pruning) | Trader Intelligence (Phase 10.2–10.4) | 300 archetypes monotonically grow → memory_tensor bloat → signal noise |
| I7 | Governance is serial under tick frequency | Governance (Phase 7) | PolicyEngine becomes hot-path bottleneck |

v3.2 closes these **without** introducing parallel write paths,
mutable runtime adaptation, or new event types.

**One reframe was required.** I7 as proposed ("cached approval fast
lane") would create a second write path, violating INV-37
(governance is the only authority). The v3.2 reframe replaces it with
a **constant-time PolicyEngine lookup table** — every decision still
ledgered, still auditable, no parallel approval path.

---

## 1. THE 7 FOLD-IN ITEMS

### 1.1 Meta-Controller `O(1)` Fallback Lane (I1 — INV-48)

**Path:** `intelligence_engine/meta_controller/policy/execution_policy.py`
**Phase:** 6.T1b (lands with the H1 sub-package split)
**Spec ID:** META-EP-02

**What it is:**
A precomputed `FALLBACK_POLICY` returned by `execution_policy.decide(...)`
when the per-tick latency budget is exceeded **or** when any of
{Belief State, Pressure Vector, Confidence Engine} are stale / missing.
The fallback is a constant `(allow_trade=False, size=0, reason="DEGRADED")`
in SAFE/PAPER/SHADOW modes, and a constant-time conservative size
table keyed on `(SystemMode, mode_age_bucket)` in CANARY/LIVE/AUTO.

**Why it matters:**
The "off hot path" claim in v3.1 was *logical* (meta-controller does
not run inside `execution_engine.fast_execute`), not *physical*
(`position_sizer.size()` and `execution_policy.decide()` still gate
trade size and existence). Without an `O(1)` fallback, latency in any
upstream coherence module cascades into the hot path.

**Contract:**

```python
# intelligence_engine/meta_controller/policy/execution_policy.py

@dataclass(frozen=True)
class ExecutionDecision:
    allow_trade: bool
    size_bps: int          # 0 in fallback path
    reason: str            # one of "OK" | "DEGRADED" | "DENIED_BY_INTENT" | ...
    fallback: bool         # True iff returned by the fallback lane

class ExecutionPolicy:
    def decide(
        self,
        signal: SignalEvent,
        budget_ns: int,
        clock: Callable[[], int],
    ) -> ExecutionDecision: ...
```

**Invariant (INV-48):** when `clock() - start > budget_ns` OR any
upstream projection is unavailable, `decide()` returns the precomputed
`FALLBACK_POLICY` for the current `SystemMode`. The fallback path
must:
- Have **no** dependency on Belief State / Pressure Vector / Confidence Engine
- Be **constant-time** (lookup, not computation)
- Be **deterministic** (table is content-hashed and ledgered at boot)

**Authority lint rule (B14):**
```
intelligence_engine/meta_controller/policy/execution_policy.py:
  required_paths = {"FALLBACK_POLICY", "_fallback_lane"}
  fallback_lane_imports must subset {core.contracts, registry}
```

**CI test:** `tests/meta_controller/test_execution_policy_fallback.py`
- exercises `decide()` with a synthetic clock advancing past budget
- asserts `fallback=True`, `allow_trade=False` in SAFE/PAPER/SHADOW
- asserts deterministic across 1k repetitions

---

### 1.2 Regime Transition Hysteresis (I2 — INV-49)

**Path:** `intelligence_engine/strategy_runtime/regime_detector.py`
(IND-REG-01, already shipped in Phase 3 — extended in 6.T1e)
**Phase:** 6.T1e (lands after System Intent Engine; extends regime_router with INV-49 gate)
**Spec ID:** IND-REG-02

**What it is:**
Regime transitions in `regime_router` must be **persistence-bounded**
or **delta-bounded** before they take effect. Belief State stays a
pure read-only projection (preserves v3.1 INV); the hysteresis lives
inside the consumer, not the projection.

**Contract:**

```python
@dataclass(frozen=True)
class RegimeHysteresisConfig:
    persistence_ticks: int     # min consecutive ticks in new regime
    confidence_delta: float    # min jump in regime confidence to override persistence
```

**Invariant (INV-49):**
A regime transition `R_old → R_new` is committed iff:
- the new regime has held for `persistence_ticks` consecutive Belief
  State updates **OR**
- `belief.regime_confidence(R_new) - belief.regime_confidence(R_old) >= confidence_delta`

Both `persistence_ticks` and `confidence_delta` are versioned in
`registry/regime_hysteresis.yaml` and reloaded only via Governance
patch pipeline.

**CI test:** `tests/strategy_runtime/test_regime_hysteresis.py`
- 100-tick monotonic sequence: no flap
- 100-tick alternating sequence: zero transitions committed
- step-function jump above `confidence_delta`: transition committed in 1 tick

**Safety gate (SAFE-42):** Belief State updates that would trigger a
regime transition without satisfying INV-49 are logged as
`SystemEvent(kind=REGIME_HOLD)` (deterministic, ledgered).

---

### 1.3 Cross-Signal Entropy in Pressure Vector (I3 — INV-50)

**Path:** `core/coherence/pressure_vector.py`
**Phase:** 6.T1a
**Spec ID:** SCL-04 (refined)

**What it is:**
The Pressure Vector dataclass shape stays **unchanged**:

```python
@dataclass(frozen=True)
class PressureVector:
    perf: float
    risk: float
    drift: float
    latency: float
    uncertainty: float
```

What changes is the **derivation** of `uncertainty`. v3.1 defined it
loosely as "model uncertainty"; v3.2 binds it to a deterministic
composition of two terms:

```
uncertainty = clip(
    α · per_signal_uncertainty
  + β · cross_signal_entropy(active_signal_population),
    0.0, 1.0,
)
```

where `cross_signal_entropy` is the Shannon entropy of the
`{BUY, SELL, HOLD}` distribution across active plugins/agents,
normalized to `[0, 1]`. `α` and `β` are versioned coefficients in
`registry/pressure.yaml` (default `α=0.5, β=0.5`).

**Why it matters:**
Without this, 5 plugins emitting `BUY @ 0.9` and 5 emitting `SELL @ 0.9`
each look "high-confidence, low-uncertainty" individually, and the
Pressure Vector reports low uncertainty — but the system is in
maximum *internal disagreement*. Entropy correctly flags this.

**Invariant (INV-50):**
`uncertainty` MUST be a function of *both* per-signal uncertainty and
the cross-signal directional distribution. The composition function
is pure (no clocks, no PRNG) and CI-pinned.

**CI test:** `tests/coherence/test_pressure_uncertainty_entropy.py`
- 10 signals all `BUY @ 0.9` → uncertainty close to per-signal floor
- 5 `BUY @ 0.9` + 5 `SELL @ 0.9` → uncertainty ≥ entropy threshold
- Same input must produce identical uncertainty across replays

**Safety gate (SAFE-43):** when entropy term ≥ `entropy_high_water`
from `registry/pressure.yaml`, Pressure Vector reports
`safety_modifier = entropy_high_water_modifier` (default `0.5`).
Governance hard-override (=0) still wins (INV-31 refined).

---

### 1.4 Typed Agent Context Schema (I4 — Schema delta + B15 lint)

**Path:** `core/contracts/events.py` (`SignalEvent` extended) +
`registry/agent_context_keys.yaml` (new)
**Phase:** 10.8 (lands with `agents/` namespace activation)
**Spec ID:** AGENT-CTX-01

**What it is:**
Agents are heavier than plugins (stateful, isolated, with bounded
read-only memory). To justify their existence beyond "expensive
plugins with memory", they must be allowed to emit *structured
metadata* alongside `SignalEvent` — and the meta-controller must be
allowed to read it.

The metadata is **typed** (string-only payload, like
`OperatorRequest`) and **CI-validated against an allowlist**.
This preserves replay determinism (no free-form blobs) and INV-45
(memory references are read-only citations of memory_tensor entries,
not weight updates).

**Schema delta:**

```python
# core/contracts/events.py
@dataclass(frozen=True)
class SignalEvent(Event):
    ...
    # NEW (v3.2): optional, typed, allowlist-validated metadata.
    # Empty dict == legacy emitter (plugins, pre-v3.2 agents).
    agent_context: Mapping[str, str] = field(default_factory=dict)
```

**Allowlisted keys** (defined in `registry/agent_context_keys.yaml`):

| Key | Allowed values | Purpose |
|---|---|---|
| `horizon` | `intraday` / `swing` / `position` | Time horizon hint to meta-controller |
| `conviction_type` | `mean_reversion` / `breakout` / `trend` / `liquidity` / `regime` | Strategy archetype hint |
| `memory_ref` | content-hash of memory_tensor row (hex, 32–64 chars) | Read-only citation of supporting memory |
| `regime_assumption` | one of `BeliefState.regime` enum values | Regime under which this signal was emitted |
| `confidence_band` | `low` / `medium` / `high` | Coarse confidence bucket (in addition to numeric `confidence`) |

**Invariant (INV-N/A — schema, enforced by lint):**
- `agent_context` must be a `Mapping[str, str]` (no nesting, no other types)
- Every key MUST appear in `registry/agent_context_keys.yaml`
- Every value MUST satisfy the per-key value rule from the allowlist

**Authority lint rule (B15):**
```
intelligence_engine/agents/*/__init__.py:
  any SignalEvent emitted with agent_context whose keys ∉ allowlist
  ⇒ B15 violation
```

**CI test:** `tests/contracts/test_agent_context_schema.py` +
`tests/test_authority_lint.py::test_b15_*`

**Safety gate (SAFE-44):** unrecognized `agent_context` keys cause
the meta-controller to **drop the agent_context dict entirely** for
that signal (signal still consumed; metadata ignored fail-closed).

**Compatibility:** plugins and pre-v3.2 agents emit empty
`agent_context = {}` and are unaffected.

---

### 1.5 Richer Simulation Outcome (I5 — Schema delta)

**Path:** `core/contracts/events.py` (`SystemEvent.simulation_outcome`
payload extended) + `simulation/strategy_arena/simulation_outcome.py`
**Phase:** 10.1 (lands with Simulation vPro)
**Spec ID:** SIM-OUT-02

**What it is:**
Currently `SystemEvent(kind=SIMULATION_RESULT)` carries only
`StrategyRanking`. v3.2 extends the payload (still `Mapping[str, str]`
with structured sub-fields, still off-bus, still seed-locked) to
include:

```python
@dataclass(frozen=True)
class SimulationOutcome:
    ranking: StrategyRanking                       # existing
    failure_modes: tuple[FailureMode, ...]         # NEW
    regime_performance_map: Mapping[str, float]    # NEW (regime → avg pnl)
    adversarial_breakdowns: tuple[AdversarialBreakdown, ...]  # NEW

@dataclass(frozen=True)
class FailureMode:
    strategy_id: str
    regime: str
    breaking_condition: str   # registry-allowlisted enum
    severity: str             # "warn" | "halt"
```

**Why it matters:**
The meta-controller can now downweight regime-fragile strategies
*before* they reach LIVE — instead of waiting for ledger feedback.
Still off-bus, still seed-locked (INV-40), still SystemEvent-only.

**Invariant** (extends INV-40 — Reflexive Sim determinism): the
extended payload is also deterministic given `(seed, params, order_trace)`.

**Authority lint:** existing B6 (simulation isolation) covers this;
no new rule needed.

**Safety gate (SAFE-45):** any strategy with a `severity=halt`
FailureMode in its current regime cannot transition `CANARY → LIVE`.

**CI test:** `tests/simulation/test_simulation_outcome_extended.py`

---

### 1.6 Archetype Lifecycle (I6 — INV-51)

**Path:** `registry/trader_archetypes.yaml` (schema extended) +
`intelligence_engine/strategy_runtime/archetype_lifecycle.py` (new)
**Phase:** 10.2–10.4 (lands with Trader Intelligence ingest)
**Spec ID:** ARCH-LC-01

**What it is:**
Each archetype declares a lifecycle state and a decay rate.
Auto-demotion happens **offline** in the Learning Engine (preserves
INV-15 replay determinism); promotion / removal is **only** via the
Governance patch pipeline (preserves INV-37).

**Schema extension:**

```yaml
# registry/trader_archetypes.yaml
- id: BREAKOUT_RETAIL_07
  state: ACTIVE              # ACTIVE | DEGRADED | RETIRED
  decay_rate: 0.02           # per-week multiplicative decay applied to performance_score
  performance_score: 0.78    # in [0, 1]; updated only by learning_engine offline run
  feature_space: [...]       # from v3.1 INV-46
  regime_scope: [...]
  correlation_class: ...
```

**Invariant (INV-51):**
- An archetype in `RETIRED` state may not emit signals.
- Auto-demotion (`ACTIVE → DEGRADED` or `DEGRADED → RETIRED`)
  happens **only** in `learning_engine.evaluator.archetype_evaluator`
  during scheduled offline runs.
- Auto-promotion (`DEGRADED → ACTIVE`) is **forbidden** — promotion
  goes through the patch pipeline + HITL gate.
- The `decay_rate` is applied multiplicatively per-week to
  `performance_score` (deterministic, no clocks at runtime — week
  index is a function of ledger sequence number).

**Authority lint rule (B16):**
```
registry/trader_archetypes.yaml:
  every archetype must declare {state, decay_rate, performance_score,
                                 feature_space, regime_scope, correlation_class}
intelligence_engine/agents/archetype_*:
  may not import learning_engine (offline-only ownership)
```

**Safety gate (SAFE-46):** an archetype with `performance_score ≤ 0.30`
held for ≥ 4 consecutive offline runs is automatically transitioned to
`RETIRED` and emits a `SystemEvent(kind=ARCHETYPE_RETIRED)` ledger row
for HITL review.

**CI test:** `tests/strategy_runtime/test_archetype_lifecycle.py`

---

### 1.7 PolicyEngine Constant-Time Lookup (I7 reframed)

**Path:** `governance_engine/control_plane/policy_engine.py`
(GOV-CP-01 — refined, no API change)
**Phase:** 7 (perf-hardening — locked spec)
**Spec ID:** GOV-CP-01-PERF

**What it is — and what it is NOT:**

| Proposed (rejected) | Reframed (approved) |
|---|---|
| Cached "fast-lane" approval bypass | PolicyEngine internal pre-compilation |
| Two write paths (cached vs full) | One write path, faster |
| Cache invalidation logic | No cache (pure precompute at init) |
| Approval bypass for "known safe" signals | Every signal still goes through PolicyEngine |

The reframed change: at `PolicyEngine.__init__`, all `(action,
current_mode, request_kind)` triples have their decisions
**precomputed** into a frozen lookup table:

```python
# governance_engine/control_plane/policy_engine.py
class PolicyEngine:
    def __init__(self, ...):
        self._decision_table: Mapping[
            tuple[OperatorAction, SystemMode, str],
            tuple[bool, RejectionCode | None],
        ] = self._compile_decision_table()
        self._table_hash: str = _content_hash(self._decision_table)

    def decide(self, request: OperatorRequest) -> PolicyDecision:
        key = (request.action, self._current_mode, request.payload.get("kind", ""))
        allowed, reject = self._decision_table[key]   # O(1)
        return PolicyDecision(...)
```

**Why this preserves all invariants:**
- **INV-37 (single authority):** PolicyEngine remains the only authority. No parallel path.
- **INV-12 (single ledger writer):** Every decision still flows through `LedgerAuthorityWriter`. No bypass.
- **INV-15 (replay determinism):** The table is content-hashed at boot and the hash is written into a `SystemEvent(kind=POLICY_TABLE_INSTALLED)` ledger row. Replay reconstructs the same table from the same registry rules.
- **Auditability:** Every `decide()` call still produces a ledger row. The fast path is internal optimization, not a structural change.

**Invariant** (no new INV; refines existing GOV-CP-02 contract):
`PolicyEngine.decide()` is `O(1)` after `__init__`.
Table installation produces a `POLICY_TABLE_INSTALLED` ledger row.
Hot reload of policy registry triggers a new table install + new ledger row.

**Authority lint:** existing GOV-CP-02 lint covers this; no new rule needed.

**Safety gate (SAFE-47):** if the table hash on replay does not match
the original `POLICY_TABLE_INSTALLED` row, the engine **fails closed**
(refuses to start) — equivalent to a registry tamper.

**CI test:** `tests/governance/test_policy_engine_constant_time.py`
- exercises 1k decisions against synthetic table
- asserts replay-determinism on table hash
- asserts table-mismatch fails closed

---

## 2. NEW INVARIANTS (v3.2)

| INV | Subject | One-line rule |
|---|---|---|
| INV-48 | Meta-Controller Fallback | When latency budget exceeded or upstream stale, `execution_policy.decide()` returns precomputed `FALLBACK_POLICY` (constant-time, no Belief/Pressure/Confidence deps) |
| INV-49 | Regime Transition Hysteresis | Belief-State regime transitions commit iff persisted ≥ N ticks OR confidence delta ≥ θ; both versioned in `registry/regime_hysteresis.yaml` |
| INV-50 | Pressure Uncertainty Entropy-Aware | `uncertainty = clip(α·per_signal_uncertainty + β·cross_signal_entropy, 0, 1)`; deterministic, replay-pinned |
| INV-51 | Archetype Lifecycle | Each archetype declares `{state, decay_rate, performance_score}`; auto-demotion offline only; auto-promotion forbidden (patch + HITL only); RETIRED archetypes may not emit signals |

**Refinements to existing invariants:** none. v3.1's INV-31, INV-37,
INV-38, INV-40, INV-46 are all preserved as-is; v3.2 layers on top.

---

## 3. NEW AUTHORITY LINT RULES (v3.2)

| Rule | Module(s) | Purpose |
|---|---|---|
| B14 | `intelligence_engine/meta_controller/policy/execution_policy.py` | Fallback lane present + restricted imports (INV-48) |
| B15 | `intelligence_engine/agents/*` | `SignalEvent.agent_context` keys must subset registry allowlist (I4 schema) |
| B16 | `registry/trader_archetypes.yaml` + `intelligence_engine/agents/archetype_*` | Archetype rows declare lifecycle fields; archetype agents may not import learning_engine (INV-51) |

**No existing rules modified.** B6 (simulation isolation) covers I5
without change. Existing GOV-CP-02 lint covers I7 reframe without
change.

---

## 4. NEW SAFETY GATES (v3.2)

| Gate | Description |
|---|---|
| SAFE-42 | Regime transition that fails INV-49 is logged as `REGIME_HOLD` SystemEvent (no transition, no flap) |
| SAFE-43 | Pressure entropy ≥ `entropy_high_water` ⇒ `safety_modifier = entropy_high_water_modifier` (Governance hard-override still wins) |
| SAFE-44 | Unrecognized `agent_context` keys cause meta-controller to drop the dict for that signal (fail-closed metadata) |
| SAFE-45 | Strategy with `severity=halt` FailureMode in current regime cannot transition CANARY → LIVE |
| SAFE-46 | Archetype with `performance_score ≤ 0.30` for ≥ 4 offline runs auto-transitions to RETIRED + ledger row |
| SAFE-47 | PolicyEngine table-hash mismatch on replay ⇒ engine fails closed (registry tamper protection) |

---

## 5. UPDATED BUILD SEQUENCE

```
PHASES 0–5 ────────────  DONE (PRs #14, #15, #23, #28, #29, #30, #31, #32, #33, #34)
PHASE 6     ────────────  Dashboard OS Control Plane (5 immutable widgets) — DONE (PR #37)
PHASE 6.T1a              Belief State + Pressure Vector              [v3.1 H2] [v3.2 I3]
PHASE 6.T1b              Meta-Controller (H1 split) + Confidence     [v3.1 H1] [v3.2 I1]
                         (lands sub-package layout perception/
                          evaluation/allocation/policy + INV-48
                          fallback lane in policy/execution_policy.py)
PHASE 6.T1c              Reward shaping (auditable, versioned)        [v3.1 H5]
PHASE 6.T1d              System Intent Engine + GOV-CP-07 setter      [v3.1 G1]
PHASE 6.T1e              Regime hysteresis activation                  [v3.2 I2]
                         (extends regime_router with INV-49 gate)
PHASE 7      ──────────  Asset systems / Neuromorphic / Optimization (locked spec)
                         + PolicyEngine constant-time table          [v3.2 I7]
PHASES 8–9   ──────────  Locked spec
PHASE 10     ──────────  Intelligence Depth Layer
   10.1                   Simulation vPro (parallel + arena +
                          adversarial) + richer SimulationOutcome    [v3.2 I5]
   10.2-10.4              Trader Intelligence (ingest + offline +
                          consumer) + Archetype Lifecycle             [v3.2 I6]
   10.5                   Macro Regime Engine
   10.6                   Cross-Asset Coupling
   10.7                   Strategic Execution + Market Impact
   10.8                   agents/ namespace + agent_context schema    [v3.2 I4]
   10.9                   trader_intelligence.proto + registry catalogs
   10.10                  Opponent Model                               [v3.1]
   10.11                  Reflexive Simulation Layer                   [v3.1]
   10.12                  Strategy Genetics                            [v3.1]
   10.13                  Regret / Counterfactual Memory               [v3.1]
   10.14                  Internal Debate Round                        [v3.1]
```

Each sub-phase is one or more PRs with a green CI gate
(`ruff` + `pytest` + `authority_lint` L1/L2/L3/B1..B16).

---

## 6. WHAT v3.2 DOES NOT DO (binding non-goals)

- ❌ Does **not** add a new event type. `SignalEvent` extends with
      a typed optional `agent_context: Mapping[str, str]`;
      `SystemEvent.simulation_outcome` extends with structured
      sub-fields. Both are schema-additive only.
- ❌ Does **not** introduce a parallel governance approval path
      (I7 reframed to single-path constant-time table).
- ❌ Does **not** allow runtime weight updates in agents
      (INV-45 from v3.1 stands).
- ❌ Does **not** allow auto-promotion of archetypes
      (INV-51 — promotion is patch-pipeline + HITL only).
- ❌ Does **not** introduce trained meta-RL coordination
      (still deferred to v42.3 per v3.1 G2 reframe).
- ❌ Does **not** modify Belief State to include hysteresis state
      (hysteresis lives in the consumer `regime_router`,
      Belief stays a pure projection).
- ❌ Does **not** modify the Pressure Vector dataclass shape
      (only the derivation of `uncertainty`).
- ❌ Does **not** reorder phases or add a new phase.

---

## 7. AUDIT TRAIL

| Decision | Source | Operator response |
|---|---|---|
| I1 — Meta-controller fallback (INV-48) | Operator stress review (post PR #36) | Approved ("appove all") |
| I2 — Regime hysteresis (INV-49) | Operator stress review | Approved |
| I3 — Entropy-aware uncertainty (INV-50) | Operator stress review | Approved |
| I4 — Typed agent_context schema | Operator stress review (reframed from free-form) | Approved typed-with-B15 |
| I5 — Richer simulation outcome | Operator stress review | Approved |
| I6 — Archetype lifecycle (INV-51) | Operator stress review | Approved |
| I7 — Governance fast-lane | Operator stress review (REFRAMED — original violated INV-37) | Approved as constant-time table |

---

## 8. NUMBERING NOTE (operator labels vs canonical INV numbers)

The operator's stress review used labels INV-38..41 for I1/I2/I3/I6.
Those numbers were **already taken** in v3.1 (Intent / Opponent Model
/ Reflexive Sim / Strategy Genetics). v3.2 uses the next free range:

| Operator label | Canonical INV in v3.2 |
|---|---|
| INV-38 (fallback) | **INV-48** |
| INV-39 (hysteresis) | **INV-49** |
| INV-40 (entropy uncertainty) | **INV-50** |
| INV-41 (archetype lifecycle) | **INV-51** |

This is bookkeeping, not a substantive change.

---

## 9. CHANGELOG

- **v3.2.0** — initial v3.2 manifest delta. Adds INV-48..51, B14/B15/B16,
  SAFE-42..47, schema deltas for `SignalEvent.agent_context` and
  `SystemEvent.simulation_outcome`. I7 reframed from cached approval
  path to PolicyEngine constant-time table. Phase ordering unchanged.
