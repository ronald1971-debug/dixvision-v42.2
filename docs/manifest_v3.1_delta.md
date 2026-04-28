# DIX VISION v42.2 — MANIFEST v3.1 DELTA

> Additive delta over the v3 manifest (PR #35). Captures the 8 v3.1
> fold-in items approved under operator decisions G1 / G2 / G3 / G4.
>
> v3.1 is **additive only**. Build Compiler Spec §1.1 freeze rules apply:
> no engine renames, no domain collapses, no module removals. Every v3.1
> node sits inside an existing engine boundary.
>
> Resolution rule: if v3 and v3.1 disagree, **v3.1 wins**.

---

## 0. WHY v3.1 EXISTS

After v3 landed (PR #35, directory_tree v3), the operator raised 8
additional gaps:

1. No "system will" — no module that says what the system **wants** to
   become.
2. Simulation lacks reflexivity (market reacts to your orders).
3. Indira lacks opponent modelling — predicts price, not other traders.
4. No clear time hierarchy of intelligence (ms → months).
5. HITL is passive; operator can't inject macro bias or strategic goals.
6. Evolution lacks genetic structure (mutation operators, crossover).
7. Memory_tensor lacks counterfactual / regret tracking.
8. No internal debate among `agents/` before decision.

v3.1 closes these gaps **without breaking the locked spec**. Six of the
eight items are direct additive extensions to existing engines. Two
required reframing (`G1`, `G2` below) before they were safe to land.

---

## 1. THE 8 FOLD-IN ITEMS

### 1.1 System Intent Engine (G1 reframed)

**Path:** `core/coherence/system_intent.py`
**Phase:** 6.T1d
**Spec ID:** SCL-06

**What it is:**
A frozen read-only projection — like Belief State + Pressure Vector —
that exposes:

```python
@dataclass(frozen=True)
class SystemIntent:
    objective: Literal["risk_adjusted_growth", "absolute_return", "capital_preservation", "exploration"]
    focus: tuple[str, ...]              # e.g. ("crypto_microstructure", "fx_carry")
    risk_mode: Literal["defensive", "balanced", "aggressive"]
    horizon: Literal["intraday", "short_term", "medium_term", "long_term"]
    intent_id: str                      # ledger row hash
    set_at: int                         # ledger sequence number
```

**The reframe (G1):** the **operator writes intent**, not the system.

| Component | Authority |
|---|---|
| Operator (via dashboard) | proposes `IntentTransition` request through `GOV-CP-07 OperatorInterfaceBridge` |
| Governance | validates request, writes `IntentTransition` event to ledger |
| `system_intent.py` | exposes the *read* projection of the latest committed intent |
| Meta-Controller | **reads** intent via L3 Protocol; routes regime + sizing accordingly |
| Indira / Execution / Learning / System | never write intent |

This is the resolution to the operator's "human as strategic layer above
AI" point. It becomes **explicit** in the architecture, instead of
implicit.

**Invariant (INV-38):** `system_intent.py` has no setter API. Only
`governance_engine.control_plane.state_transition_manager` may emit
`IntentTransition` events to the ledger. Replay reads intent from the
event log. CI test: `tests/coherence/test_intent_readonly.py`.

**Authority lint extension (B8):**
```
core/coherence/system_intent.py:
  disallowed_imports = {execution_engine, intelligence_engine, learning_engine,
                        evolution_engine, system_engine, simulation, dashboard}
  allowed_imports    = {core.contracts.*, state.ledger.reader,
                        governance_engine.control_plane (Protocol types only)}
```

---

### 1.2 Opponent Model

**Path:** `intelligence_engine/opponent_model/`
**Phase:** 10.10
**Spec IDs:** OPP-01..03

**Modules:**
- `behavior_predictor.py` — predicts likely trader actions from
  microstructure + Trader Intelligence archetype distribution
- `crowd_density.py` — estimates positioning crowdedness (using lead-lag
  + volume + funding-rate features)
- `strategy_detector.py` — infers in-market strategy populations (e.g.
  "breakout traders trapped at $X — fade-the-move opportunity")

**Boundary:** lives inside Indira. Reads from
`intelligence_engine/meta/trader_archetypes.py` (consumer side of
Trader Intelligence). Emits no new event type — its outputs feed
`meta_controller/strategy_selector.py` and `confidence_engine.py` via
in-engine call.

**Invariant (INV-39):** opponent model outputs are **probabilistic
estimates**, never authoritative. They modulate confidence and
sizing, never bypass Strategy Lifecycle FSM or Governance approval.

**Authority lint extension (B9):**
```
intelligence_engine/opponent_model/*.py:
  disallowed_imports = {execution_engine, governance_engine,
                        learning_engine, evolution_engine, system_engine,
                        simulation, dashboard}
  allowed_imports    = {core.contracts.*, intelligence_engine.meta,
                        intelligence_engine.macro, intelligence_engine.cross_asset,
                        state.ledger.reader, state.memory_tensor.trader_patterns}
```

---

### 1.3 Reflexive Simulation Layer

**Path:** `simulation/reflexive_layer/`
**Phase:** 10.11
**Spec IDs:** REFL-01..03

**Modules:**
- `impact_feedback.py` — own-order price impact loop (your trade moves
  the book → your edge changes)
- `liquidity_decay.py` — liquidity drying up under your flow
- `crowd_density_sim.py` — alpha decay due to popularity / crowding

**Why:** without this, the system breaks at scale. A backtest that
assumes "I act → market is external" produces unrealistic Sharpe at
size. `reflexive_layer/` injects "I act → market reacts → my edge
changes" loops into the simulation.

**Boundary:** lives inside `simulation/`, sits alongside
`adversarial/`. Subject to all simulation invariants: offline only
(INV-33), seeded (SAFE-29), no live-broker imports (B6).

**Invariant (INV-40):** reflexive simulation runs are deterministic
given the master seed, the impact-model parameters, and the order
trace. CI test: `tests/simulation/test_reflexive_replay_deterministic.py`.

**Existing B6 lint applies** — no extension needed.

---

### 1.4 Strategy Genetics

**Path:** `evolution_engine/genetic/`
**Phase:** 10.12
**Spec IDs:** GEN-01..03

**Modules:**
- `mutation_operators.py` — parameter / structural mutations on strategy
  definitions
- `crossover.py` — strategy crossover (combine two parent strategies'
  parameter spaces under fitness inheritance)
- `fitness_inheritance.py` — inherited fitness accounting (a child
  strategy's prior is its parents' Sharpe distribution)

**Boundary:** lives inside `evolution_engine/`. Outputs go through the
existing **patch pipeline** (`evolution_engine/patch_pipeline/`) →
Governance approval bridge → APPROVED / REJECTED / ROLLED_BACK. No
auto-deploy.

**Invariant (INV-41):** genetic outputs are
`PROPOSED_STRATEGY_DEFINITION` events, written to ledger, gated by the
patch pipeline. Mutated strategies enter the lifecycle FSM at
`PROPOSED` only. No genetic algorithm bypasses paper-trading
gate (SAFE-03 / GOV-G17) or HITL approval for LIVE promotion.

**Existing patch_pipeline lint applies** — no extension needed.

---

### 1.5 Regret / Counterfactual Memory

**Path:** `state/memory_tensor/regret/`
**Phase:** 10.13
**Spec IDs:** REG-01..03

**Modules:**
- `missed_opportunity.py` — paths not taken (signal emitted, executed
  SKIP, hindsight PnL > 0)
- `almost_trades.py` — near-miss tracking (signal nearly fired but fell
  below confidence threshold)
- `regret_log.py` — append-only regret events to ledger

**Boundary:** lives inside `state/memory_tensor/`. Read by
`learning_engine/performance_analysis/reward_shaping.py` to inform
risk-adjusted training signal (the system learns from "the trades it
should have made", not just the trades it made).

**Invariant (INV-42):** regret entries are derived offline from ledgered
events; never live-evaluated. Their derivation is bit-identical on
replay (INV-15). CI test:
`tests/memory_tensor/test_regret_replay_deterministic.py`.

**Authority lint extension (B10):**
```
state/memory_tensor/regret/*.py:
  disallowed_imports = {execution_engine, intelligence_engine,
                        governance_engine, system_engine, simulation,
                        dashboard}
  allowed_imports    = {core.contracts.*, state.ledger.reader,
                        learning_engine (Protocol types only)}
```

---

### 1.6 Internal Debate Round (G2 reframed)

**Path:** `intelligence_engine/meta_controller/debate_round.py`
**Phase:** 10.14
**Spec ID:** MC-06

**The reframe (G2):** debate is a **deterministic scoring round**, NOT
trained meta-RL coordination.

**What it is:**
For each candidate signal, agents in the registered `agents/` namespace
emit a `(stance, confidence)` pair:

```python
@dataclass(frozen=True)
class AgentStance:
    agent_id: str           # e.g. "scalper_v1"
    stance: Literal["agree", "disagree", "abstain"]
    confidence: float       # [0.0, 1.0]
    rationale_hash: str     # ledger row reference
```

`debate_round.py` aggregates stances via deterministic weighted scoring
(weights from `registry/agents.yaml`, deterministic across replay).
Output is a `DebateConsensus` consumed by
`meta_controller/confidence_engine.py` as the "alignment" component of
the composite confidence formula.

No agent learns to mimic another. No policy-gradient feedback. No
multi-agent meta-RL. Tier 3 remains deferred (extras_opinion §4).

**Invariant (INV-43):** debate output is a pure function of `(signals,
agent_stances, weights)`. No clocks, no PRNGs. Replay reproduces
debate consensus bit-for-bit. CI test:
`tests/meta_controller/test_debate_deterministic.py`.

**Existing B4 lint extended** to permit `meta_controller/debate_round.py`
to import `intelligence_engine.agents.*` (read-only).

---

### 1.7 Time Hierarchy Doctrine (no new modules)

**Path:** Manifest §X (this section)
**Phase:** Documented now; binding immediately.

**Time tiers:**

| Tier | Range | Owner module(s) |
|---|---|---|
| T0 ms | sub-tick | `execution_engine/hot_path/fast_execute.py` (T1-pure) |
| T1 sec | per-tick | `execution_engine/lifecycle/`, `intelligence_engine/signal_pipeline.py`, `meta_controller/*` |
| T2 min | per-bar | `intelligence_engine/strategy_runtime/scheduler.py`, `dyon` health monitors |
| T3 hour | aggregate | `intelligence_engine/portfolio/`, `simulation/strategy_arena/` cadence |
| T4 day | offline batch | `learning_engine/*`, `learning_engine/trader_abstraction/embedder.py` |
| T5 week | strategic | `evolution_engine/*`, `evolution_engine/patch_pipeline/` |
| T6 month | identity / mission | `core/coherence/system_intent.py` (operator-set), `governance_engine.services.patch_pipeline` cadence |

**Invariant (INV-44):** modules in tier T(n) may not block on tier
T(m≥n+2) results. Specifically:
- Hot path (T0) may not block on T1+ (per existing T1 lint).
- T1 may not block on T3+ (e.g. meta_controller may not synchronously
  call into strategy_arena).
- T4+ runs on schedulers, never on the runtime bus.

CI test: `tests/architecture/test_time_hierarchy_no_blocking.py`.

---

### 1.8 Dynamic Identity (no new modules)

**Path:** Manifest §X (this section)
**Phase:** Documented now; emergent from existing FSMs.

**What it is:**
The "system as a different kind of trader under different regimes" is
NOT a new module. It is an **emergent property** of:

1. `system_intent.py` — sets strategic direction (T6, operator-written)
2. `meta_controller.regime_router` — reads Belief State regime tag
3. `strategy_runtime.state_machine` — Strategy Lifecycle FSM gates which
   strategies are LIVE
4. `simulation.strategy_arena` — promotes/retires strategies based on
   regime-conditional fitness

The active subset of LIVE strategies under the current
`(intent, regime)` pair **is** the system's identity at any moment.
"Trend follower → mean reversion" is just `strategy_arena` retiring
trend strategies + promoting mean-reversion ones in a ranging regime,
filtered by intent.

**Non-goal:** no `system_identity_engine.py` module. Adding one would
violate Build Compiler Spec §1.1 (no inventing new architectural
shapes).

---

## 2. NEW INVARIANTS (v3.1)

| INV | Subject | One-line rule |
|---|---|---|
| INV-38 | System Intent | Operator writes via GOV-CP-07; system reads only |
| INV-39 | Opponent Model | Probabilistic only; never authoritative |
| INV-40 | Reflexive Sim | Deterministic given (seed, params, order trace) |
| INV-41 | Strategy Genetics | Outputs gated by patch pipeline + paper trading + HITL |
| INV-42 | Regret Memory | Derived offline from ledger; replay-deterministic |
| INV-43 | Debate Round | Pure function of (signals, stances, weights); no PRNG |
| INV-44 | Time Hierarchy | T(n) may not synchronously block on T(n+2)+ |
| INV-45 | Agent Non-Adaptive | Agents may store memory but may not update decision-logic weights inside the runtime — adaptation only via Learning Engine → Governance → deployment |
| INV-46 | Archetype Orthogonality | Each archetype declares `feature_space` / `regime_scope` / `correlation_class`; CI gate enforces max pairwise correlation between LIVE archetypes |
| INV-47 | Reward Shaping Auditable | Shaping function is versioned, signed, and ledgered; raw reward stream is retained; reverse-mapping `(shaped → raw, version)` must always be possible |
| INV-31 (refined) | Pressure Safety Modifier | `safety_modifier` is continuous in `[0, 1]`, monotonically non-increasing in pressure; Governance retains a hard `0` override |

---

## 3. NEW AUTHORITY LINT RULES (v3.1)

| Rule | Module(s) | Purpose |
|---|---|---|
| B8 | `core/coherence/system_intent.py` | Read-only Intent enforcement |
| B9 | `intelligence_engine/opponent_model/*` | Indira-internal isolation |
| B10 | `state/memory_tensor/regret/*` | Memory-tensor isolation |
| B11 | `intelligence_engine/agents/*` | No in-runtime weight updates (INV-45) |
| B12 | `registry/trader_archetypes.yaml` schema | Each archetype must declare `feature_space` / `regime_scope` / `correlation_class` (INV-46) |
| B13 | `learning_engine/performance_analysis/reward_shaping.py` | Shaping function must register version + raw-reward retention (INV-47) |
| B4 (extended) | `intelligence_engine/meta_controller/debate_round.py` | Permitted to import `agents/` (read-only) |

Existing B6 (simulation isolation) covers `simulation/reflexive_layer/`.
Existing patch_pipeline lint covers `evolution_engine/genetic/`.

---

## 4. NEW SAFETY GATES (v3.1)

| Gate | Description |
|---|---|
| SAFE-32 | Intent transition requires HITL approval at GOV-CP-07 |
| SAFE-33 | Opponent model confidence <0.6 fails closed (no signal modulation) |
| SAFE-34 | Reflexive sim cannot run without `master_seed` |
| SAFE-35 | Genetic crossover children enter at `PROPOSED` only |
| SAFE-36 | Regret entries are append-only; no mutation post-write |
| SAFE-37 | Debate round timeout (deterministic budget) → fall back to non-debate confidence |
| SAFE-38 | Pressure `safety_modifier` is continuous `[0, 1]`; Governance can hard-set to `0` (INV-31 refined) |
| SAFE-39 | Agent runtime modules cannot import any optimizer / autograd library (INV-45 enforcement) |
| SAFE-40 | Archetype CI gate: max LIVE pairwise correlation ≤ threshold from `registry/risk.yaml` (INV-46) |
| SAFE-41 | Reward shaping run produces `(shaped_reward, raw_reward, shaping_version, shaping_hash)` ledger row (INV-47) |

---

## 5. UPDATED BUILD SEQUENCE

```
PHASES 0–5 ────────────  DONE (PR #14, 15, 23, 28, 29, 30, 31, 32, 33, 34)
PHASE 6 ────────────────  Dashboard OS Control Plane (5 immutable widgets)  ← NEXT
PHASE 6.T1a              Belief State + Pressure Vector
PHASE 6.T1b              Meta-Controller + Confidence Engine
PHASE 6.T1c              Reward shaping
PHASE 6.T1d              System Intent Engine + GOV-CP-07 setter           [v3.1]
PHASES 7–9 ──────────────  Asset systems / Neuromorphic / Optimization (locked spec)
PHASE 10 ──────────────  Intelligence Depth Layer
   10.1                   Simulation vPro (parallel + arena + adversarial)
   10.2-10.4              Trader Intelligence (ingest + offline + consumer)
   10.5                   Macro Regime Engine
   10.6                   Cross-Asset Coupling
   10.7                   Strategic Execution + Market Impact
   10.8                   agents/ namespace
   10.9                   trader_intelligence.proto + registry catalogs
   10.10                  Opponent Model                                    [v3.1]
   10.11                  Reflexive Simulation Layer                        [v3.1]
   10.12                  Strategy Genetics                                 [v3.1]
   10.13                  Regret / Counterfactual Memory                    [v3.1]
   10.14                  Internal Debate Round                             [v3.1]
```

Each sub-phase is one or more PRs. Each PR ends with a green CI gate
(`ruff` + `pytest` + `authority_lint` L1/L2/L3/B1/B2/B3/B4/B5/B6/B7
+ new B8/B9/B10/B11/B12/B13).

---

## 6. WHAT v3.1 DOES NOT DO (binding non-goals)

- ❌ Does not let the system auto-write its own intent (G1 reframed).
- ❌ Does not introduce trained meta-RL or multi-agent policy
  gradients (G2 reframed).
- ❌ Does not add a `system_identity_engine.py` module — identity is
  emergent.
- ❌ Does not add a 5th cross-engine event type. Intent flows as
  `IntentTransition` (a SYSTEM_EVENT subtype) via the existing bus.
- ❌ Does not let opponent model write to authoritative state.
- ❌ Does not let genetic outputs auto-deploy.
- ❌ Does not modify Phase 0–9 sequence; Phase 10.10–10.14 are
  intra-Phase-10 sub-phases per E1.
- ❌ Does not let agents update their own decision-logic weights at
  runtime (H3 / INV-45). Agent state is memory only — adaptation goes
  through Learning Engine → Governance → deployment.
- ❌ Does not allow correlated archetypes to coexist as LIVE without
  CI gate approval (H4 / INV-46).
- ❌ Does not allow reward shaping without retained raw-reward stream
  and versioned, ledgered shaping function (H5 / INV-47).

---

## 7. AUDIT TRAIL

| Decision | Source | Reframe / scope |
|---|---|---|
| G1 — System Intent | Operator chat 2026-04-21 | "System decides what it wants" → "Operator writes intent via GOV-CP-07" |
| G2 — Internal Debate | Operator chat 2026-04-21 | "Multi-agent argue" → "Deterministic stance scoring round" |
| G3 — Fold v3.1 now | Operator chat 2026-04-21 | "now" |
| G4 — Phase 6 in parallel | Operator chat 2026-04-21 | "yes" — separate branch |
| H1 — Meta-controller internal split | Operator chat 2026-04-21 (post-v3.1 stabilization review) | Sub-package layout `perception / evaluation / allocation / policy` inside `meta_controller/` — audit separation, no new engine boundary |
| H2 — Continuous safety modifier | Operator chat 2026-04-21 | INV-31 refined: `safety_modifier ∈ [0, 1]` continuous, monotonic, Governance hard-override preserved |
| H3 — Agent non-adaptive constraint | Operator chat 2026-04-21 | INV-45 + B11: agents may store memory but no in-runtime weight updates |
| H4 — Archetype orthogonality | Operator chat 2026-04-21 | INV-46 + B12: declared `feature_space` / `regime_scope` / `correlation_class`, CI gate on max pairwise correlation between LIVE archetypes |
| H5 — Reward shaping invertible/auditable | Operator chat 2026-04-21 | INV-47 + B13: versioned shaping, raw-reward retention, reverse-mappable, ledgered |

---

## 8. STABILIZATION REFINEMENTS (v3.1, post-review)

Five refinements applied during the v3.1 stabilization review. None
introduces a new engine, event type, or phase reordering. All tighten
determinism, auditability, or explainability of items already approved
in §1.

### 8.1 Meta-Controller Internal Split (H1)

**Concern:** as written, `meta_controller/` owns regime routing,
strategy selection, confidence aggregation, position sizing, and
final SKIP/SHADOW/EXECUTE policy — too much power in one boundary, even
with B4 lint. Failure mode: meta-controller becomes a god-object;
plugins/agents become dumb signal emitters; explainability collapses.

**Refinement:** sub-package layout *inside* the existing
`intelligence_engine/meta_controller/` boundary. No new engine, no new
lint rule — audit separation only.

```
intelligence_engine/meta_controller/
  ├── perception/
  │   └── regime_router.py        # MC-01 (was top-level)
  ├── evaluation/
  │   ├── strategy_selector.py    # MC-02 (was top-level)
  │   ├── confidence_engine.py    # MC-03 (was top-level)
  │   └── debate_round.py         # MC-06 (v3.1)
  ├── allocation/
  │   └── position_sizer.py       # MC-04 (was top-level)
  └── policy/
      └── execution_policy.py     # MC-05 (was top-level)
```

Directory tree v3.1 reflects this layout. Phase 6.T1b lands the four
sub-packages in this shape. B4 lint applies to every sub-package
uniformly.

### 8.2 Continuous Safety Modifier (H2 — INV-31 refined)

**Concern:** original Pressure Vector contract used a binary
`safety_modifier ∈ {0, 1}`. At threshold this produces:
- regime whiplash (full trading ↔ zero trading on minor pressure deltas)
- missed recovery opportunities (cannot scale back in gradually)
- oscillation around the cutoff

**Refinement:** `safety_modifier ∈ [0, 1]` continuous.

Contract guarantees (CI-enforced):
1. **Monotonic non-increasing** in the dominant pressure component —
   higher pressure never *increases* the modifier.
2. **Governance hard-override** — GOV-CP-01..02 retains the right to
   force `safety_modifier = 0` regardless of computed value (SAFE-38).
3. **Replay-deterministic** — modifier is a pure function of
   `(pressure_vector, intent, governance_overrides)`. No PRNG, no
   clock.
4. **No upward bias** — modifier may not be raised by any module
   except the canonical pressure-to-modifier mapping in
   `core/coherence/performance_pressure.py`.

INV-31 is refined, not replaced. The original "safety modifier exists
and is bounded in `[0, 1]`" property is preserved; the binary
restriction is removed.

### 8.3 Agent Non-Adaptive Constraint (H3 — INV-45)

**Concern:** stateful `agents/` (per C2) can implicitly learn / adapt
/ bias outputs in-runtime. Without a guardrail, this smuggles RL into
the deterministic runtime path — violates INV-15 + the
"adaptation = offline + Governance-approved" philosophy.

**Refinement:** new invariant INV-45.

> Agents MAY:
> - read past events from the ledger
> - maintain in-process memory (e.g. recent regime tag, last fill,
>   short-window features)
> - emit `AgentStance` / `SIGNAL_EVENT` outputs as a deterministic
>   function of `(inputs, memory)`
>
> Agents MUST NOT:
> - update decision-logic weights / parameters at runtime
> - import any optimizer, autograd, or gradient-step library
> - mutate any state visible to other agents
>
> Agent adaptation only happens via:
> Learning Engine (offline) → Governance approval → deployment

Enforcement:
- **B11 lint rule** — disallowed imports for `intelligence_engine/agents/*`:
  `{torch.optim, torch.autograd, jax, optax, sklearn (fit-style),
   any custom "learn" / "update_weights" / "backward" symbol}`
- **SAFE-39** runtime gate — fails closed at startup if any agent
  module imports a forbidden symbol.

### 8.4 Archetype Orthogonality (H4 — INV-46)

**Concern:** Phase 10 ships up to 300 trader archetypes. Without
orthogonality, the system gets registry bloat, overlapping strategies,
correlation explosion — the confidence engine's independence
assumption breaks, the conflict resolver scales badly, and Phase 10
becomes noise instead of intelligence.

**Refinement:** every archetype declares three orthogonality keys.

```yaml
# registry/trader_archetypes.yaml schema (v3.1)
- archetype_id: "breakout_momentum_us_equities"
  feature_space: ["price_breakout_zscore", "volume_spike", "vix_regime"]
  regime_scope: ["TRENDING_UP", "VOLATILE"]
  correlation_class: "momentum.breakout"
  ...
```

CI enforcement:
- **B12 lint rule** — yaml schema validation: missing
  `feature_space` / `regime_scope` / `correlation_class` → archetype
  rejected at registry load.
- **SAFE-40 runtime gate** — max pairwise correlation between LIVE
  archetypes must remain ≤ threshold from `registry/risk.yaml`. Excess
  correlation → Strategy Lifecycle FSM demotes the lowest-Sharpe
  member to `SHADOW` (deterministic tiebreak).
- Correlation matrix is computed offline by
  `learning_engine/trader_abstraction/` and ledgered as a
  `CorrelationCheckpoint` event.

This preserves the value of 300 archetypes (diverse hypothesis pool)
without the failure mode (300 noisy, correlated emitters).

### 8.5 Reward Shaping Invertible / Auditable (H5 — INV-47)

**Concern:** reward shaping influences which strategies the Evolution
Engine *generates*, which transitively biases runtime behaviour. Even
though shaping itself is offline, its bias is durable. Without an
audit trail, the operator can't reconstruct *why* the system evolved
the way it did.

**Refinement:** new invariant INV-47.

> Every reward shaping run produces a ledger row containing:
> - `shaping_version` — semver of the shaping function
> - `shaping_hash` — content hash of the shaping function source
> - `raw_reward_window` — reference to retained raw rewards
> - `shaped_reward_window` — reference to shaped rewards
> - `inverse_mapping_test_result` — CI-asserted invertibility check
>
> The raw reward stream is **never overwritten** by shaped values.
> A reverse mapping `(shaped, version) → (raw, applied_function)` must
> always be reconstructible from the ledger.

Enforcement:
- **B13 lint rule** —
  `learning_engine/performance_analysis/reward_shaping.py` must register
  every shaping function via `ShapingRegistry.register(version, hash,
  fn)` and emit a `ShapingRun` ledger row per execution.
- **SAFE-41 runtime gate** — missing `ShapingRun` row → reward shaping
  run is rejected; raw rewards used directly with a warning logged.

---

## 9. CHANGELOG

| Version | Change |
|---|---|
| v3 (PR #35) | Tier 1 + agents/ + Phase 10 from 20-extras |
| **v3.1** | **+ 8 fold-in items (G1–G4): Intent / Opponent / Reflexive / Genetics / Regret / Debate / Time Hierarchy / Dynamic Identity. + 5 stabilization refinements (H1–H5): meta-controller internal split, continuous safety modifier (INV-31 refined), agent non-adaptive constraint (INV-45), archetype orthogonality (INV-46), reward shaping audit trail (INV-47)** |

End of v3.1 manifest delta. Build Compiler Spec §1.0–§1.1 freeze rules
apply.
