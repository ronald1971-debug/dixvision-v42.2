# DIX VISION v42.2 — MANIFEST v3.3 DELTA

> Additive delta over v3.2 (`docs/manifest_v3.2_delta.md`, PR #38).
> Captures the 5 self-correction findings approved under operator
> decisions J1 / J2 / J3 / J4 / J5.
>
> v3.3 is **additive only**. Build Compiler Spec §1.0–§1.1 freeze
> rules apply: no engine renames, no domain collapses, no module
> removals, no event-type explosion. Every v3.3 node sits inside
> an existing engine boundary.
>
> Resolution rule: **v3.3 wins over v3.2 wins over v3.1 wins over v3**
> when in conflict.

---

## 0. WHY v3.3 EXISTS

v3.2 closed 7 *resilience* gaps under tick-frequency stress. Stress-
testing v3.2 against **strategic** failure modes surfaced a different
class of risk: the system can be deterministic, governance-bound, and
hot-path-safe — and still drift, because it has no closed loop on
**whether its own lenses are right**.

| # | Gap | Layer | Failure mode if ignored |
|---|---|---|---|
| J1 | Meta-controller is a single deterministic decision surface | Meta-Controller | One mis-tuned `confidence_engine` or `position_sizer` deterministically corrupts every trade — no internal divergence signal |
| J2 | Belief State + Pressure Vector have no calibration feedback loop | Coherence + Learning | Projection drifts silently; reward shaping optimises strategies against a stale lens |
| J3 | Reward shaping output is not component-decomposed in the ledger | Learning | Phase 5 loop optimises a composite, but operators can't tell **what** is being optimised — invisible objective drift |
| J4 | Agents are stateful but have no introspection contract | Agents (Phase 10.8) | Agent debugging becomes archaeology; behavioural opacity violates operator audit principle |
| J5 | Strategy Arena has no sim-vs-live realism check | Simulation + Learning | Closed epistemic loop — Strategy Arena becomes a fantasy optimiser if simulation realism drifts |

v3.3 closes these **without** adding write paths, runtime adaptation,
new event types, or new authority surfaces. Two of the five additions
(J1 shadow path, J5 realism tracker) are *passive observers* — they
**read** the system and **emit `SystemEvent` ledger rows**, never
gate execution.

---

## 1. THE 5 FOLD-IN ITEMS

### 1.1 Shadow Meta-Controller (J1 — INV-52)

**Path:** `intelligence_engine/meta_controller/policy/shadow_policy.py`
**Phase:** 6.T1b (lands alongside the v3.1-H1 sub-package split)
**Spec ID:** META-EP-03

**What it is:**
A second `ExecutionPolicy` instance that runs in parallel with the
primary `execution_policy.py` on the same `SignalEvent` input, but
**never reaches PolicyEngine** and **never produces a trade**.

For every primary `decide(...)` call, the shadow path also computes
its `ExecutionDecision` (using a different parameterisation, a
different fallback policy, or a different scoring weight set).
The two decisions are diffed and emitted as a ledger row:

```python
SystemEvent(
    kind="META_DIVERGENCE",
    payload={
        "primary": ExecutionDecision(...),
        "shadow":  ExecutionDecision(...),
        "delta":   {"size_delta_bps": int, "allow_changed": bool, ...},
        "shadow_version": str,
    },
)
```

**Why it matters:**
v3.2 made meta-controller deterministic and bounded. It did not
remove the underlying risk: a single mis-tuned scoring weight inside
`confidence_engine` or `position_sizer` is **silently** wrong every
tick. Without an internal divergence signal, that error is invisible
until PnL diverges. The shadow path generates a continuous, ledgered
"alternative reality" that Learning can mine for systematic primary
weaknesses — without ever influencing live trading.

**Invariant (INV-52):**
- Shadow `decide()` MUST be invoked for every primary `decide()`
  invocation (no sampling).
- Shadow output MUST NOT reach `governance_engine.*`,
  `execution_engine.*`, or any module that participates in the
  hot path beyond emitting a `SystemEvent`.
- Shadow path MUST be deterministic and replay-stable
  (no PRNG, no clocks beyond the supplied `clock` callable).
- The cost of the shadow path is bounded:
  shadow latency ≤ `2 × primary_budget_ns` (CI-pinned).

**Authority lint rule (B17):**
```
intelligence_engine/meta_controller/policy/shadow_policy.py:
  imports of governance_engine.*           ⇒ B17 violation
  imports of execution_engine.*            ⇒ B17 violation
  any return path that does not emit       ⇒ B17 violation
    SystemEvent(kind="META_DIVERGENCE")
```

**CI test:** `tests/meta_controller/test_shadow_divergence.py`
- 1k synthetic signals exercised through both paths
- asserts shadow never produces an `OrderEvent`
- asserts every `decide()` produces exactly one `META_DIVERGENCE` row
- asserts replay-determinism on `META_DIVERGENCE` payload bytes

**Safety gate (SAFE-48):**
If shadow latency exceeds `2 × primary_budget_ns` for `≥ N` consecutive
ticks (config in `registry/meta_controller.yaml`), the shadow path
is **suspended** (primary continues unaffected) and a
`SHADOW_SUSPENDED` `SystemEvent` is emitted. Learning treats the gap
as a known calibration window.

---

### 1.2 Belief + Pressure Calibration Loop (J2 — INV-53)

**Path:** `learning_engine/calibration/coherence_calibrator.py` (new
package `learning_engine/calibration/`)
**Phase:** 6.T1c (initial offline run); deeper integration in Phase 5
follow-on
**Spec ID:** CAL-01

**What it is:**
An **offline, governance-gated** module that periodically scans the
ledger window `[t_n - W, t_n]` and computes per-window calibration
divergence for the two key projections:

```python
@dataclass(frozen=True)
class CalibrationReport:
    window_start: int       # ledger sequence
    window_end:   int
    belief_vs_actual:  float    # belief.regime vs realised regime classifier (0=match, 1=opposite)
    pressure_vs_realized: float # pressure.{risk,latency,perf} vs realised constraints
    components: Mapping[str, float]   # per-dimension breakdown
    calibrator_version: str
```

The report is emitted as a `SystemEvent`:

```python
SystemEvent(kind="CALIBRATION_REPORT", payload=asdict(report))
```

**Why it matters:**
v3.1 made Belief State + Pressure Vector pure read-only projections.
v3.2 added entropy-aware uncertainty (INV-50). Neither closed the
loop on whether the *projection itself* is right. If Belief
mis-classifies regime systematically, every downstream consumer
(`regime_router`, `confidence_engine`, sizing) is deterministically
wrong — and the system optimises *strategies* against a broken lens
forever.

**Invariant (INV-53):**
- The calibrator MUST run offline only
  (no clocks, no production-bus reads at runtime).
- The calibrator MUST NOT write to Belief State or Pressure Vector.
- The calibrator MUST emit `CALIBRATION_REPORT` exactly once per
  configured window, ledgered.
- A `CALIBRATION_REPORT` row with `belief_vs_actual ≥ θ` or
  `pressure_vs_realized ≥ θ` (thresholds versioned in
  `registry/calibration.yaml`) **MUST** be inspected by HITL via
  the patch pipeline before any belief/pressure parameter change.

**Reward-shaping coupling:**
The reward shaper consumes the most recent `CALIBRATION_REPORT` and
adds a `lens_calibration_penalty` component to every reward
breakdown (see §1.3 J3 below). This penalises strategies whose
ledgered behaviour is conditioned on a known-miscalibrated lens
window — preventing Learning from optimising into the calibration
gap.

**Authority lint rule:** none new — existing `B14`-class import
restrictions on `learning_engine/*` keep the calibrator offline.

**CI test:** `tests/calibration/test_coherence_calibrator.py`
- synthetic ledger window with known-good belief → divergence ≈ 0
- synthetic ledger window with regime-flipped belief → divergence ≈ 1
- replay-determinism on `CALIBRATION_REPORT` bytes

**Safety gate (SAFE-49):**
If `belief_vs_actual ≥ critical_threshold` from
`registry/calibration.yaml` for `≥ N` consecutive windows, the
calibrator emits `BELIEF_CALIBRATION_HALT` `SystemEvent` and the
`OperatorRequest` queue is auto-populated with a HITL inspection
ticket. Live trading is **not** halted by the calibrator alone
(governance authority preserved); only operators can halt.

---

### 1.3 Per-Component Reward Audit (J3 — INV-47 EXTENSION)

**Path:** `learning_engine/performance_analysis/reward_shaping.py`
(extends v3.1 H5)
**Phase:** 6.T1c
**Spec ID:** REW-AUD-01 (extension of REW-AUD-00 from v3.1)

**What it is:**
v3.1 INV-47 made the reward shaping function versioned and the raw
PnL retained alongside the shaped reward. v3.3 J3 extends this to
require the shaper to ledger the **per-component decomposition** of
each shaped reward, not just the composite scalar:

```python
@dataclass(frozen=True)
class RewardBreakdown:
    raw_pnl:                float
    components:             Mapping[str, float]  # e.g. {"pnl": 0.7, "risk_adj": -0.2, ...}
    shaping_version:        str
    lens_calibration_penalty: float = 0.0   # from J2 INV-53
    sim_overconfidence_penalty: float = 0.0 # from J5 INV-55

    @property
    def shaped_reward(self) -> float:
        return sum(self.components.values()) \
             - self.lens_calibration_penalty \
             - self.sim_overconfidence_penalty
```

Every shaped reward emits a corresponding `SystemEvent`:

```python
SystemEvent(kind="REWARD_BREAKDOWN", payload=asdict(breakdown))
```

**Why it matters:**
A composite reward is the *invisible objective function* of the
entire learning loop. Without per-component decomposition, operators
can see *that* the system is optimising and *how well*, but not
*toward what*. Component-level ledger rows make objective drift
visible: if `risk_adj` quietly grows from 5% to 40% of the composite
over months, that is a silent change in what the system is
maximising — and operators get to see it.

**Invariant (INV-47, EXTENDED):**
The original v3.1 INV-47 stands. v3.3 adds:
- The shaper MUST emit `REWARD_BREAKDOWN` for every reward applied
  to a training batch (no sampling, no aggregation).
- `RewardBreakdown.components` keys MUST subset the registered
  allowlist in `registry/reward_components.yaml` (versioned).
- `RewardBreakdown.shaping_version` MUST match a ledgered
  `REWARD_SHAPER_INSTALLED` row produced by the patch pipeline.

**Authority lint rule (B18):**
```
learning_engine/performance_analysis/reward_shaping.py:
  RewardBreakdown.components keys ∉ registry/reward_components.yaml
    ⇒ B18 violation (build-time check on the registry,
       runtime check fail-closed)
```

**Dashboard surface:**
A new read-only widget `RewardComponentTrend.tsx` (Phase 6.T1c
follow-on) renders the per-component time series from
`REWARD_BREAKDOWN` rows. **Read-only** — does not violate the v3
"5 immutable widgets" rule because it sits in the existing
read-only panel surface.

**CI test:** `tests/learning/test_reward_breakdown.py`
- every shaped reward produces one `REWARD_BREAKDOWN` row
- unknown component key fails build (B18)
- replay-determinism on `REWARD_BREAKDOWN` bytes
- composition: `shaped_reward = Σ components − penalties`

---

### 1.4 Agent Introspection Contract (J4 — INV-54)

**Path:** `intelligence_engine/agents/_base.py` (new abstract base)
+ `core/contracts/agent.py` (new `AgentIntrospection` Protocol)
**Phase:** 10.8 (lands with `agents/` namespace activation)
**Spec ID:** AGENT-INTRO-01

**What it is:**
Every `agents/` entry MUST implement two read-only introspection
methods:

```python
# core/contracts/agent.py
@runtime_checkable
class AgentIntrospection(Protocol):
    def state_snapshot(self) -> Mapping[str, str]:
        """
        Return a compact, deterministic snapshot of agent internal
        state (keys subset of registry/agent_state_keys.yaml).
        Pure: no side effects, no clock, no PRNG.
        """

    def recent_decisions(self, n: int) -> Sequence[AgentDecisionTrace]:
        """
        Return the last ≤ n decisions made by the agent, with
        rationale tags. Read from a bounded internal ring buffer.
        """

@dataclass(frozen=True)
class AgentDecisionTrace:
    signal_id:        str
    direction:        str   # "BUY" | "SELL" | "HOLD"
    confidence:       float
    rationale_tags:   tuple[str, ...]   # subset of registry/agent_rationale_tags.yaml
    memory_refs:      tuple[str, ...]   # content-hashes of memory_tensor rows
```

A new `OperatorRequest` (`AGENT_INTROSPECT`, GOV-CP-07) lets HITL
sample any agent's `state_snapshot()` + `recent_decisions(n)` on
demand and emits the result as a `SystemEvent` ledger row.

**Why it matters:**
v3.1 INV-45 already forbids runtime weight updates inside
`agents/`. v3.3 adds the *transparency* counterpart:
plugins are transparent because they are stateless; agents are
permitted statefulness, so they pay for it with mandatory
introspection. Otherwise debugging an agent under operator audit
becomes archaeology — exactly the failure mode operator §4 flagged.

**Invariant (INV-54):**
- Every concrete class in `intelligence_engine/agents/` MUST satisfy
  `AgentIntrospection` (CI-checked via abstract base + lint).
- `state_snapshot()` and `recent_decisions()` MUST be pure functions
  (no side effects, no event emission, no PRNG, no clock).
- `recent_decisions()` MUST be O(1) per call (ring buffer; no
  scanning of memory_tensor or ledger).
- `state_snapshot()` keys MUST subset `registry/agent_state_keys.yaml`.

**Authority lint rule (B19):**
```
intelligence_engine/agents/*/__init__.py:
  class missing state_snapshot() or recent_decisions()
    ⇒ B19 violation
  state_snapshot() that imports core/clocks.* or random.*
    ⇒ B19 violation
```

**CI test:** `tests/agents/test_introspection_contract.py`
- every concrete agent class implements both methods
- two snapshots taken with no intervening events are equal
  (purity check)
- `recent_decisions(n)` truncates to ≤ n
- snapshot keys subset of registry allowlist

---

### 1.5 Sim-Realism Tracker + Reward Penalty (J5 — INV-55)

**Path:** `learning_engine/calibration/sim_realism_tracker.py`
(same `learning_engine/calibration/` package as J2)
**Phase:** 10.1 (Simulation vPro lands the data); Phase 6.T1c (reward
penalty wiring)
**Spec ID:** CAL-02

**What it is:**
An **offline, governance-gated** companion to J2. Where J2 calibrates
the *projections* (Belief, Pressure), J5 calibrates the *simulation
itself* by computing per-strategy / per-regime / per-window
divergence between simulated and realised outcomes:

```python
@dataclass(frozen=True)
class SimRealismReport:
    strategy_id:  str
    regime:       str
    window_start: int
    window_end:   int
    sim_pnl_z:    float   # z-score of sim PnL vs realised PnL
    sim_drawdown_z: float
    sim_sharpe_z: float
    sim_overconfidence: float   # ∈ [0, 1] — degree of "good in sim, bad in live"
    tracker_version: str
```

Emitted as `SystemEvent(kind="SIM_REALISM_REPORT", payload=...)`.

**Reward-shaping coupling (binding):**
The reward shaper consumes the latest `SimRealismReport` for each
`(strategy_id, regime)` pair and applies a
`sim_overconfidence_penalty = w · sim_overconfidence` term to the
reward of any training batch generated through that strategy in
that regime. Strategies that look strong in the Strategy Arena but
fail in CANARY/LIVE pay a deterministic penalty in their own
training loop — without any runtime pathway change.

**Why it matters:**
v3.1 wired Strategy Arena and Reflexive Sim. v3.2 added richer
`SimulationOutcome`. Neither closed the realism loop. Without J5,
the Strategy Arena can self-reinforce strategies that are simulation-
local optima but live-poor — exactly the closed epistemic loop
operator §5 flagged.

**Invariant (INV-55):**
- The tracker MUST run offline only.
- The tracker MUST NOT write to Belief State, Pressure Vector,
  Simulation, or Strategy Lifecycle FSM.
- The tracker emits `SIM_REALISM_REPORT` exactly once per configured
  window per `(strategy_id, regime)` pair.
- The reward shaper's `sim_overconfidence_penalty` MUST be sourced
  exclusively from the most recent ledgered `SIM_REALISM_REPORT`
  for the matching `(strategy_id, regime)`; no live recomputation.

**Authority lint rule:** none new — covered by existing `B6`
(simulation isolation) for the read side, `B18` (J3 reward
allowlist) for the write side.

**CI test:** `tests/calibration/test_sim_realism_tracker.py`
- synthetic windows where sim ≈ live → `sim_overconfidence ≈ 0`
- windows where sim » live → `sim_overconfidence` near 1
- reward penalty composes deterministically with
  `RewardBreakdown.sim_overconfidence_penalty`
- replay-determinism on `SIM_REALISM_REPORT` bytes

**Safety gate (SAFE-50):**
A strategy with `sim_overconfidence ≥ critical_threshold` for
`≥ N` consecutive windows cannot transition `CANARY → LIVE`
(extends SAFE-45 from v3.2 with a calibration-driven gate).

---

## 2. NEW INVARIANTS (v3.3)

| INV | Subject | One-line rule |
|---|---|---|
| INV-52 | Shadow Meta-Controller | Shadow `decide()` runs for every primary `decide()`, never reaches governance/execution, only emits `META_DIVERGENCE` |
| INV-53 | Belief+Pressure Calibration | Offline calibrator emits `CALIBRATION_REPORT` per window; never writes coherence; HITL-gated patch path only |
| INV-54 | Agent Introspection | Every `agents/` class implements pure `state_snapshot()` + `recent_decisions(n)`, ring-buffered, allowlist-keyed |
| INV-55 | Sim-Realism + Penalty | Offline tracker emits `SIM_REALISM_REPORT`; reward shaper applies deterministic `sim_overconfidence_penalty` sourced only from ledgered reports |

**Refinement to existing invariant:**
- **INV-47 (v3.1 H5)** is **extended** (not replaced) by J3:
  per-component `RewardBreakdown` ledger row + B18 allowlist + dashboard widget.

---

## 3. NEW AUTHORITY LINT RULES (v3.3)

| Rule | Module(s) | Purpose |
|---|---|---|
| B17 | `intelligence_engine/meta_controller/policy/shadow_policy.py` | Shadow path may not import `governance_engine.*` or `execution_engine.*`; every `decide()` MUST emit `META_DIVERGENCE` (INV-52) |
| B18 | `learning_engine/performance_analysis/reward_shaping.py` + `registry/reward_components.yaml` | `RewardBreakdown.components` keys must subset registry allowlist (INV-47 extended) |
| B19 | `intelligence_engine/agents/*` + `registry/agent_state_keys.yaml` | Concrete agent classes implement `AgentIntrospection`; `state_snapshot()` is pure (INV-54) |

**No existing rules modified.**

---

## 4. NEW SAFETY GATES (v3.3)

| Gate | Description |
|---|---|
| SAFE-48 | Shadow path latency `> 2× primary_budget_ns` for ≥ N ticks ⇒ shadow auto-suspended; `SHADOW_SUSPENDED` ledgered (primary unaffected) |
| SAFE-49 | `belief_vs_actual ≥ critical_threshold` for ≥ N windows ⇒ `BELIEF_CALIBRATION_HALT` SystemEvent + auto HITL ticket (operator-only halt, no automated trading halt) |
| SAFE-50 | `sim_overconfidence ≥ critical_threshold` for ≥ N windows ⇒ strategy blocked from `CANARY → LIVE` (extends SAFE-45) |

---

## 5. UPDATED BUILD SEQUENCE

```
PHASES 0–5 ────────────  DONE (PRs #14, #15, #23, #28, #29, #30, #31, #32, #33, #34)
PHASE 6     ────────────  Dashboard OS Control Plane (5 immutable widgets) — DONE (PR #37)
PHASE 6.T1a              Belief State + Pressure Vector              [v3.1 H2] [v3.2 I3]
PHASE 6.T1b              Meta-Controller (H1 split) + Confidence     [v3.1 H1] [v3.2 I1] [v3.3 J1]
                         (lands sub-package layout perception/
                          evaluation/allocation/policy + INV-48
                          fallback lane in policy/execution_policy.py
                          + INV-52 shadow lane in
                          policy/shadow_policy.py)
PHASE 6.T1c              Reward shaping + per-component audit         [v3.1 H5] [v3.3 J2 J3 J5]
                         (lands learning_engine/calibration/
                          coherence_calibrator.py + sim_realism_tracker.py
                          + RewardBreakdown ledger row + B18 lint)
PHASE 6.T1d              System Intent Engine + GOV-CP-07 setter      [v3.1 G1]
PHASE 6.T1e              Regime hysteresis activation                  [v3.2 I2]
                         (extends regime_router with INV-49 gate)
PHASE 7      ──────────  Asset systems (forex / stocks / crypto / memecoin) (locked spec)
                         + PolicyEngine constant-time table          [v3.2 I7]
PHASE 8      ──────────  Neuromorphic + AutoLearn (locked spec)
PHASE 9      ──────────  Optimization layer (locked spec)
PHASE 10     ──────────  Intelligence Depth Layer
   10.1                   Simulation vPro + richer SimulationOutcome  [v3.2 I5]
                          + sim_realism_tracker upstream data         [v3.3 J5]
   10.2-10.4              Trader Intelligence + Archetype Lifecycle    [v3.2 I6]
   10.5                   Macro Regime Engine
   10.6                   Cross-Asset Coupling
   10.7                   Strategic Execution + Market Impact
   10.8                   agents/ namespace + agent_context schema    [v3.2 I4]
                          + AgentIntrospection contract               [v3.3 J4]
   10.9                   trader_intelligence.proto + registry catalogs
   10.10                  Opponent Model                               [v3.1]
   10.11                  Reflexive Simulation Layer                   [v3.1]
   10.12                  Strategy Genetics                            [v3.1]
   10.13                  Regret / Counterfactual Memory               [v3.1]
   10.14                  Internal Debate Round                        [v3.1]
```

**No phase reordering.** All v3.3 nodes attach to phases already
scheduled by v3.1 or v3.2.

---

## 6. WHAT v3.3 DOES NOT DO (binding non-goals)

- ❌ Does **not** add a new event type. `META_DIVERGENCE`,
  `CALIBRATION_REPORT`, `REWARD_BREAKDOWN`, `SIM_REALISM_REPORT`,
  `SHADOW_SUSPENDED`, `BELIEF_CALIBRATION_HALT` are all **`SystemEvent`
  kinds**, not new event classes (schema-additive only).
- ❌ Does **not** introduce a parallel governance approval path
  (J1 shadow lane is *non-acting* — never reaches PolicyEngine).
- ❌ Does **not** allow runtime adaptation in agents
  (INV-45 still stands; J4 is read-only introspection).
- ❌ Does **not** allow the calibrator to write Belief State or
  Pressure Vector (INV-53 — calibrator is read-only on coherence).
- ❌ Does **not** allow the realism tracker to write Strategy
  Lifecycle FSM (INV-55 — penalty applied via reward shaping
  only, not via direct state mutation).
- ❌ Does **not** modify the reward function shape — only adds
  per-component decomposition + two penalty terms.
- ❌ Does **not** reorder phases or add a new phase.

---

## 7. AUDIT TRAIL

| Decision | Source | Operator response |
|---|---|---|
| J1 — Shadow meta-controller (INV-52) | Operator 8-point review (post-PR-#38) | Approved ("aprove all") |
| J2 — Belief+Pressure calibration loop (INV-53) | Operator 8-point review | Approved |
| J3 — Per-component reward audit (INV-47 EXTENSION) | Operator 8-point review | Approved as INV-47 extension (not new INV) |
| J4 — Agent introspection contract (INV-54) | Operator 8-point review | Approved |
| J5 — Sim-realism tracker + reward penalty (INV-55) | Operator 8-point review | Approved |

---

## 8. NUMBERING NOTE

v3.1 used INV-37..47 (Intent / Opponent / Reflexive / Genetics /
Regret / Debate / Identity / Time-tier / Agent-non-adaptive /
Archetype-orthogonality / Reward-audit).
v3.2 used INV-48..51 (Fallback / Hysteresis / Entropy /
Archetype-lifecycle).
v3.3 uses **INV-52..55** for J1 / J2 / J4 / J5. **J3 is not a new
INV** — it is an extension of v3.1 INV-47 (reward auditability).

---

## 9. CHANGELOG

- **v3.3.0** — initial v3.3 manifest delta. Adds INV-52..55 (J1/J2/J4/J5),
  extends INV-47 (J3), adds B17/B18/B19, SAFE-48..50, schema-additive
  `SystemEvent` kinds for `META_DIVERGENCE`, `CALIBRATION_REPORT`,
  `REWARD_BREAKDOWN`, `SIM_REALISM_REPORT`. New package
  `learning_engine/calibration/` (offline, governance-gated). New
  module `intelligence_engine/meta_controller/policy/shadow_policy.py`.
  New abstract base `intelligence_engine/agents/_base.py` + Protocol
  `core/contracts/agent.py`. No phase reordering. No new event types.
  No engine boundary changes.
