# v3.6.0 — BEHAVIOR-P2 — Closed learning loop

**Status:** added
**Scope:** offline-engine behavior, no runtime wiring yet
**PR:** behavior priority P2 — *Trade Result → Score → Adjust Weights*
**Source:** user directive *"You are DONE designing. Now you must
activate behavior."*

This delta closes the third leg of the learning loop. Pieces 1 + 2
already shipped:

| Step           | Module                                                  | Output                          |
|----------------|---------------------------------------------------------|---------------------------------|
| Trade Result   | `execution_engine.protections.feedback`                 | `TradeOutcome`                  |
| Score          | `learning_engine.lanes.reward_shaping`                  | `RewardBreakdown` (J3 component attribution) |
| **Adjust Weights** | **`learning_engine.lanes.weight_adjuster` (this delta)**| **`LearningUpdate` → `SystemEvent(UPDATE_PROPOSED)`** |

The adjuster reads a window of `RewardBreakdown` records, computes per
*tracked weight* the Pearson correlation between the named component
contribution and `shaped_reward`, and proposes bounded nudges as
`LearningUpdate` records. The proposals flow through the existing
`UpdateEmitter → SystemEvent(UPDATE_PROPOSED)` ratchet — Governance
remains the sole authority that promotes a proposal to an applied
update.

## What it does (one window)

For each `WeightBinding`:

1. Project the breakdowns onto two equal-length sequences:
   the binding's component value and `shaped_reward`.
2. If fewer than `min_samples` rows carry the component → no update.
3. If either sequence has zero variance → no update (Pearson undefined).
4. If `|r| < correlation_floor` → no update (uninformative window).
5. Otherwise propose
   `new = clip(current + clip(learning_rate · r, ±max_nudge),
   [min_weight, max_weight])`.
6. If the post-clip value equals the current value (already saturated
   at the boundary in the direction of the nudge) → no update.

Each window evaluation also yields a `WeightAdjustment` diagnostic per
binding so callers (and tests) can inspect *why* a particular nudge
fired without re-running the math.

## Invariants and safety rules introduced

### INV-63 — *Weight adjuster is pure / deterministic / no I/O*

`learning_engine.lanes.weight_adjuster.propose_weight_updates` is a
pure function of `(ts_ns, breakdowns, bindings, config)`. No clock
read, no PRNG, no I/O. Two replays return byte-identical output —
covered by `test_replay_determinism`.

**Enforcement:** unit test + L2 layering (the offline engine cannot
import any runtime engine).

### SAFE-65 — *Per-step nudges are bounded*

A single window evaluation may not move a weight by more than
`max_nudge_per_step` and may not push it outside
`[min_weight, max_weight]`. Caller-supplied weights already outside
that envelope are rejected fail-fast (rather than silently saturating).

**Enforcement:**

- `test_max_nudge_per_step_clips_oversized_raw_step`
- `test_post_clip_bounds_pin_to_max_weight`
- `test_post_clip_bounds_pin_to_min_weight`
- `test_current_value_outside_envelope_rejected`

### SAFE-66 — *Adjuster outputs are proposals, not mutations*

Nothing in `weight_adjuster.py` touches a runtime config or weight
table. The output is a tuple of `LearningUpdate` records routed
through `UpdateEmitter` → `SystemEvent(UPDATE_PROPOSED)`. The existing
governance ratchet (`UPDATE_PROPOSED → UPDATE_APPLIED`) is the sole
path to an applied weight change. This makes every adjustment
reversible and ledger-visible.

**Enforcement:** `test_proposed_updates_flow_through_update_emitter`
asserts the emitted `SystemEvent` carries `sub_kind=UPDATE_PROPOSED`
and the adjuster's diagnostics ride on `meta` for offline calibration.

## Scope (what this delta does NOT do)

- **Does not wire the adjuster into any runtime engine.** No
  `MetaControllerHotPath`, no `ConfidenceEngine`, no `IntelligenceEngine`
  imports the new module. The runtime → governance → runtime feedback
  arc is left for a follow-up PR — that arc requires Governance to
  consume `UPDATE_PROPOSED` and emit `UPDATE_APPLIED`, which is a
  separate behavior change with its own audit surface.
- **Does not replace the existing `RewardBreakdown` shape.** The
  adjuster reads breakdowns as-is; no schema change.
- **Does not introduce a new `SystemEventKind`.** Reuses
  `UPDATE_PROPOSED`.

## Behavior priority status

- P2 — Closed learning loop **(this delta)**
- P3 — System → Governance hard coupling (hazards throttle/block
  execution) — pending
- P4 — Decision trace per trade (why / influences / confidence
  breakdown) — pending
- P5 — Evolution loop (mutation_proposer → sandbox → promotion) —
  pending

CRL (multi-AI arbitration) remains deferred per user direction.
