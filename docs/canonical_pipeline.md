# DIX VISION — Canonical Runtime Pipeline

> Architectural reference. Non-normative prose. Authoritative
> contracts live in the manifest deltas (v3 → v3.4) and the
> `tools/authority_lint.py` rule set. This document **names** the
> dominant loop so new contributors and agents do not have to
> reverse-engineer it from PRs.

---

## 1. The dominant loop

```
                          ┌──────────────────────┐
                          │   Market Data        │
                          │   (broker / sensor)  │
                          └─────────┬────────────┘
                                    │  MarketEvent
                                    ▼
   ┌────────────────┐     ┌──────────────────────┐
   │  System Engine │────▶│  Intelligence Engine │
   │  (Dyon)        │     │  (Indira)            │
   │  hazards /     │     │  signals + meta-     │
   │  drift / state │     │  controller          │
   └────────┬───────┘     └─────────┬────────────┘
            │                       │  SignalEvent +
            │  SystemEvent           │  ExecutionDecision
            │  (telemetry)           ▼
            │             ┌──────────────────────┐
            │             │  Governance Engine   │
            │             │  approves / rejects /│
            │             │  constrains          │
            │             └─────────┬────────────┘
            │                       │  PolicyDecision
            │                       ▼
            │             ┌──────────────────────┐
            │             │  Execution Engine    │
            │             │  orders + fills +    │
            │             │  lifecycle FSM       │
            │             └─────────┬────────────┘
            │                       │  ExecutionEvent
            ▼                       ▼
   ┌────────────────────────────────────────────┐
   │  State + Ledger (hash-chained, replay)     │
   └─────────────────┬──────────────────────────┘
                     │  read-only
                     ▼
              ┌────────────────────┐
              │  Learning Engine   │
              │  (offline reward   │
              │   shaping + cal)   │
              └─────────┬──────────┘
                        │  PatchProposal
                        ▼
              ┌────────────────────┐
              │  Evolution Engine  │
              │  (gated patch FSM) │
              └─────────┬──────────┘
                        │  via Governance approval
                        ▼
                   (mutates registry / config —
                    never the runtime path directly)
```

**One-line summary.** Market Data → Dyon → Indira → Governance →
Execution → Ledger → Learning → Evolution → (back into Governance).

## 2. Stage ownership

| Stage           | Owning engine          | Sole producer of                              |
|-----------------|------------------------|-----------------------------------------------|
| Market ingest   | broker adapters        | `MarketEvent`                                 |
| Hazard / drift  | `system_engine`        | `SystemEvent(kind=HAZARD_*)`, `DRIFT_*`       |
| Signal          | `intelligence_engine`  | `SignalEvent` *(B22 enforced)*                |
| Meta-decision   | `intelligence_engine`  | `ExecutionDecision`, `META_AUDIT`, `META_DIVERGENCE` |
| Belief / Pressure | `intelligence_engine`| `BELIEF_STATE_SNAPSHOT`, `PRESSURE_VECTOR_SNAPSHOT` |
| Approval        | `governance_engine`    | `PolicyDecision`, `MODE_TRANSITION`, `APPROVAL` |
| Order / fill    | `execution_engine`     | `ExecutionEvent` *(B21 enforced)*             |
| Ledger write    | `governance_engine`    | hash-chained ledger row (sole writer; INV-37) |
| Reward shaping  | `learning_engine`      | `RewardBreakdown`                              |
| Patch proposal  | `evolution_engine`     | `PatchProposal` (gated by Governance)         |

## 3. Dominance rule

The runtime path is **the only path that can move capital**. Every
arrow in §1 is one of:

* a typed bus event (`MarketEvent` / `SignalEvent` / `ExecutionEvent`
  / `SystemEvent`), or
* a Protocol-mediated control-plane call (e.g. `policy_engine.decide`).

There is no "shortcut" between two engines. In particular:

* `governance_engine` cannot import `execution_engine` (B1 + B20).
* `governance_engine` cannot construct an `ExecutionEvent` (B21).
* `learning_engine` and `evolution_engine` do not appear in the hot
  path (L2 / L3); they sit on the **read-only** side of the ledger
  and propose changes to **registry / config**, never to the runtime
  call chain.
* `learning_engine` and `evolution_engine` do not import each other
  (L1).

## 4. The triad

Three engines own the runtime trade:

* **Indira** — `intelligence_engine` — **decides**.
* **Executor** — `execution_engine` — **executes**.
* **Governance** — `governance_engine` — **approves / rejects /
  constrains**, never trades.

`system_engine` (Dyon) is the **fourth** engine in the runtime tier.
It is a sensor / state surface — it does not place orders and does
not approve them. Its outputs feed Indira (informs decisions) and
the ledger (informs the offline tier).

The triad lock is INV-56 (`docs/manifest_v3.4_delta.md` §1.1) and is
enforced by lint rules **B1 / B17 / B20 / B21 / B22**.

## 5. What this document is not

* It is **not** a list of file paths. The directory tree
  (`docs/directory_tree.md`) is the authority on layout.
* It is **not** a list of invariants. The manifest deltas are.
* It is **not** a sequence diagram. Each tick can run in parallel
  across symbols; the diagram in §1 shows the **logical** flow per
  tick, not the physical scheduling.
* It is **not** a place to add new behaviour. New behaviour goes
  through a manifest delta + a code PR.
