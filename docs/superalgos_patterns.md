# C-20 Superalgos — Visual Orchestration Architecture Reference

> **Tier:** PATTERN_ONLY (study-only, no code). No Superalgos code is
> used in production. This document extracts three architectural
> patterns from the Superalgos open-source project and maps each to its
> canonical DIX v42.2 counterpart.
>
> **Adapted from:** [Superalgos/Superalgos](https://github.com/Superalgos/Superalgos) (Apache-2.0 licence).

---

## 1. Plugin Node Decomposition Pattern → `registry/plugins.yaml`

### Superalgos pattern

Superalgos represents every strategy component as a **typed node** in
a hierarchical workspace tree. Nodes carry:

* a fixed `type` (e.g. `Trading System`, `Trading Strategy`,
  `Initial Definition`, `Open Stage`, `Close Stage`),
* an `id` (UUID),
* `referenceParent` edges (typed links to data-mine or
  indicator nodes),
* a `config` JSON block with node-specific parameters (e.g.
  `targetRate`, `targetSize`, `formula`).

Nodes are loadable "plugins" — zip archives dropped into a
`Plugins/` directory, each declaring dependencies on other plugins.
The workspace runtime resolves the graph, checks compatibility, and
instantiates nodes in topological order.

**Key properties:**

| Property              | Superalgos              | Relevance to DIX                        |
|-----------------------|-------------------------|-----------------------------------------|
| Node identity         | UUID + type string      | `plugins.yaml` `spec_id` + `name`       |
| Dependency resolution | `referenceParent` edges | `depends_on` list in `plugins.yaml`     |
| Lifecycle state       | Workspace open/close    | `status: DISABLED / SHADOW / ACTIVE`    |
| Config projection     | Inline JSON `config`    | YAML `config` block per slot entry      |
| Plugin discovery      | Filesystem walk         | `PluginRegistry.discover()` scan        |

### DIX mapping

| Superalgos concept               | DIX equivalent                                                          |
|----------------------------------|-------------------------------------------------------------------------|
| Plugin node type                 | `plugins.yaml` entry under its engine namespace                         |
| `referenceParent` edge           | `depends_on` list per entry                                             |
| Plugin workspace                 | `PluginRegistry` (runtime manifest loaded from `registry/plugins.yaml`) |
| Node `config`                    | YAML `config:` block (validated at boot)                                |
| Node lifecycle (load → activate) | `DISABLED → SHADOW → ACTIVE` status FSM                                |

**Takeaway for `registry/`:** DIX's `plugins.yaml` already implements
the core Superalgos plugin node decomposition: typed identity,
declarative dependency graph, per-node config, and a three-state
lifecycle. The Superalgos pattern adds **visual positioning** (x/y
coordinates per node for the canvas), which DIX does not need because
the dashboard2026 widget tree is layout-manager-driven, not
canvas-coordinate-driven.

---

## 2. Visual Workflow Graph Design → `dashboard2026/`

### Superalgos pattern

Superalgos renders the workspace graph as a **force-directed node
canvas**. Each node is a draggable circle or card; edges are drawn as
bezier curves. The user builds strategies by connecting nodes
visually:

```
Trading System ─┬─ Trading Strategy ─┬─ Open Stage ──── Take Position
                │                    └─ Close Stage ─── Take Profit / Stop Loss
                └─ Trading Strategy ─── ...
```

The canvas supports:

* **Zoom / pan** with mouse wheel + drag.
* **Node expand / collapse** — a subtree can be hidden behind its
  parent.
* **Live status overlay** — nodes glow green/yellow/red for
  active/warning/error during a live session.
* **Detach / dock** — nodes can be floated into separate windows.

### DIX mapping

| Superalgos canvas concept          | DIX equivalent in `dashboard2026/`                          |
|------------------------------------|-------------------------------------------------------------|
| Force-directed graph               | `StrategyRegistryFSM.tsx` widget (DAG of strategy states)   |
| Node expand / collapse             | `CommandPalette` (Ctrl-K) collapse/expand sections          |
| Live status overlay (green/amber)  | `WidgetStatusChip` + `LiveStatusPill`                       |
| Detach / dock panels               | `PopoutButton` (window.open per-widget)                     |
| Zoom / pan                         | Layout-profile density toggle (compact / comfortable)       |

**Takeaway for `dashboard2026/`:** Superalgos's canvas-based graph is
an alternative rendering of the same topology DIX expresses through
the `StrategyRegistryFSM` widget (which shows `PROPOSED → SHADOW →
CANARY → LIVE → RETIRED` strategy states as a DAG). DIX chose a
**grid-based widget layout** over a free-form canvas; the tradeoff is
lower visual expressiveness in exchange for **deterministic layout**
(important for operator trust — widgets don't drift). The
Superalgos-style status overlay (node glow) maps cleanly to
`WidgetStatusChip` and `LiveStatusPill`, which already colour-code
live vs stale data per widget.

---

## 3. Strategy DAG Concept → `evolution_engine/`

### Superalgos pattern

Superalgos represents each trading strategy as a **directed acyclic
graph** (DAG) of typed nodes. The DAG has a fixed schema:

```
Trading System
  └── Trading Strategy
        ├── Trigger Stage       (entry conditions)
        ├── Open Stage          (position sizing + order type)
        ├── Manage Stage        (trailing stop / take-profit rules)
        └── Close Stage         (exit conditions)
```

Each stage is itself a sub-DAG (conditions → formulas → situations).
The strategy mutates by **adding / removing / reconnecting nodes** in
the DAG — Superalgos calls this a "strategy design space" traversal.

When a user modifies a strategy, Superalgos serialises the entire
workspace as JSON and replays it from the root node. This gives
**reproducible state** — the same workspace JSON always yields the
same runtime state.

### DIX mapping

| Superalgos DAG concept          | DIX equivalent in `evolution_engine/`                         |
|---------------------------------|---------------------------------------------------------------|
| Strategy node graph             | `StrategyChromosome` (flat parameter vector per strategy)     |
| Stage sub-DAG                   | Strategy decomposition: `StrategyComponent` registry entries  |
| Design space traversal          | `StructuralEvolutionLoop` + `CMA-ES optimizer` + `DEAP arena` |
| Workspace JSON serialisation    | `PatchProposal` (frozen, BLAKE2b-16 digest, deterministic)    |
| Replay from root                | `StateReconstructor` ledger replay (INV-15 determinism)       |

**Takeaway for `evolution_engine/`:** Superalgos treats the strategy
as a mutable graph; DIX collapses that graph into a **parameter
vector** (`StrategyChromosome`) that the evolution engine can mutate
with CMA-ES / DEAP / Nevergrad operators. The mutation result is
always a `PatchProposal` — a frozen, hash-anchored diff that passes
through the governance pipeline (HARDEN-04 freeze check → promotion
gates → approval queue). This is strictly more constrained than
Superalgos's free-form graph editing, which has no governance gate.

DIX's `StateReconstructor` serves the same role as Superalgos's
"replay from workspace JSON" — given the ledger (analogous to the
workspace JSON), the entire system state is reproducible (INV-15).

---

## Summary table

| # | Superalgos pattern               | DIX equivalent                                   | Location                          |
|---|----------------------------------|--------------------------------------------------|-----------------------------------|
| 1 | Plugin node decomposition        | `plugins.yaml` + `PluginRegistry`                | `registry/plugins.yaml`           |
| 2 | Visual workflow graph            | `StrategyRegistryFSM` + layout widgets           | `dashboard2026/`                  |
| 3 | Strategy DAG + design space      | `StrategyChromosome` + `PatchProposal` pipeline  | `evolution_engine/`               |

---

## Authority discipline

* **PATTERN_ONLY (study-only).** No code, no imports, no runtime
  behaviour. This document is purely a design reference.
* **No framework adoption.** Superalgos is `Install: source only` per
  the master directive. No `pip` dependency, no `npm` dependency, no
  code vendored.
* **No Superalgos code in production.** DIX implements the three
  patterns independently; this document anchors the *conceptual debt*
  to Superalgos so contributors can trace design inspiration.

## When to update this doc

* A new Superalgos release changes the node-type schema → update the
  DAG diagram in §3.
* DIX replaces `plugins.yaml` with a programmatic registry → update
  §1 mapping.
* DIX adopts a canvas-based dashboard → update §2 mapping to reflect
  the convergence.
