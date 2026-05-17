# C-19 AutoHedge — Role Mapping

> **Tier:** PATTERN_ONLY (advisory). AutoHedge itself is **never run in
> production**. This document anchors the read-only pattern catalog at
> `intelligence_engine/agents/autohedge_patterns.py`.
>
> **Adapted from:** [The-Swarm-Corporation/AutoHedge](https://github.com/The-Swarm-Corporation/AutoHedge) (MIT licence).

## Why this exists

AutoHedge ships a five-role portfolio-decomposition pattern:

1. **market_analyst** — read macro / regime context
2. **technical_analyst** — score price / order-flow signals
3. **risk_manager** — enforce per-position and account-level risk gates
4. **portfolio_optimizer** — size and allocate across the candidate set
5. **execution_manager** — dispatch the chosen orders

DIX already does all five things, but in **separate engines**, behind
already-frozen contracts. C-19 extracts the *pattern* — not the
framework — and pins which DIX module fulfils each role. This lets
future PRs reason about coverage gaps without restating the
architecture every time.

## Role → DIX module table

| AutoHedge role          | DIX module                                          | How DIX fulfils it                                                                                       |
|-------------------------|-----------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `MARKET_ANALYST`        | `intelligence_engine/macro/regime_engine.py`        | `MacroRegimeEngine` classifies a `MacroSnapshot` into a regime label + confidence.                       |
| `TECHNICAL_ANALYST`     | `intelligence_engine/plugins/`                      | The IND-L01..L12 plugin family produces `SignalEvent`s (momentum / VPIN / footprint / sentiment / …).    |
| `RISK_MANAGER`          | `governance_engine/control_plane/risk_evaluator.py` | `RiskEvaluator` + `RiskSnapshot.halted` + the GOV-CP-07 hazard throttle adapter gate every intent.       |
| `PORTFOLIO_OPTIMIZER`   | `intelligence_engine/portfolio/`                    | `PortfolioAllocator` + `ExposureManager` compute allocations and exposure caps over the candidate set.   |
| `EXECUTION_MANAGER`     | `execution_engine/engine.py`                        | `ExecutionEngine.execute(intent)` is the single chokepoint for all order dispatch (HARDEN-02).           |

The canonical pattern catalog is enforced at import time:
`AUTOHEDGE_PATTERN_CATALOG` covers every `AutoHedgeRole` exactly once,
all `dix_module` anchors are unique, and the
`canonical_consensus_flow()` sequence visits every role exactly once.
Reverse lookup is exact-match: `autohedge_role_for_dix_module(path)`
returns the role tag (or `None`).

## Consensus flow

AutoHedge runs the roles in a fixed, sequential order. DIX mirrors the
same flow across separate engines:

```
market_analyst       →  intelligence_engine/macro/regime_engine.py
technical_analyst    →  intelligence_engine/plugins/                     (SignalEvent producers)
risk_manager         →  governance_engine/control_plane/risk_evaluator.py
portfolio_optimizer  →  intelligence_engine/portfolio/
execution_manager    →  execution_engine/engine.py                       (HARDEN-02 chokepoint)
```

Each arrow is a **typed-event boundary** in DIX. The intelligence tier
emits `SignalEvent`s; the governance tier turns them into
`GovernanceDecision`s; the execution tier converts decisions into
`ExecutionIntent`s and dispatches them. The AutoHedge consensus shape
falls out naturally — DIX simply spreads the five roles across
engines, where AutoHedge keeps them in one process.

## Authority discipline

* **OFFLINE_ONLY (advisory).** No `SignalEvent` /
  `ExecutionIntent` / `GovernanceDecision` / `PatchProposal` is ever
  constructed in this module (B27 / B28 / INV-71).
* **B1 engine isolation.** The pattern catalog references DIX modules
  **only as string anchors**; it does **not import**
  `execution_engine.*`, `governance_engine.*`, `system_engine.*`, or
  `evolution_engine.*`. This is pinned by AST guard tests.
* **INV-15 determinism.** No `random` / `time` / `datetime` /
  `secrets` / `os` / `asyncio` imports.
* **No framework adoption.** The upstream AutoHedge package is not a
  dependency. `NEW_PIP_DEPENDENCIES = ()`. The five-role committee
  *runtime* lives at `intelligence_engine/agents/trading_agents_bridge.py`
  (C-18); C-19 is read-only design reference.

## When to update this doc

* A new DIX module supersedes one of the role anchors → update both
  the catalog entry and the table above.
* A new AutoHedge release adds a sixth role → extend `AutoHedgeRole`,
  add a `AutoHedgePatternRole` entry, extend the table here, and the
  AST tests will surface any coverage gap.
* The role anchors are renamed / moved → the import-time invariant
  check `_verify_catalog_invariants()` will keep the catalog
  internally consistent, but you must update the table above by hand.
