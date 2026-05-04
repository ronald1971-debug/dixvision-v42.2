# sensory/, learning_engine/, evolution_engine/ (combined)

## sensory/ — 23 files

### Purpose

Read-only inputs. Every external feed lands here, is shaped into a
`SignalEvent` / `NewsItem` / `MacroSnapshot` / `OnChainSnapshot`, and
is passed through SCVS validation before reaching `intelligence_engine`.

Sub-packages:

* `web_autolearn/` (PRs #175, #177) — WEBLEARN-01..04, 10. Crawler,
  AI filter, curator, pending buffer, seeds.yaml.
* `onchain/contracts.py`, `regulatory/contracts.py`, `dev/contracts.py`,
  `alt/contracts.py`, `cognitive/contracts.py` — resolves the
  forward-declared schema paths in `data_source_registry.yaml`.
* `neuromorphic/{indira_signal,dyon_anomaly,governance_risk}.py`
  (PR #179) — NEUR-01..03 pure feature extractors.
* `coindesk_rss/` (PR #102), `fred/` (PR #108), `bls/` (PR #121),
  `binance_ws/` (PR #67), `pumpfun/` (PR #153), `raydium/` (PR #153),
  `tradingview/` (PR #96), `news/` (CoinDesk + opennews-mcp + fanout).

### Static-analysis result

* 23 files, 14 with findings — **all 14 are ruff-format drift only**.
* No orphan modules. No semantic findings.

### Observations

* The B26 lint forbids `sensory/*` from constructing
  `ExecutionIntent` or importing `execution_engine.adapters`.
  Verified clean.
* `sensory/web_autolearn/` is a perimeter — currently emits to a
  pending buffer, not directly into `intelligence`. The TI Playwright
  pipeline (Sensory-S1.D) will be the first feedback path.

### Verdict

**HEALTHY.** No bugs. The Sensory-S1.D TI pipeline is the next
build-out.

---

## learning_engine/ — 9 files

### Purpose

Closed-loop learner. Consumes `TradeOutcome`, computes adjustments,
emits `LearningUpdate` proposals (never direct mutations).

Sub-packages:

* `engine.py` (42 lines) — composer.
* `lanes/weight_adjuster.py` — INV-63 closed loop.
* `update_emitter.py` — emits `LearningUpdate`; gated by
  `LearningEvolutionFreezePolicy` (PR #81).
* `regret/` — counterfactual / missed-opportunity tracking
  (Phase 10.13 stub).

### Wiring

* `ExecutionEngine` → `FeedbackCollector` → `LearningEngine.process`
  (PR #140). Hazard-throttled REJECTs also feed it (PR #143).
* `LearningEngine.update_emitter.emit(...)` → `governance_engine`
  `UpdateValidator` → `UpdateApplier` (PR #114). The governance side
  is the only authority that mutates registries.

### Static-analysis result

* 9 files, 6 with findings — **all 6 are ruff-format drift only**.
* No orphan modules. No semantic findings.

### Verdict

**HEALTHY.** Loop is closed and gated. The full `state/memory_tensor/`
and Phase-10 evolution interaction layer remain queued.

---

## evolution_engine/ — 14 files

### Purpose

Long-loop learner. Strategy genetics (mutation/crossover), evolution
of trader archetypes, evolution-side strategy pool. Does not mutate
governance state directly — emits proposals.

Sub-packages:

* `engine.py` (41 lines) — composer.
* `genetic/` — Phase 10.12 mutation/crossover/inheritance (skeleton).
* `strategy_pool.py`, `pool/` — candidate strategy evolution.

### Static-analysis result

* 14 files, 10 with findings — **all 10 are ruff-format drift only**.
* No orphan modules. No semantic findings.

### Observations

* Currently consumes `LearningUpdate` proposals from the closed
  learning loop and uses them as fitness signals.
* `genetic/` package is canonical-tree skeleton; no genetic operators
  are yet active in production.

### Verdict

**HEALTHY but partial.** Loop is wired, full Phase 10.12 is queued.
