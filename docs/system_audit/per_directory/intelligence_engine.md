# intelligence_engine/ — signal generation + meta-controller (57 files)

## Purpose

Stateful engine that consumes `MarketTick` + `NewsItem` + plugin
outputs and emits `SignalEvent`. Never executes — `B25/B26/B27` lint
forbids it from importing `execution_engine.adapters` or constructing
`ExecutionIntent` directly. This is the INV-56 Triad Lock decider.

Sub-packages:

* `engine.py` — pipeline composer.
* `meta_controller/` — `runtime_adapter.py`, `hot_path.py`,
  `orchestrator.py`. INV-53 calibration snapshots, J3 audit ledger.
* `strategy_runtime/` — regime detector + strategy state machine +
  conflict resolver + scheduler.
* `strategy_library/` — decomposition + components + registry.
* `news/news_projection.py` — `NewsItem -> SignalEvent` projection.
* `cognitive/` — chat graph, approval edge, registry-driven chat
  model. Consumed by `/api/cognitive/chat/*`.
* `learning/slow_loop.py` — closed-loop learning bridge (PR #140).
* `knowledge/news_index.py` — FAISS-style news knowledge index.
* `mcp/opennews.py` — MCP adapter for opennews (PR #156).
* `runtime_context.py` — `RuntimeContextBuilder` (PR #141).

## Wiring

* `ui/server.py` builds the engine at startup, registers the active
  plugin chain, and feeds `MarketTick`/`SignalEvent` per HTTP.
* `IntelligenceFeedbackSink` (PR #140) receives `TradeOutcome` from
  `execution_engine.protections.feedback.FeedbackCollector`.
* The chat graph uses `AuditLedgerCheckpointSaver` (PR #83) so
  LangGraph state lives in the audit ring, not on disk — keeps
  determinism.

## Static-analysis result

* 57 files, 23 with findings — 22 are ruff-format drift, 1 is a
  vulture false positive (`run_manager` parameter required by the
  LangChain `BaseChatModel._generate` interface; cannot be removed
  without breaking the contract).
* No orphan modules. Every plugin under `cognitive/chat/` is imported
  via `intelligence_engine.cognitive.chat.__init__`.

## Deep-read observations

* `engine.py` (236 lines) is a pure composer — it instantiates
  plugins, drains them per-tick, and emits.
* `meta_controller/hot_path.py` — single mutator for the
  meta-controller's per-tick state; respects INV-15 (no clock reads,
  no PRNG).
* `cognitive/approval_edge.py` — gates SignalEvent emission behind
  operator approval (PR #87). The right side of the Triad Lock —
  cognitive plugins propose; the operator approves; only then does
  the engine emit.
* `runtime_context.py` — populated on the hot path (PR #141) so
  every per-tick decision sees a coherent runtime snapshot.

## Risks / gaps

* None blocking.
* `learning/slow_loop.py` is the only place that touches
  `LearningEvolutionFreezePolicy`; if a future learning lane lands
  outside it, B81 lint must extend to cover it. (Currently only B81
  scope is `update_emitter.py` + `slow_loop.py`.)

## Verdict

**HEALTHY.** Triad Lock honoured, B25/B26/B27 lint clean. The
vulture finding is a false positive; document and ignore.
