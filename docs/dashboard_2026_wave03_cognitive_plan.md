# Dashboard-2026 — Wave-03 Cognitive Plan (LangGraph)

**Status:** PLANNING ONLY (no LangGraph code yet). Locked in as the
forward integration contract so the wave-01 / wave-02 work doesn't
make decisions that contradict it.

**Author:** Devin · session continuing from PR #68 (FeedRunner race
fix, merged).
**Scope:** Where LangGraph fits in DIX VISION, where it must not, and
the safety rails that have to be in place before the first import.

---

## 1. Position summary

LangGraph (LangChain's graph-orchestration framework) is a **good fit
for two specific subsystems** of DIX VISION and **dangerous if used
anywhere else**. The plan below applies it selectively, behind hard
isolation enforced by an authority-lint rule (`B24`) and a new
invariant (`INV-67`).

This document does NOT introduce any LangGraph dependency yet. The
wave-01 (current) and wave-02 (React port) work explicitly does not
require LangGraph and must remain runnable without it. Wave-03 adds
it inside the cognitive subsystem only.

## 2. Where LangGraph fits

### 2.1 `intelligence_engine.cognitive.chat.*` — Indira / Dyon chat backends

LangGraph orchestrates the multi-turn conversation graphs:

* turn loops with retries and tool-use,
* human-in-the-loop pauses (operator approval before any
  `SignalEvent` hits the bus),
* persistent context per chat session (BeliefState +
  PressureVector + portfolio snapshots),
* dynamic supervisor → specialist handoffs (e.g. regime detector,
  microstructure analyst, sentiment, liquidity forecaster) within a
  single turn.

The chat backends produce **proposals** (`SignalEvent` with
`autonomy_required=True` for high-risk turns), never
`ExecutionEvent` directly. Governance promotes / rejects each
proposal as it would any plugin-emitted signal.

### 2.2 `evolution_engine.dyon.patch_pipeline.*` — Dyon's self-coding

LangGraph orchestrates the long-running graph:

* `PatchProposal → Sandbox → Shadow → Canary → Live`,
* multi-agent code review during `Sandbox` (separate review +
  regression-search + safety-check agents collaborating),
* re-entry on failure with state preserved (LangGraph checkpoints).

The pipeline still flows through the `governance_engine` for every
state transition; LangGraph does not own the FSM, it drives the
work *between* state transitions.

## 3. Where LangGraph must not go

| Subsystem | Reason |
|---|---|
| `execution_engine.*` | Hot path. INV-15 replay determinism + sub-millisecond budgets. LangGraph adds non-determinism (LLM outputs drift), GIL contention, and a 30+ transitive-dep footprint. |
| `governance_engine.*` | Order-blind by INV-56. Authority decisions are O(1) table lookups (PR #59), not graph walks. LangGraph here would invite a "smart governance" anti-pattern that's untestable. |
| `system_engine.*` | Hazards must be deterministic and pure. |
| `core.contracts.*` / `core.coherence.*` | Pure projections. No I/O, no LLM. |
| The canonical event bus | Stays as the integration boundary. LangGraph publishes *into* the bus, never replaces it. |

## 4. Five non-negotiables

### 4.1 Quarantine determinism (INV-67)

LangGraph runs are inherently non-deterministic — LLM outputs drift.
A new invariant states:

> **INV-67 — Cognitive subsystems are advisory; deterministic replay
> verifies the typed bus, not the cognitive graphs.** INV-15
> (replay determinism) applies to the hot path and the canonical
> bus. LangGraph subsystems are quarantined as
> `nondeterministic_advisory_only`. Their outputs become typed bus
> events (proposals) which Governance gates; the bus is replayable,
> the graphs are not.

### 4.2 Authority-lint rule B24 — import containment

Only `intelligence_engine.cognitive.*` and `evolution_engine.dyon.*`
may import `langgraph` / `langchain*` / `langsmith`. Any other
importer is a hard CI failure. Rule shipped in branch c (this PR)
even though no module imports LangGraph yet — defensive, prevents
wave-02 / future work from drifting.

### 4.3 Registry-driven LLM adapter (B23 still applies)

LangGraph's `ChatOpenAI`, `ChatAnthropic`, etc. are **forbidden** in
chat widget code. Wave-03 ships a thin
`RegistryDrivenChatModel(BaseChatModel)` that:

* takes a `TaskClass` at construction,
* resolves an ordered provider list via
  `core.cognitive_router.select_providers()`,
* dispatches each turn to the first eligible provider with
  fallback on transient failure (`SOURCE_FALLBACK_ACTIVATED`
  audit, SCVS-10).

The graph nodes call this adapter. They never name a vendor. Adding
a new AI is still a registry-only change.

### 4.4 Audit-ledger checkpoints (not LangGraph's default SQLite)

`langgraph` ships a default `SqliteSaver` for checkpoints. We
replace it with a custom `BaseCheckpointSaver` that writes to the
canonical audit ledger (`state.ledger`). Otherwise:

* two sources of truth → "time-travel debugging" claim is no
  longer auditable,
* SCVS-style retention policies (PII, GDPR-style, operator-defined
  retention windows) don't apply to LangGraph's SQLite store,
* an external party with disk access can read graph state without
  going through Governance.

### 4.5 LangSmith off by default and self-hosted if enabled

LangSmith is a SaaS — sending DecisionTrace / patch proposals /
chat transcripts to a third-party observability service is a
governance violation by default. Either:

* Self-host LangSmith on infra you control, OR
* Use OpenTelemetry → existing audit ledger (preferred — single
  source of truth).

The shipped configuration sets `LANGCHAIN_TRACING_V2=false` and
the `langsmith` package stays in `requirements-dev.txt` only (not
`requirements.txt`).

## 5. Sequencing

| Wave | Scope | LangGraph? |
|---|---|---|
| **wave-01** (current branch c) | vanilla skeleton + `/api/ai/providers` + cognitive router + B23 + B24 + this plan doc + INV-67 entry | **No** |
| **wave-02** | React port (Vite + shadcn/ui + TanStack Query + Lightweight Charts v5), Pydantic→TS codegen, chat-turn streaming endpoint | **No** — chat turns are direct provider calls via the cognitive router so the registry-driven contract is exercised end-to-end before LangGraph layers on top |
| **wave-03 (cognitive)** | LangGraph lands inside `intelligence_engine.cognitive.chat` for Indira / Dyon, behind a feature flag. Custom audit-ledger checkpoint saver. `RegistryDrivenChatModel`. B24 enforced. INV-67 active. LangSmith off / self-hosted. | **Yes**, this scope only |
| **wave-04** | LangGraph extends into `evolution_engine.dyon.patch_pipeline` once chat is stable. Multi-agent code review graph. | **Yes**, this scope only |

Doing it in this order lets the cognitive router prove out before
the graph layer goes on top. Starting with LangGraph would let the
graph implicitly define provider routing and the registry-driven
invariant rots.

## 6. Dependency footprint

When wave-03 lands, `requirements.txt` adds:

```
langgraph==<exact pinned>
langchain-core==<exact pinned>
```

`requirements-dev.txt` (only when self-hosted observability is
wired in wave-03+) adds:

```
langsmith==<exact pinned>
```

Pin tightly. LangChain's surface churns hard between minor versions;
an unpinned upgrade has bricked production agents before. Use
`==` everywhere, not `>=`.

This is a +30..40 transitive-dep increase from today's 8-direct
baseline. Audit cost is real but manageable if quarantined.

## 7. Acceptance criteria for wave-03 PR

* B24 still passes (no leakage outside cognitive / dyon scope).
* B23 still passes (no vendor names in chat widget JS / Python).
* All existing tests still pass; INV-15 replay tests are unchanged
  (LangGraph runs do not participate).
* New tests:
  * `RegistryDrivenChatModel` resolves providers from the
    registry, not from class names.
  * Custom checkpoint saver writes to the audit ledger and reads
    back identical state.
  * LangSmith is OFF in the default config.
  * Adding a new provider row to the registry surfaces in chat
    without any chat-widget code change.

## 8. Out of scope for this document

* Choice of specific LLM providers — that stays a registry decision.
* Memecoin domain (W1) — LangGraph is only inside
  `intelligence_engine.cognitive.*` / `evolution_engine.dyon.*` and
  must not cross into the memecoin trio's isolated pipeline.
* Live mode promotion — orthogonal; gated by Governance as today.
