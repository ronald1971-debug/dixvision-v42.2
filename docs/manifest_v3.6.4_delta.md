# Manifest delta — v3.6.4 (Dashboard-2026 wave-01 / cognitive prep)

This delta lands the **Dashboard-2026 wave-01** scaffolding (registry-
driven AI provider router, Indira / Dyon chat widget skeletons,
per-form widget grid with memecoin trio isolated per W1) and the
**forward safety rails** for the wave-03 LangGraph integration.

No LangGraph dependency is added — wave-01 explicitly does not need
it. The new lint rule (B24) and invariant (INV-67) fire defensively
so that wave-02 / wave-03 cannot drift past the boundary unnoticed.

The full wave-03 plan (where LangGraph fits, where it must not, the
custom audit-ledger checkpoint saver, the registry-driven
``RegistryDrivenChatModel``, the LangSmith default-off policy, the
sequencing) lives in
[docs/dashboard_2026_wave03_cognitive_plan.md](dashboard_2026_wave03_cognitive_plan.md).

## New invariants

### INV-67 — Cognitive subsystems are advisory; deterministic replay verifies the typed bus, not the cognitive graphs

INV-15 (replay determinism) applies to the hot path and the
canonical event bus. Cognitive subsystems
(``intelligence_engine.cognitive.*`` and the future
``evolution_engine.dyon`` LangGraph workflows) are quarantined as
``nondeterministic_advisory_only``:

* Their outputs become typed bus events (proposals — ``SignalEvent``
  with ``autonomy_required=True`` for high-risk turns,
  ``PatchProposal`` for self-coding work) which Governance gates.
* The bus is replayable; the graphs are not.
* INV-15 invariants check the typed bus only. They do NOT
  re-execute LangGraph runs or LLM calls.

This preserves INV-15's meaningfulness (replay must be byte-
identical) while allowing non-deterministic ML orchestration in a
clearly bounded subsystem.

## New lint rules

### B23 — Registry-driven AI providers (Dashboard-2026 wave-01)

Chat widget files (``ui/static/chat_widget.js``,
``ui/static/indira_chat.html``, ``ui/static/dyon_chat.html``, and
any future ``intelligence_engine.cognitive.chat.*`` /
``ui.cognitive.chat.*`` Python module) may not contain any string
literal naming a specific AI vendor.

Single source of truth: ``registry/data_source_registry.yaml`` (rows
with ``category: ai``). Chat widgets read this registry via
``GET /api/ai/providers`` and surface whatever it returns. Adding a
new provider is a registry-only change — no widget edit.

### B24 — LangGraph / LangChain import containment (INV-67)

Only ``intelligence_engine.cognitive.*`` and
``evolution_engine.dyon.*`` may import ``langgraph``,
``langchain*``, or ``langsmith``. Hot-path engines
(``execution_engine``, ``governance_engine``, ``system_engine``)
and the deterministic core (``core``) must never import any of
these surfaces — graph orchestration is non-deterministic and is
quarantined as advisory-only.

Rule fires defensively even before any module imports LangGraph
(currently none) so future work cannot drift past the boundary
unnoticed.

## Registry schema extension

``registry/data_source_registry.yaml`` rows whose
``category: ai`` may now declare a ``capabilities`` tuple. Allowed
values:

```
reasoning, code_gen, multimodal, realtime_search,
long_context, tool_use, agent_orchestration
```

The cognitive router (`core.cognitive_router`) selects providers per
``TaskClass`` by capability superset match. Five task classes ship:

| TaskClass                          | Required capabilities                |
|------------------------------------|--------------------------------------|
| ``INDIRA_REASONING``               | ``reasoning``                        |
| ``DYON_CODING``                    | ``code_gen``, ``long_context``       |
| ``INDIRA_MULTIMODAL_RESEARCH``     | ``reasoning``, ``multimodal``        |
| ``DYON_AGENT_ORCHESTRATION``       | ``agent_orchestration``, ``tool_use``|
| ``INDIRA_REALTIME_RESEARCH``       | ``realtime_search``                  |

The router is **pure**: given a registry snapshot and a TaskClass it
returns an ordered tuple of eligible providers with no I/O, no
clock, no PRNG. Two calls with identical inputs produce byte-
identical outputs (consistent with INV-15 on the typed-bus side and
with the public projection contract on the chat widget side).

## New module surface

```
core/cognitive_router/
  __init__.py
  task_class.py                # TaskClass enum + required_capabilities()
  router.py                    # AIProvider + enabled_ai_providers()
                               #   + select_providers()

ui/static/
  dashboard2026.css            # vanilla skeleton stylesheet
  chat_widget.js               # registry-driven chat widget runtime
  indira_chat.html             # Indira chat skeleton
  dyon_chat.html               # Dyon chat skeleton
  forms_grid.html              # per-form widget grid (memecoin isolated)

docs/
  dashboard_2026_wave03_cognitive_plan.md   # NEW (wave-03 plan)
```

New HTTP routes on ``ui.server``:

| Method | Path                  | Purpose                                |
|--------|-----------------------|----------------------------------------|
| GET    | ``/api/ai/providers`` | Registry-driven AI provider list (with optional ``?task=`` filter via the cognitive router). |
| GET    | ``/indira-chat``      | Indira chat skeleton page.             |
| GET    | ``/dyon-chat``        | Dyon chat skeleton page.               |
| GET    | ``/forms-grid``       | Per-form widget grid (memecoin isolated). |

No new dependencies. ``langgraph`` / ``langchain`` are explicitly
NOT added in this delta — they belong to wave-03.
