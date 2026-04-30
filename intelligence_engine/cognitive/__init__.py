"""Cognitive subsystem (Dashboard-2026 wave-03).

Per the wave-03 cognitive plan
(``docs/dashboard_2026_wave03_cognitive_plan.md``) and INV-67, the
``intelligence_engine.cognitive.*`` package is one of two scopes
(alongside ``evolution_engine.dyon.*``) permitted by authority-lint
rule **B24** to import ``langgraph`` / ``langchain*`` / ``langsmith``.

Outputs of this subsystem are *advisory only* — the deterministic
canonical bus and INV-15 replay tests do not exercise these graphs.
The cognitive surface produces typed bus events (proposals); only
``governance_engine`` may promote a proposal to an action.
"""
