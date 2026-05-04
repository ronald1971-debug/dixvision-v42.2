"""Sensory perimeter (NEUR-01..03, WEBLEARN-01..10).

The :mod:`sensory` package is the system's perception layer. It is
*outside* the four canonical engines (intelligence / execution /
governance / system) and feeds them via SignalEvent / NewsItem /
TraderArchetype inputs only — never by directly mutating engine state.

Sub-packages (per :file:`docs/directory_tree.md`):

  * :mod:`sensory.neuromorphic` — NEUR-01..03 signal processors;
  * :mod:`sensory.web_autolearn` — WEBLEARN-01..10 web ingestion +
    HITL gate.

Authority discipline: nothing under :mod:`sensory` is allowed to
import an engine, write to the audit ledger, or mutate the
``SystemMode`` FSM. The only legal output is a typed value (NewsItem,
SocialPost, OnChainMetric, TraderArchetype) consumed by the
intelligence engine.
"""
