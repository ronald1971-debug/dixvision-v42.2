-- DIX VISION v42.2 — neuromorphic_axioms.lean
-- Phase 0 Build Plan §1.1 extension — axioms that bound the
-- neuromorphic triad (mind.plugins.neuromorphic_signal,
-- execution.monitoring.neuromorphic_detector,
-- governance.signals.neuromorphic_risk). This file is the
-- authoritative specification; any runtime behaviour that violates
-- an axiom below is a bug in the runtime, not the spec.
--
-- Operator rule (locked verbatim):
-- "Neuromorphic components may observe, detect, and advise. They may
--  never decide, execute, or modify system state. Their outputs are
--  events. Their models are immutable at runtime. Their existence is
--  audited."

-- ── N1: observation-only authority ───────────────────────────────────
axiom N1_no_decision_authority :
  ∀ (c : NeuromorphicComponent) (a : Action),
    Issues(c, a) → a ∈ {EventEmission, HeartbeatPing, HealthReport}
    -- A neuromorphic component MUST NOT call any function annotated
    -- with @enforce_governance / @enforce_full / @enforce_domain.

-- ── N2: event-only outputs ───────────────────────────────────────────
axiom N2_event_only_outputs :
  ∀ (c : NeuromorphicComponent) (o : Output),
    Produces(c, o) → o.kind = "EVENT"
    -- Outputs are SPIKE_SIGNAL_EVENT, SYSTEM_ANOMALY_EVENT, or
    -- RISK_SIGNAL_EVENT. Nothing else.

-- ── N3: model immutability at runtime ────────────────────────────────
axiom N3_model_immutable_at_runtime :
  ∀ (c : NeuromorphicComponent),
    Editable(c.model, ProcessUptime) = False
    -- Model weights and topology are frozen when the process starts.
    -- Adaptation is offline only; online changes require sandbox gate.

-- ── N4: ledger audit ─────────────────────────────────────────────────
axiom N4_every_output_ledger_audited :
  ∀ (c : NeuromorphicComponent) (o : Event),
    Produces(c, o) → ∃ (l : LedgerEvent),
      l.event_type = "NEUROMORPHIC" ∧ l.sub_type = o.type
    -- Every event a neuromorphic component emits writes a ledger row.

-- ── N5: dead-man for detectors ───────────────────────────────────────
axiom N5_detector_self_heartbeat :
  ∀ (c : NeuromorphicComponent),
    (NowMonotonic - c.last_heartbeat) < 3 * c.heartbeat_interval
      ∨ KillSwitch.triggered
    -- If a detector goes silent beyond 3× its heartbeat interval, the
    -- system dead-man trips — fail-closed, no silent failure.

-- ── N6: authority-lint forbidden primitives ──────────────────────────
axiom N6_forbidden_calls :
  ∀ (c : NeuromorphicComponent) (f : Symbol),
    f ∈ {
      "governance.kernel.evaluate",
      "mind.fast_execute.fast_execute_trade",
      "execution.engine.execute",
      "security.operator.*",
      "system.fast_risk_cache.update",
      "system.fast_risk_cache.halt_trading",
      "system.fast_risk_cache.enter_safe_mode",
      "core.registry.Registry.register"
    } → ¬ Calls(c, f)
    -- authority_lint rule C2 enforces this statically. A failing
    -- lint blocks the sandbox pipeline — no runtime override.

-- ── N7: governance consumes neuromorphic advice as advisory-only ─────
axiom N7_advisory_only :
  ∀ (s : RiskSignalEvent) (d : GovernanceDecision),
    Influences(s, d) → d.determinism_tier = "HARD_RULE"
    -- Governance may read a RISK_SIGNAL_EVENT as a *feature input*,
    -- but the final decision must still come from a deterministic,
    -- replayable hard rule (threshold, constraint compiler, policy).

-- ── N8: STDP is offline only ─────────────────────────────────────────
axiom N8_no_runtime_rewiring :
  ∀ (c : NeuromorphicComponent) (e : Edge),
    Modify(c.topology, e, runtime=True) = False
    -- No online STDP in production. Offline training produces new
    -- weights; new weights require sandbox + two-person gate to load.

-- ── Summary ──────────────────────────────────────────────────────────
-- N1..N8 preserve the core authority split (Indira trades, Dyon
-- maintains, Governance decides, Operator-above-all) while adding a
-- neuromorphic *sensory* layer in front of each. Any future PR that
-- weakens any axiom must pass the two-person hardware-key gate.
