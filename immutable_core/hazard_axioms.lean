-- DIX VISION v42.2 — hazard_axioms.lean
-- Phase 0 Build Plan §1.1 — NEW immutable axioms governing the
-- SYSTEM_HAZARD event channel. These mirror the non-negotiable design
-- rules stated in the CANONICAL BUILD PLAN §0 and §4.
--
-- This file is a declarative specification; the *runtime* enforcement
-- lives in governance/kernel.py, execution/hazard/*, and
-- enforcement/decorators.py. If a runtime check disagrees with an axiom
-- below, the runtime check is the bug — update the runtime to match.
--
-- Lean4 encoding will follow in a later phase; the text form here is
-- the authoritative specification reviewed by humans.

-- ── A. Single emergency channel ──────────────────────────────────────
axiom H1_single_channel :
  ∀ (signal : SystemProblem),
    signal.path = HazardBus  -- SYSTEM_HAZARD is the ONLY emergency channel.

-- ── B. Producer/consumer authority split ────────────────────────────
axiom H2_dyon_sole_producer :
  ∀ (e : HazardEvent),
    e.source ∈ DyonDomain
    -- Indira and Governance MUST NOT directly emit SYSTEM_HAZARD events.

axiom H3_governance_sole_consumer :
  ∀ (e : HazardEvent),
    Handler(e) = GovernanceKernel._on_hazard
    -- Only Governance interprets hazards into mode changes.

-- ── C. Non-blocking contract ────────────────────────────────────────
axiom H4_non_blocking_producer :
  ∀ (e : HazardEvent),
    ¬ BlocksOn(e, TradingHotPath)
    -- Hazard emission must never block mind.fast_execute.

axiom H5_queue_overflow_fails_closed :
  ∀ (q : HazardQueue),
    q.full → Drop(e) ∧ LogError(e)
    -- A full queue drops the newest event AND emits an error;
    -- it MUST NOT silently overwrite older events.

-- ── D. Severity → response mapping (enforced by severity_classifier) ─
axiom H6_critical_halts_trading :
  ∀ (e : HazardEvent),
    e.severity = CRITICAL → State.trading_allowed = False
    ∧ State.governance_mode = "EMERGENCY_HALT"
    ∧ State.active_hazards := State.active_hazards + 1

axiom H7_high_enters_safe_mode :
  ∀ (e : HazardEvent),
    e.severity = HIGH ∧ ¬ should_halt_trading(e)
      → State.trading_allowed = False
      ∧ State.governance_mode = "SAFE_MODE"
      ∧ State.active_hazards := State.active_hazards + 1

axiom H8_medium_observe :
  ∀ (e : HazardEvent),
    e.severity = MEDIUM → State unchanged ∧ Ledger.append(e)
    -- Medium hazards are recorded but do not change mode.

-- ── E. Ledger durability ────────────────────────────────────────────
axiom H9_every_hazard_logged :
  ∀ (e : HazardEvent),
    ∃ (l : LedgerEvent), l.event_type = "HAZARD" ∧ l.sub_type = e.hazard_type
    -- Every hazard emission MUST produce a ledger record.

-- ── F. Override gate ────────────────────────────────────────────────
axiom H10_override_requires_two_person_gate :
  ∀ (op : HazardOverride),
    op.effect = "clear_active_hazards" ∨ op.effect = "force_normal"
      → op.approvers.length = 2 ∧ op.hardware_keys.length = 2
    -- Clearing active hazards or forcing NORMAL mode requires the
    -- operator-above-all two-person hardware-key gate (manifest §22).

-- ── Summary ─────────────────────────────────────────────────────────
-- These 10 axioms define the hazard-channel contract. PR #1 added the
-- runtime primitives (HazardBus, HazardType enum, severity_classifier,
-- GovernanceKernel._on_hazard). PR #12 (governance round-11/12) brought
-- the SAFE_MODE and EMERGENCY_HALT runtime paths in line with H6/H7.
