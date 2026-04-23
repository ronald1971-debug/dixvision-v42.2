-- DIX VISION v42.2 — safety_axioms.lean
-- Phase 0 Build Plan §1.1 — Immutable safety floors. These axioms are
-- encoded as Python invariants in immutable_core/constants.py
-- (SafetyAxioms dataclass, @frozen) and enforced in
-- governance/constraint_compiler.py + enforcement/decorators.py.
-- A Lean4 proof encoding will follow in a later phase; this text form
-- is the authoritative specification that the runtime must obey.

-- ── Financial floors (manifest §1) ─────────────────────────────────
axiom S1_max_drawdown_floor :
  ∀ (r : RiskCache),
    r.circuit_breaker_drawdown ≤ 0.04
    -- 4% portfolio drawdown is the *maximum* the operator can configure;
    -- tighter is fine, looser is impossible.

axiom S2_per_trade_loss_floor :
  ∀ (r : RiskCache),
    r.circuit_breaker_loss_pct ≤ 0.01
    -- 1% per-trade loss cap is the maximum; operator can only lower it.

-- ── Fail-closed default (manifest §3) ──────────────────────────────
axiom S3_fail_closed :
  ∀ (decision : Decision),
    decision.status = Unknown → decision.action = Reject
    -- When Governance cannot reach a clear yes, the answer is no.

-- ── Credentials locality (manifest §22 + Phase 8) ──────────────────
axiom S4_credentials_local_only :
  ∀ (c : Credential),
    c.store = LocalKeystore ∧ c.plaintext_outside_ram = False
    -- Credentials never leave the operator's machine in plaintext.

-- ── Fast-path latency budget (manifest §5, §15) ────────────────────
axiom S5_fast_path_latency_budget :
  ∀ (t : Trade),
    t.fast_path_duration ≤ 5.0  -- milliseconds
    -- fast_execute_trade must complete under 5ms or escalate to
    -- enforce_full + LatencyGuard emits a hazard.

-- ── Frozen hot-path functions (manifest §22) ───────────────────────
axiom S6_frozen_hot_path :
  ∀ (m : Module),
    m ∈ {mind.fast_execute, system.fast_risk_cache, interrupt.*} →
      Changes(m) must pass SandboxPipeline ∧ TwoPersonGate
    -- Hot-path code only changes through the two-person gate.

-- ── Ledger immutability (manifest §7) ──────────────────────────────
axiom S7_ledger_append_only :
  ∀ (l : LedgerEvent),
    l.sha256.prev = SHA256(l.index - 1).full_row
    -- The ledger is a hash-chain; any rewrite fails
    -- state.ledger.hash_chain.verify_full_chain().

-- ── Foundation integrity (Phase 0 §1.1) ────────────────────────────
axiom S8_foundation_hash_pinned :
  ∀ (boot : BootSequence),
    boot.step_1_foundation_check =
      SHA256(immutable_core/foundation.py) = immutable_core/foundation.hash
    -- If prod env and hashes disagree, boot triggers kill_switch.

-- ── Kill-switch path (manifest §15) ────────────────────────────────
axiom S9_kill_switch_uses_only_stdlib :
  ∀ (call : kill_switch.trigger),
    call.transitive_imports ⊆ PythonStdlib
    -- The kill switch must not depend on any code that might itself be
    -- broken when it's needed.

axiom S10_kill_switch_is_idempotent :
  ∀ (t₁ t₂ : Instant),
    kill_switch._killed(t₁) ∧ t₂ > t₁ → kill_switch.trigger(t₂) = NoOp
    -- Multiple invocations are safe; only the first has effect.

-- ── Manifest alignment ─────────────────────────────────────────────
-- S1..S10 are the ten non-negotiable safety axioms. Any PR that relaxes
-- any axiom above must be treated as a manifest amendment and therefore
-- routed through the two-person hardware-key gate (manifest §22).
