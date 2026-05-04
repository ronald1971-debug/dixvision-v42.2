import { apiPost } from "./base";

/**
 * Execution intent submission — the ONLY write path DIX MEME uses for
 * manual orders, sniper hits, and copy mirrors.
 *
 * Every intent goes through `/api/dashboard/action/intent` →
 * Governance → ExecutionEngine. The autonomy mode the operator picks
 * on TradePage maps to the `risk_mode` field below: manual vs
 * semi-auto vs full-auto controls the operator-approval requirement
 * Governance applies, NOT a separate execution path.
 */

export type GovernanceDecisionEnvelope = {
  approved: boolean;
  summary: string;
  decision: Record<string, unknown>;
};

export type IntentRequest = {
  /** "trade" / "snipe" / "copy" / "exit" / "rebalance" — short label. */
  objective: string;
  /** "manual" / "semi-auto" / "full-auto" — autonomy band. */
  risk_mode: string;
  /** "intra-second" / "intra-minute" / "intra-hour" / "intra-day". */
  horizon: string;
  /** Free-text focus tokens (pair, chain, wallet, strategy). */
  focus?: ReadonlyArray<string>;
  reason?: string;
  requestor?: string;
};

export const submitIntent = (body: IntentRequest) =>
  apiPost<GovernanceDecisionEnvelope>("/api/dashboard/action/intent", body);

export type ModeRequest = {
  target_mode: string;
  reason?: string;
  operator_authorized?: boolean;
  requestor?: string;
  consent_operator_id?: string;
  consent_policy_hash?: string;
  consent_nonce?: string;
  consent_ts_ns?: string;
};

export const submitMode = (body: ModeRequest) =>
  apiPost<GovernanceDecisionEnvelope>("/api/dashboard/action/mode", body);

export type KillRequest = {
  reason?: string;
  requestor?: string;
};

export const submitKill = (body: KillRequest) =>
  apiPost<GovernanceDecisionEnvelope>("/api/dashboard/action/kill", body);
