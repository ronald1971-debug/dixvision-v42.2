import type {
  OperatorActionResponse,
  OperatorModeRequest,
  OperatorSummaryResponse,
  OperatorUnlockRequest,
} from "@/types/generated/api";

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export async function fetchOperatorSummary(
  signal?: AbortSignal,
): Promise<OperatorSummaryResponse> {
  const res = await fetch(`${BASE}/api/operator/summary`, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(
      `GET /api/operator/summary failed: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as OperatorSummaryResponse;
}

export interface KillRequestBody {
  reason: string;
  requestor?: string;
}

export async function postOperatorKill(
  body: KillRequestBody,
): Promise<OperatorActionResponse> {
  const res = await fetch(`${BASE}/api/operator/action/kill`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      reason: body.reason,
      requestor: body.requestor ?? "operator",
    }),
  });
  if (!res.ok) {
    throw new Error(
      `POST /api/operator/action/kill failed: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as OperatorActionResponse;
}

/**
 * POST /api/operator/action/unlock — request the
 * `LOCKED → SAFE` transition through the governance bridge.
 *
 * Mirrors the kill route shape: typed request, decision-bearing
 * `OperatorActionResponse` (approved + summary + decision +
 * audit_id) so the dashboard can show the bridge's reason for
 * approval or refusal.
 */
export async function postOperatorUnlock(
  body: OperatorUnlockRequest = {},
): Promise<OperatorActionResponse> {
  const res = await fetch(`${BASE}/api/operator/action/unlock`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      reason: body.reason ?? "operator unlock",
      requestor: body.requestor ?? "operator",
    }),
  });
  if (!res.ok) {
    throw new Error(
      `POST /api/operator/action/unlock failed: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as OperatorActionResponse;
}

/**
 * POST /api/operator/action/mode — request a `REQUEST_MODE`
 * transition (e.g. `SAFE → PAPER`, `LIVE → AUTO`). Hardening-S1
 * item 8 edges (`SAFE → PAPER` and `LIVE → AUTO`) require the
 * full consent envelope; other forward edges accept just
 * `target_mode + reason`.
 */
export async function postOperatorMode(
  body: OperatorModeRequest,
): Promise<OperatorActionResponse> {
  const res = await fetch(`${BASE}/api/operator/action/mode`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      target_mode: body.target_mode,
      reason: body.reason ?? "operator mode request",
      requestor: body.requestor ?? "operator",
      operator_authorized: body.operator_authorized ?? false,
      consent_operator_id: body.consent_operator_id ?? "",
      consent_policy_hash: body.consent_policy_hash ?? "",
      consent_nonce: body.consent_nonce ?? "",
      consent_ts_ns: body.consent_ts_ns ?? 0,
    }),
  });
  if (!res.ok) {
    throw new Error(
      `POST /api/operator/action/mode failed: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as OperatorActionResponse;
}
