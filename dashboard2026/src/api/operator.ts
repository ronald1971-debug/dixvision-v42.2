import type {
  OperatorActionResponse,
  OperatorSummaryResponse,
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
