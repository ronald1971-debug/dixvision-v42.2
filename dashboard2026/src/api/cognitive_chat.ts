import type {
  ApprovalDecisionRequest,
  ApprovalDecisionResponse,
  ApprovalsListResponse,
  ChatStatusResponse,
  ChatTurnRequest,
  ChatTurnResponse,
} from "@/types/generated/api";

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export async function fetchChatStatus(
  signal?: AbortSignal,
): Promise<ChatStatusResponse> {
  const res = await fetch(`${BASE}/api/cognitive/chat/status`, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(
      `GET /api/cognitive/chat/status failed: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as ChatStatusResponse;
}

export async function postChatTurn(
  body: ChatTurnRequest,
): Promise<ChatTurnResponse> {
  const res = await fetch(`${BASE}/api/cognitive/chat/turn`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    // Surface the FastAPI detail message in the thrown Error so the
    // page can render it verbatim — useful when the server returns
    // a 502 "no chat transport configured" or a 503 "feature disabled".
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      // ignore — the body was not JSON; fall back to status line
    }
    throw new Error(`POST /api/cognitive/chat/turn failed: ${detail}`);
  }
  return (await res.json()) as ChatTurnResponse;
}

export async function fetchApprovals(
  options: { includeDecided?: boolean; signal?: AbortSignal } = {},
): Promise<ApprovalsListResponse> {
  const includeDecided = options.includeDecided ?? false;
  const url =
    `${BASE}/api/cognitive/chat/approvals` +
    (includeDecided ? `?include_decided=true` : ``);
  const res = await fetch(url, {
    signal: options.signal,
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(
      `GET /api/cognitive/chat/approvals failed: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as ApprovalsListResponse;
}

async function decideApproval(
  requestId: string,
  verb: "approve" | "reject",
  body: ApprovalDecisionRequest = {},
): Promise<ApprovalDecisionResponse> {
  const res = await fetch(
    `${BASE}/api/cognitive/chat/approvals/${encodeURIComponent(requestId)}/${verb}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(body),
    },
  );
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      // ignore — body was not JSON
    }
    throw new Error(
      `POST /api/cognitive/chat/approvals/${requestId}/${verb} failed: ${detail}`,
    );
  }
  return (await res.json()) as ApprovalDecisionResponse;
}

export function approveApproval(
  requestId: string,
  body: ApprovalDecisionRequest = {},
): Promise<ApprovalDecisionResponse> {
  return decideApproval(requestId, "approve", body);
}

export function rejectApproval(
  requestId: string,
  body: ApprovalDecisionRequest = {},
): Promise<ApprovalDecisionResponse> {
  return decideApproval(requestId, "reject", body);
}
