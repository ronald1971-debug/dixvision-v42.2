/**
 * Read-only fetchers for the `/api/dashboard/...` router (DASH-1).
 * The vanilla `dashboard_routes.py` already exposes these endpoints;
 * the wave-02 SPA consumes them through TanStack Query.
 */

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export interface ModeSnapshot {
  current_mode: string;
  legal_targets: string[];
  is_locked: boolean;
}

export interface ModeResponse {
  mode: ModeSnapshot;
}

export async function fetchMode(signal?: AbortSignal): Promise<ModeSnapshot> {
  const res = await fetch(`${BASE}/api/dashboard/mode`, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(
      `GET /api/dashboard/mode failed: ${res.status} ${res.statusText}`,
    );
  }
  const body = (await res.json()) as ModeResponse;
  return body.mode;
}
