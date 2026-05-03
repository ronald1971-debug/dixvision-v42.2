/**
 * Plugin manager API client.
 *
 * Backed by ``ui/plugin_routes.py``. Two endpoints:
 *
 * * ``GET  /api/plugins``                          → list every plugin
 * * ``POST /api/plugins/{id}/lifecycle``           → flip a plugin's
 *   lifecycle (DISABLED / SHADOW / ACTIVE). The server normalizes
 *   case and writes a ``PLUGIN_LIFECYCLE`` row to the authority
 *   ledger on success.
 */

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export interface PluginRecord {
  id: string;
  category: string;
  version: string;
  lifecycle: string;
  lifecycle_options: string[];
  description: string;
  ledger_kind: string;
}

export interface PluginListResponse {
  plugins: PluginRecord[];
}

export async function fetchPlugins(
  signal?: AbortSignal,
): Promise<PluginListResponse> {
  const res = await fetch(`${BASE}/api/plugins`, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(
      `GET /api/plugins failed: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as PluginListResponse;
}

export async function setPluginLifecycle(
  pluginId: string,
  lifecycle: string,
  opts: { reason?: string } = {},
): Promise<PluginRecord> {
  const res = await fetch(
    `${BASE}/api/plugins/${encodeURIComponent(pluginId)}/lifecycle`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        lifecycle,
        requestor: "dashboard",
        reason: opts.reason ?? "",
      }),
    },
  );
  if (!res.ok) {
    let detail = "";
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail ?? "";
    } catch {
      // ignore — fall back to status text
    }
    throw new Error(
      `POST /api/plugins/${pluginId}/lifecycle failed: ` +
        `${res.status} ${res.statusText}${detail ? ` — ${detail}` : ""}`,
    );
  }
  return (await res.json()) as PluginRecord;
}
