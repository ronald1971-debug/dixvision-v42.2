/**
 * Shared API-base helper. Every fetch / EventSource in the SPA goes
 * through this so the `VITE_API_BASE` env var is honoured uniformly.
 *
 * Empty default keeps current behaviour (relative URLs against the
 * FastAPI host that serves the SPA), while non-empty overrides — e.g.
 * a CDN-hosted SPA pointing at a separate API host or a /proxy/ prefix
 * — work without touching widget code.
 */
export const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(
  /\/$/,
  "",
);

export function apiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  if (!path.startsWith("/")) {
    return `${API_BASE}/${path}`;
  }
  return `${API_BASE}${path}`;
}
