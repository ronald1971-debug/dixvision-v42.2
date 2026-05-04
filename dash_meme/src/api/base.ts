/**
 * DIX MEME → FastAPI harness HTTP client.
 *
 * All execution intents (manual orders, sniper hits, copy mirrors) flow
 * through `/api/dashboard/action/intent` — the SAME chokepoint that
 * `/dash2/` uses. There is no parallel authority surface here; every
 * write is mediated by Governance.
 */

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: unknown,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      try {
        body = await res.text();
      } catch {
        body = null;
      }
    }
    throw new ApiError(
      res.status,
      body,
      `HTTP ${res.status} ${res.statusText} on ${res.url}`,
    );
  }
  return (await res.json()) as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path, {
    method: "GET",
    credentials: "same-origin",
    headers: { Accept: "application/json" },
  });
  return unwrap<T>(res);
}

export async function apiPost<T>(
  path: string,
  body: Record<string, unknown>,
): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  return unwrap<T>(res);
}
