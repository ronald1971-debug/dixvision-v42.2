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
    // Read the body as text first, then attempt JSON.parse — the Fetch API
    // marks the body stream as "used" after any body method is invoked, so
    // calling `res.text()` after a failed `res.json()` would always throw
    // "body stream already read" and silently lose the server's error
    // payload (Devin Review BUG_0001 on PR #181).
    let body: unknown = null;
    try {
      const raw = await res.text();
      try {
        body = JSON.parse(raw);
      } catch {
        body = raw;
      }
    } catch {
      body = null;
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
