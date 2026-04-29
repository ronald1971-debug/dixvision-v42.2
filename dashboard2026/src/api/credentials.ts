import type { CredentialsStatusResponse } from "@/types/generated/api";

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export async function fetchCredentialsStatus(
  signal?: AbortSignal,
): Promise<CredentialsStatusResponse> {
  const res = await fetch(`${BASE}/api/credentials/status`, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(
      `GET /api/credentials/status failed: ${res.status} ${res.statusText}`,
    );
  }
  // The server validates the shape via a Pydantic response_model, so
  // the JSON body matches `CredentialsStatusResponse` byte-for-byte.
  // We still cast at the boundary instead of running a runtime
  // validator — keeping the wave-02 dependency surface narrow.
  return (await res.json()) as CredentialsStatusResponse;
}
