import { useEffect, useState } from "react";

import { apiUrl } from "@/api/base";

/**
 * Autonomy Mode (PR-#2 spec §0.2 + §6).
 *
 * This is the *operator-attention* axis, orthogonal to the System
 * Mode FSM (LOCKED/SAFE/PAPER/CANARY/LIVE/AUTO):
 *
 *   - USER_CONTROLLED: every intent waits for an operator click.
 *   - SEMI_AUTO: auto-trades inside operator-set envelope; settings
 *     adjustable on the fly.
 *   - FULL_AUTO: auto-trades without asking; one-click fall-back to
 *     SEMI_AUTO / USER_CONTROLLED while trades are running.
 *
 * Persisted to `localStorage` so the operator's last choice survives
 * a reload. Per the spec every change must emit
 * `OPERATOR/SETTINGS_CHANGED` with before/after + mode-at-time + ISO
 * timestamp; the audit hook is centralised here.
 */
export type AutonomyMode = "USER_CONTROLLED" | "SEMI_AUTO" | "FULL_AUTO";

export const AUTONOMY_MODES: readonly AutonomyMode[] = [
  "USER_CONTROLLED",
  "SEMI_AUTO",
  "FULL_AUTO",
];

const STORAGE_KEY = "dixvision.dash2026.autonomy";

function readStored(): AutonomyMode {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw && (AUTONOMY_MODES as readonly string[]).includes(raw)) {
      return raw as AutonomyMode;
    }
  } catch {
    // localStorage unavailable — safest default below.
  }
  return "USER_CONTROLLED";
}

function writeStored(mode: AutonomyMode): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // ignore quota / privacy mode
  }
}

const listeners = new Set<(mode: AutonomyMode) => void>();
let current: AutonomyMode = readStored();

export function getAutonomyMode(): AutonomyMode {
  return current;
}

export function setAutonomyMode(mode: AutonomyMode): void {
  if (mode === current) return;
  const previous = current;
  current = mode;
  writeStored(mode);
  // Audit hook: spec §0.2 + §1.3 — every settings dial writes
  // OPERATOR/SETTINGS_CHANGED to the ledger. Until the backend route
  // is wired we emit to console + a fire-and-forget POST that the
  // server may ignore.
  void emitAudit(previous, mode);
  for (const listener of listeners) listener(mode);
}

async function emitAudit(
  previous: AutonomyMode,
  next: AutonomyMode,
): Promise<void> {
  const payload = {
    kind: "OPERATOR/SETTINGS_CHANGED",
    setting: "autonomy_mode",
    previous,
    next,
    timestamp_iso: new Date().toISOString(),
  };
  try {
    await fetch(apiUrl("/api/operator/audit"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,
    });
  } catch {
    // Best-effort. Local audit fallback.
  }
  if (typeof window !== "undefined" && "console" in window) {
    console.info("[audit]", payload);
  }
}

export function useAutonomyMode(): [
  AutonomyMode,
  (next: AutonomyMode) => void,
] {
  const [mode, setMode] = useState<AutonomyMode>(current);
  useEffect(() => {
    const listener = (next: AutonomyMode) => setMode(next);
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);
  return [mode, setAutonomyMode];
}
