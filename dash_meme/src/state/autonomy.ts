import { useEffect, useState } from "react";

/**
 * Autonomy band — the operator-facing slider on TradePage / SniperPage /
 * CopyTradingPage. Maps to the `risk_mode` field on outgoing
 * `IntentRequest`s, which Governance interprets as the operator-approval
 * requirement to apply:
 *
 *   manual    → every intent is a one-shot, operator confirms.
 *   semi-auto → operator pre-authorises within risk caps; intents above
 *               cap fall back to manual.
 *   full-auto → AUTO mode (per Hardening-S1.F drift oracle); attention
 *               relaxed within drift bounds.
 *
 * No autonomy mode is a parallel execution path — they all funnel
 * through the same /api/dashboard/action/intent chokepoint.
 */
export type AutonomyMode = "manual" | "semi-auto" | "full-auto";

const STORAGE_KEY = "dixmeme.autonomy";

function readStored(): AutonomyMode {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw === "manual" || raw === "semi-auto" || raw === "full-auto") {
      return raw;
    }
  } catch {
    // localStorage may be disabled (private mode) — fall through.
  }
  return "manual";
}

const listeners = new Set<(mode: AutonomyMode) => void>();
let current: AutonomyMode = readStored();

export function getAutonomy(): AutonomyMode {
  return current;
}

export function setAutonomy(mode: AutonomyMode) {
  current = mode;
  try {
    window.localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // ignore — we still notify in-memory subscribers.
  }
  listeners.forEach((fn) => fn(mode));
}

export function useAutonomy(): [AutonomyMode, (m: AutonomyMode) => void] {
  const [mode, setMode] = useState<AutonomyMode>(current);
  useEffect(() => {
    const fn = (next: AutonomyMode) => setMode(next);
    listeners.add(fn);
    return () => {
      listeners.delete(fn);
    };
  }, []);
  return [mode, setAutonomy];
}
