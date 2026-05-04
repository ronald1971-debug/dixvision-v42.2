import { useEffect, useState } from "react";

/**
 * Selected pair — the one the explorer / multichart / trade pages key
 * off. Persisted in localStorage so a page refresh keeps the operator
 * on the same token. The shape is intentionally loose so we can carry
 * raw provider IDs (mint address, pool address, etc.) without fighting
 * the type system before we extract a strict TS type from the backend.
 */
export type SelectedPair = {
  /** Display symbol e.g. "BONK/SOL" — operator-facing label. */
  symbol: string;
  /** "solana" / "ethereum" / "base" / "bsc". */
  chain: string;
  /** Provider-specific id — Pump.fun mint, Raydium pool, …  optional. */
  poolId?: string;
};

const STORAGE_KEY = "dixmeme.selected_pair";

const DEFAULT_PAIR: SelectedPair = {
  symbol: "BONK/SOL",
  chain: "solana",
};

function readStored(): SelectedPair {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (
        parsed &&
        typeof parsed === "object" &&
        typeof (parsed as { symbol?: unknown }).symbol === "string" &&
        typeof (parsed as { chain?: unknown }).chain === "string"
      ) {
        return parsed as SelectedPair;
      }
    }
  } catch {
    // ignore
  }
  return DEFAULT_PAIR;
}

const listeners = new Set<(p: SelectedPair) => void>();
let current = readStored();

export function getSelectedPair(): SelectedPair {
  return current;
}

export function setSelectedPair(pair: SelectedPair) {
  current = pair;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(pair));
  } catch {
    // ignore
  }
  listeners.forEach((fn) => fn(pair));
}

export function useSelectedPair(): [
  SelectedPair,
  (p: SelectedPair) => void,
] {
  const [pair, setPair] = useState<SelectedPair>(current);
  useEffect(() => {
    const fn = (p: SelectedPair) => setPair(p);
    listeners.add(fn);
    return () => {
      listeners.delete(fn);
    };
  }, []);
  return [pair, setSelectedPair];
}
