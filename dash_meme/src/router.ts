import { useEffect, useState } from "react";

// DIX MEME route names. Each maps 1:1 to a top-level page component.
export type Route =
  | "explorer"      // landing — Pair Explorer (chart + audit + holders)
  | "pools"         // Pool Explorer
  | "bigswap"       // Big Swap Explorer (large tx feed)
  | "multichart"    // 2x2 / 4x1 multi-pair chart grid
  | "trade"         // manual / semi-auto / full-auto order entry
  | "copy"          // CopyTrading — wallet allowlist + mirrors
  | "sniper"        // Sniper — pre-launch / migration queue
  | "multiswap"     // multi-pair execution batching
  | "wallet"        // Wallet Info — balances + history
  | "stats";        // global stats — gainers / losers / hot / new

const ALL_ROUTES: readonly Route[] = [
  "explorer",
  "pools",
  "bigswap",
  "multichart",
  "trade",
  "copy",
  "sniper",
  "multiswap",
  "wallet",
  "stats",
];

const DEFAULT_ROUTE: Route = "explorer";

function parseHash(hash: string): Route {
  const stripped = hash.replace(/^#\/?/, "").split("/")[0] ?? "";
  if ((ALL_ROUTES as readonly string[]).includes(stripped)) {
    return stripped as Route;
  }
  return DEFAULT_ROUTE;
}

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() =>
    parseHash(window.location.hash),
  );

  useEffect(() => {
    const onHash = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return route;
}

export function navigate(route: Route, suffix = "") {
  window.location.hash = `#/${route}${suffix}`;
}

// Optional sub-state in the URL hash, e.g. `#/explorer/SOL/BONK`.
export function useHashSuffix(): string {
  const [suffix, setSuffix] = useState<string>(() =>
    extractSuffix(window.location.hash),
  );
  useEffect(() => {
    const onHash = () => setSuffix(extractSuffix(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return suffix;
}

function extractSuffix(hash: string): string {
  const parts = hash.replace(/^#\/?/, "").split("/");
  return parts.slice(1).join("/");
}
