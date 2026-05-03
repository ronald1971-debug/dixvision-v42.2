import { useEffect, useState } from "react";

/**
 * Hash-based router for the wave-02 SPA.
 *
 * The 2026 dashboard rebuild (DASH-A) extends the route space from the
 * original three system pages (credentials/operator/chat) to also
 * cover the seven asset-class surfaces called out in PR #2 §3:
 * spot, perps, dex, memecoin (own dashboard per operator directive),
 * forex, stocks, nft.
 *
 * The FastAPI mount serves the SPA index for the `/dash2/` root only;
 * deep links use the `#/<route>` form so the server keeps serving
 * `index.html` regardless of path.
 */
export type AssetRoute =
  | "spot"
  | "perps"
  | "dex"
  | "memecoin"
  | "forex"
  | "stocks"
  | "nft";

export type SystemRoute =
  | "operator"
  | "credentials"
  | "chat"
  | "indira"
  | "dyon"
  | "testing"
  | "onchain"
  | "ai"
  | "orderflow"
  | "governance"
  | "risk"
  | "charting"
  | "market"
  | "positions"
  | "trading"
  | "plugins";

export type Route = AssetRoute | SystemRoute;

const ASSET_ROUTES: readonly AssetRoute[] = [
  "spot",
  "perps",
  "dex",
  "memecoin",
  "forex",
  "stocks",
  "nft",
];

const SYSTEM_ROUTES: readonly SystemRoute[] = [
  "operator",
  "credentials",
  "chat",
  "indira",
  "dyon",
  "testing",
  "onchain",
  "ai",
  "orderflow",
  "governance",
  "risk",
  "charting",
  "market",
  "positions",
  "trading",
  "plugins",
];

const ALL_ROUTES: readonly Route[] = [...ASSET_ROUTES, ...SYSTEM_ROUTES];

/**
 * Default landing route. Memecoin is the surface the operator
 * explicitly wanted as its own dashboard, so it is the first thing
 * shown when `/dash2/` is opened without a hash.
 */
export const DEFAULT_ROUTE: Route = "memecoin";

export function isAssetRoute(route: Route): route is AssetRoute {
  return (ASSET_ROUTES as readonly string[]).includes(route);
}

export function parseRoute(hash: string): Route {
  const cleaned = hash
    .replace(/^#\/?/, "")
    .replace(/^popout\//, "")
    .trim();
  if (cleaned === "") return DEFAULT_ROUTE;
  for (const route of ALL_ROUTES) {
    if (cleaned === route) return route;
  }
  return DEFAULT_ROUTE;
}

export function isPopoutRoute(hash: string): boolean {
  return hash.startsWith("#/popout/");
}

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() =>
    parseRoute(window.location.hash),
  );
  useEffect(() => {
    const handler = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);
  return route;
}

export function useIsPopout(): boolean {
  const [popout, setPopout] = useState<boolean>(() =>
    isPopoutRoute(window.location.hash),
  );
  useEffect(() => {
    const handler = () => setPopout(isPopoutRoute(window.location.hash));
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);
  return popout;
}

export const ASSET_ROUTE_LIST = ASSET_ROUTES;
export const SYSTEM_ROUTE_LIST = SYSTEM_ROUTES;
