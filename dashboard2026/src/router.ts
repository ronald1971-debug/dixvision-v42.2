import { useEffect, useState } from "react";

/**
 * Minimal hash-based router. Avoids pulling in `react-router` for what
 * is, today, two pages. The FastAPI mount serves the SPA index for the
 * `/dash2/` root only; deep links use the `#/<route>` form so the
 * server keeps serving `index.html` regardless of path.
 */
export type Route = "credentials" | "operator" | "chat";

const ROUTES: readonly Route[] = ["credentials", "operator", "chat"];

export function parseRoute(hash: string): Route {
  const cleaned = hash.replace(/^#\/?/, "").trim();
  if (cleaned === "") return "credentials";
  for (const route of ROUTES) {
    if (cleaned === route) return route;
  }
  return "credentials";
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
