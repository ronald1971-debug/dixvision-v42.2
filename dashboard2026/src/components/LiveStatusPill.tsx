import { useIsFetching, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

/**
 * Header pill that reflects whether the dashboard is currently
 * fetching live data and how stale the most recent *successful* fetch
 * is.
 *
 * The pill subscribes to TanStack's {@link QueryCache} and records the
 * wall-clock timestamp every time a query transitions to a successful
 * settled state. Failed fetches never advance the freshness clock — so
 * during a backend outage the pill correctly degrades through stale
 * into offline.
 *
 * The active-fetch indicator additionally watches {@link useIsFetching}
 * so the pill momentarily shows "live · syncing" while a poll is in
 * flight (whether or not it ultimately succeeds).
 *
 * State machine
 * -------------
 * - **LIVE**     fetch in flight or last successful fetch ≤ 5 s ago
 * - **STALE**    last successful fetch > 5 s and ≤ 30 s ago
 * - **OFFLINE**  last successful fetch > 30 s ago, or no fetch has
 *                ever succeeded
 *
 * The component is purely visual; it never blocks rendering, never
 * raises, and has no impact on the rest of the SPA when polling is
 * paused (e.g. tab hidden in browser).
 */
export function LiveStatusPill() {
  const queryClient = useQueryClient();
  const inFlight = useIsFetching();

  // Seed lastSuccessAt from any query that has already produced data
  // before this component mounted (e.g. on route change), so we don't
  // briefly render OFFLINE while the first poll lands.
  const [lastSuccessAt, setLastSuccessAt] = useState<number | null>(() => {
    const queries = queryClient.getQueryCache().getAll();
    let max: number | null = null;
    for (const q of queries) {
      const updated = q.state.dataUpdatedAt;
      if (updated && updated > 0 && (max === null || updated > max)) {
        max = updated;
      }
    }
    return max;
  });
  const [now, setNow] = useState<number>(() => Date.now());

  // Bump the wall-clock once per second so the pill's age label is
  // always accurate without depending on TanStack re-renders.
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(id);
  }, []);

  // Subscribe to the global QueryCache and only advance the freshness
  // clock when at least one query has actually succeeded (i.e. has
  // fresh `data` and `status === 'success'`). Errored fetches are
  // intentionally ignored — that's the whole point of the offline
  // indicator.
  useEffect(() => {
    const cache = queryClient.getQueryCache();
    return cache.subscribe((event) => {
      if (event.type !== "updated") return;
      const action = event.action;
      if (!action || action.type !== "success") return;
      const query = event.query;
      if (query.state.status !== "success") return;
      const ts = query.state.dataUpdatedAt;
      if (ts && ts > 0) {
        setLastSuccessAt((prev) => (prev === null || ts > prev ? ts : prev));
      }
    });
  }, [queryClient]);

  const ageMs = lastSuccessAt === null ? Infinity : now - lastSuccessAt;
  const fetching = inFlight > 0;

  let label: string;
  let className: string;

  if (fetching && lastSuccessAt !== null) {
    label = "live · syncing";
    className = "bg-accent/10 border-accent text-accent";
  } else if (lastSuccessAt === null) {
    label = fetching ? "connecting…" : "offline";
    className = fetching
      ? "bg-accent/10 border-accent text-accent"
      : "bg-red-500/10 border-red-500 text-red-400";
  } else if (ageMs <= 5_000) {
    label = `live · ${Math.max(1, Math.round(ageMs / 1_000))}s`;
    className = "bg-emerald-500/10 border-emerald-500 text-emerald-400";
  } else if (ageMs <= 30_000) {
    label = `stale · ${Math.round(ageMs / 1_000)}s`;
    className = "bg-amber-500/10 border-amber-500 text-amber-400";
  } else {
    label = "offline";
    className = "bg-red-500/10 border-red-500 text-red-400";
  }

  return (
    <span
      data-testid="live-status-pill"
      className={`rounded border px-2 py-1 font-mono text-xs ${className}`}
      title={
        lastSuccessAt === null
          ? "No successful fetch yet"
          : `Last successful fetch: ${new Date(lastSuccessAt).toLocaleTimeString()}`
      }
    >
      {label}
    </span>
  );
}
