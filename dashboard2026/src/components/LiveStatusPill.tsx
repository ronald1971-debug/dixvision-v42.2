import { useIsFetching } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

/**
 * Header pill that reflects whether the dashboard is currently
 * fetching live data and how stale the most recent successful fetch
 * is.
 *
 * The pill watches every active TanStack Query in the app via
 * {@link useIsFetching}. Each time the in-flight count drops back to
 * zero we treat that as "a poll just completed", record the wall-clock
 * timestamp, and surface the age of that timestamp in human terms.
 *
 * State machine
 * -------------
 * - **LIVE**     fetch in flight or last success ≤ 5 s ago
 * - **STALE**    last success > 5 s and ≤ 30 s ago
 * - **OFFLINE**  last success > 30 s ago, or no fetch has succeeded yet
 *
 * The component is purely visual; it never blocks rendering, never
 * raises, and has no impact on the rest of the SPA when polling is
 * paused (e.g. tab hidden in browser).
 */
export function LiveStatusPill() {
  const inFlight = useIsFetching();
  const [lastFetchAt, setLastFetchAt] = useState<number | null>(null);
  const [now, setNow] = useState<number>(() => Date.now());
  const prevInFlightRef = useRef<number>(0);

  // Bump the wall-clock once per second so the pill's age label is
  // always accurate without depending on TanStack re-renders.
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(id);
  }, []);

  // A fetch just *finished* whenever the in-flight count transitions
  // from > 0 down to 0. Record the moment so we can age-out below.
  useEffect(() => {
    if (prevInFlightRef.current > 0 && inFlight === 0) {
      setLastFetchAt(Date.now());
    }
    prevInFlightRef.current = inFlight;
  }, [inFlight]);

  const ageMs = lastFetchAt === null ? Infinity : now - lastFetchAt;
  const fetching = inFlight > 0;

  let label: string;
  let className: string;

  if (fetching) {
    label = "live · syncing";
    className = "bg-accent/10 border-accent text-accent";
  } else if (lastFetchAt === null) {
    label = "offline";
    className = "bg-red-500/10 border-red-500 text-red-400";
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
        lastFetchAt === null
          ? "No successful fetch yet"
          : `Last successful fetch: ${new Date(lastFetchAt).toLocaleTimeString()}`
      }
    >
      {label}
    </span>
  );
}
