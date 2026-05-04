import { useStreamState } from "@/state/realtime";

/**
 * Persistent operator banner that appears whenever the real-time
 * bridge (``state/realtime.ts``) has fallen back to its mock
 * generator.
 *
 * AUDIT-P1.4 — when SSE at ``/api/dashboard/stream`` is unreachable
 * the bridge silently swaps in a deterministic Math.random / sin
 * mock so widgets keep rendering. The only previous indicator was
 * the small {@link LiveStatusPill}, which the operator can easily
 * miss on a busy multi-monitor layout. The 6 order-flow widgets
 * (LiquidityHeatmap, FootprintChart, CVDChart, AggressorRatio,
 * SweepIcebergMonitor, DOMClickLadder) all consume this stream and
 * with the mock active they show synthetic data that is visually
 * indistinguishable from real venue feed.
 *
 * The banner therefore renders a high-contrast amber strip with a
 * dismiss-resistant message so the operator always knows when the
 * orderflow surface is in mock mode. The banner disappears as soon
 * as live data resumes.
 *
 * The component is purely visual — it never blocks rendering, never
 * raises, and observes only ``useStreamState`` (which already powers
 * the LiveStatusPill).
 */
export function MockDataBanner() {
  const state = useStreamState();
  if (state !== "mock") return null;
  return (
    <div
      data-testid="mock-data-banner"
      role="alert"
      className="flex items-center justify-center gap-2 border-b border-amber-500/60 bg-amber-500/10 px-3 py-1 text-xs font-medium text-amber-300"
    >
      <span aria-hidden="true">⚠</span>
      <span>
        Streaming offline — orderflow widgets are showing
        deterministic mock data, not live venue feed.
      </span>
    </div>
  );
}
