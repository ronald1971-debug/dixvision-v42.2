/**
 * Per-widget live/mock status chip.
 *
 * The dashboard ships with two grades of indicator:
 *
 *   1. The global ``MockDataBanner`` strip at the top of the page,
 *      which fires only when the SSE bridge in ``state/realtime.ts``
 *      has fallen back to its deterministic mock generator.
 *   2. The ``LiveStatusPill`` in the nav, which counts polling
 *      successes vs failures across the whole surface.
 *
 * Both are surface-wide. P1.6 closes the per-widget gap so each
 * panel exposes a small chip that announces, for that panel only,
 * whether its data is the real backend projection or the in-widget
 * deterministic FALLBACK skeleton. The pattern was first introduced
 * by the six PR #351 widgets (RouteGraph, PoolHealth, GasEstimator,
 * FundingTable, OracleSpread, LiquidationMap); this module extracts
 * it into one reusable component so every mock-fed widget on the
 * dashboard can show the same indicator.
 *
 * The chip is intentionally minimal — it is a single rounded badge
 * that fits inside any widget header without rebalancing its
 * layout. ``mode`` controls only the colour and label.
 */
export type WidgetSourceMode = "live" | "mock";

export interface WidgetStatusChipProps {
  /**
   * Whether the widget's currently rendered data is the live backend
   * projection (``"live"``) or the in-widget deterministic skeleton
   * (``"mock"``). Most pre-P1.5 widgets have no backend route yet
   * and pass ``"mock"`` literally; widgets that wire a ``useQuery``
   * over their projection route pass ``live ? "live" : "mock"``.
   */
  mode: WidgetSourceMode;
  /**
   * Optional override of the chip label. Defaults to the upper-cased
   * mode (``LIVE``/``MOCK``).
   */
  label?: string;
  /**
   * Optional extra Tailwind utility classes appended to the chip.
   * Use sparingly — the chip already fits the surface's compact
   * widget header.
   */
  className?: string;
}

const BASE_CLASS =
  "rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase";
const LIVE_CLASS = "border-emerald-500/40 bg-emerald-500/10 text-emerald-300";
const MOCK_CLASS = "border-amber-500/40 bg-amber-500/10 text-amber-300";

export function WidgetStatusChip({
  mode,
  label,
  className,
}: WidgetStatusChipProps) {
  const tone = mode === "live" ? LIVE_CLASS : MOCK_CLASS;
  const text = label ?? (mode === "live" ? "LIVE" : "MOCK");
  return (
    <span
      data-widget-source={mode}
      className={`${BASE_CLASS} ${tone}${className ? ` ${className}` : ""}`}
    >
      {text}
    </span>
  );
}
