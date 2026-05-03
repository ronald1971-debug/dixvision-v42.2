import { AggressorRatio } from "@/widgets/orderflow/AggressorRatio";
import { CVDChart } from "@/widgets/orderflow/CVDChart";
import { DOMClickLadder } from "@/widgets/orderflow/DOMClickLadder";
import { FootprintChart } from "@/widgets/orderflow/FootprintChart";
import { LiquidityHeatmap } from "@/widgets/orderflow/LiquidityHeatmap";
import { SweepIcebergMonitor } from "@/widgets/orderflow/SweepIcebergMonitor";

/**
 * Tier-2 — Order-flow edge surface.
 *
 * Six widgets giving the operator the same depth-of-market signal a
 * Bookmap-class cockpit ships:
 *   - LiquidityHeatmap  (time × price × resting size)
 *   - FootprintChart    (per-price aggressor split)
 *   - CVDChart          (cumulative volume delta)
 *   - AggressorRatio    (rolling buy vs sell aggressor share)
 *   - SweepIcebergMonitor (sweep / iceberg / block detector)
 *   - DOMClickLadder    (click-to-stage limit orders, approval-gated)
 *
 * All widgets consume the canonical SSE `ticks` and `depth` channels.
 * In the absence of live feeds the SSE bridge runs the deterministic
 * mock generator so the surface still demonstrates structure.
 */
export function OrderFlowPage() {
  return (
    <div className="flex h-full flex-col gap-3 overflow-auto p-3">
      <header className="rounded border border-border bg-surface px-3 py-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
          Order-flow edge
        </h2>
        <p className="mt-1 text-[11px] leading-snug text-slate-400">
          Bookmap-class depth + footprint + CVD + aggressor ratio + sweep /
          iceberg / block detector + click-to-stage DOM ladder. All staged
          orders pass through the operator-approval edge (INV-72) before the
          execution engine sees them.
        </p>
      </header>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <div className="xl:col-span-2 h-[420px]">
          <LiquidityHeatmap />
        </div>
        <div className="h-[420px]">
          <DOMClickLadder />
        </div>
        <div className="h-[320px]">
          <FootprintChart />
        </div>
        <div className="h-[320px]">
          <CVDChart />
        </div>
        <div className="h-[320px]">
          <AggressorRatio />
        </div>
        <div className="h-[320px] xl:col-span-3">
          <SweepIcebergMonitor />
        </div>
      </div>
    </div>
  );
}
