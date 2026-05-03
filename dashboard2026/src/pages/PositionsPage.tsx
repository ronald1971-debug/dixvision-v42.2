import { DrawdownCurve } from "@/widgets/positions/DrawdownCurve";
import { ExposureBreakdown } from "@/widgets/positions/ExposureBreakdown";
import { FillsHistory } from "@/widgets/positions/FillsHistory";
import { FundingHistory } from "@/widgets/positions/FundingHistory";
import { IntradayPnLCurve } from "@/widgets/positions/IntradayPnLCurve";
import { OpenOrdersPanel } from "@/widgets/positions/OpenOrdersPanel";
import { RiskParityAllocator } from "@/widgets/positions/RiskParityAllocator";

/**
 * G-track surface — Positions / PnL.
 *
 * Seven widgets covering portfolio state and analytics:
 *   - OpenOrdersPanel       (working / partial)
 *   - FillsHistory          (audit-ledger projection)
 *   - IntradayPnLCurve      (mark-to-market)
 *   - DrawdownCurve         (underwater plot)
 *   - ExposureBreakdown     (sector / venue / asset pivot)
 *   - FundingHistory        (perp funding ledger)
 *   - RiskParityAllocator   (inverse-vol allocator + stage)
 */
export function PositionsPage() {
  return (
    <div className="grid h-full grid-cols-1 gap-3 overflow-auto p-3 lg:grid-cols-2 xl:grid-cols-3">
      <div className="min-h-[320px] xl:col-span-2">
        <IntradayPnLCurve />
      </div>
      <div className="min-h-[320px]">
        <DrawdownCurve />
      </div>
      <div className="min-h-[360px] xl:col-span-2">
        <OpenOrdersPanel />
      </div>
      <div className="min-h-[360px]">
        <ExposureBreakdown />
      </div>
      <div className="min-h-[360px] xl:col-span-2">
        <FillsHistory />
      </div>
      <div className="min-h-[360px]">
        <FundingHistory />
      </div>
      <div className="min-h-[360px] xl:col-span-3">
        <RiskParityAllocator />
      </div>
    </div>
  );
}
