import { FearGreed } from "@/widgets/market/FearGreed";
import { HotMovers } from "@/widgets/market/HotMovers";
import { IVSurface } from "@/widgets/market/IVSurface";
import { LongShortRatio } from "@/widgets/market/LongShortRatio";
import { OpenInterestPanel } from "@/widgets/market/OpenInterestPanel";
import { PutCallRatio } from "@/widgets/market/PutCallRatio";
import { SentimentGauge } from "@/widgets/market/SentimentGauge";
import { Watchlist } from "@/widgets/market/Watchlist";

/**
 * H-track surface — market context.
 *
 * Mounts 8 read-only widgets that give the operator a 360° view of
 * positioning, sentiment, and option-implied risk across crypto/perp
 * venues. None of these stage execution intents; they exist purely
 * to inform the operator before they touch the F-track entry surface.
 */
export function MarketContextPage() {
  return (
    <div className="space-y-3">
      <header className="rounded border border-border bg-surface px-3 py-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
          Market context
        </h2>
        <p className="mt-0.5 text-[11px] text-slate-500">
          watchlist · movers · sentiment composite · F&amp;G · long/short ·
          OI · put/call · IV surface — all read-only, all SCVS-registered.
        </p>
      </header>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-3">
        <div className="min-h-[360px] xl:col-span-2">
          <Watchlist />
        </div>
        <div className="min-h-[360px]">
          <HotMovers />
        </div>
        <div className="min-h-[300px]">
          <SentimentGauge />
        </div>
        <div className="min-h-[300px]">
          <FearGreed />
        </div>
        <div className="min-h-[300px]">
          <PutCallRatio />
        </div>
        <div className="min-h-[280px]">
          <LongShortRatio />
        </div>
        <div className="min-h-[280px]">
          <OpenInterestPanel />
        </div>
        <div className="min-h-[320px] xl:col-span-3">
          <IVSurface />
        </div>
      </div>
    </div>
  );
}
