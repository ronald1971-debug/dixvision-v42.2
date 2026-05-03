import { ExchangeFlows } from "@/widgets/onchain/ExchangeFlows";
import { OpenInterestMatrix } from "@/widgets/onchain/OpenInterestMatrix";
import { StablecoinSupply } from "@/widgets/onchain/StablecoinSupply";
import { TVLDashboard } from "@/widgets/onchain/TVLDashboard";
import { WhaleWatcher } from "@/widgets/onchain/WhaleWatcher";

/**
 * Tier-5 on-chain analytics surface.
 *
 * Cross-asset macro signals that don't belong to a single asset
 * page: whale transfers, exchange net-flows, stablecoin supply,
 * DeFi TVL, and the perp OI matrix. Mounted under #/onchain in
 * the System nav group.
 */
export function OnChainPage() {
  return (
    <div className="grid h-full grid-cols-1 gap-3 overflow-auto p-3 lg:grid-cols-2 xl:grid-cols-3">
      <div className="min-h-[360px] xl:col-span-2">
        <WhaleWatcher />
      </div>
      <div className="min-h-[360px]">
        <ExchangeFlows />
      </div>
      <div className="min-h-[360px]">
        <StablecoinSupply />
      </div>
      <div className="min-h-[360px] xl:col-span-2">
        <TVLDashboard />
      </div>
      <div className="min-h-[360px] xl:col-span-3">
        <OpenInterestMatrix />
      </div>
    </div>
  );
}
