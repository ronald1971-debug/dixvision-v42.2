import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { GasEstimator } from "@/widgets/dex/GasEstimator";
import { PoolHealth } from "@/widgets/dex/PoolHealth";
import { RouteGraph } from "@/widgets/dex/RouteGraph";
import { OrderForm } from "@/widgets/OrderForm";
import { PositionsPanel } from "@/widgets/PositionsPanel";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";

import { AssetPageShell } from "./AssetPageShell";

const SYMBOL = "SOL/USDC";

const ITEMS: GridItemSpec[] = [
  {
    i: "chart",
    x: 0,
    y: 0,
    w: 8,
    h: 10,
    minW: 4,
    minH: 6,
    render: () => <ChartPanel symbol={SYMBOL} />,
  },
  {
    i: "route",
    x: 8,
    y: 0,
    w: 4,
    h: 5,
    minW: 3,
    minH: 4,
    render: () => <RouteGraph symbol={SYMBOL} />,
  },
  {
    i: "pool",
    x: 8,
    y: 5,
    w: 4,
    h: 5,
    minW: 3,
    minH: 4,
    render: () => <PoolHealth symbol={SYMBOL} />,
  },
  {
    i: "swap",
    x: 0,
    y: 10,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <OrderForm symbol={SYMBOL} />,
  },
  {
    i: "positions",
    x: 4,
    y: 10,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <PositionsPanel />,
  },
  {
    i: "sltp",
    x: 8,
    y: 10,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <SLTPBuilder form="dex" />,
  },
  {
    i: "coherence",
    x: 0,
    y: 17,
    w: 6,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <CoherencePanel />,
  },
  {
    i: "gas",
    x: 6,
    y: 17,
    w: 6,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <GasEstimator />,
  },
];

export function DexPage() {
  return (
    <AssetPageShell
      title="DEX / DeFi"
      asset="DEX"
      description="Intent-based execution, MEV-aware routing, and synthesized stops. Default widgets per PR-#2 spec §3.3 (Jupiter Juno / 1inch Fusion+ / CowSwap)."
    >
      <AssetGrid storageKey="dex" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
