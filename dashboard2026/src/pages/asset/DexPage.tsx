import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { PlaceholderWidget } from "@/components/PlaceholderWidget";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
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
    render: () => (
      <PlaceholderWidget
        title="Route Graph"
        subtitle="Jupiter Juno · 1inch Fusion+ · CowSwap solver auction"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "pool",
    x: 8,
    y: 5,
    w: 4,
    h: 5,
    minW: 3,
    minH: 4,
    render: () => (
      <PlaceholderWidget
        title="Pool Health"
        subtitle="liquidity · 24h volume · LP concentration"
        badge="DASH-K"
        status="stub"
      />
    ),
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
    render: () => (
      <PlaceholderWidget
        title="Gas Estimator"
        subtitle="Helius p50/p75/p90 · base-fee + tip · MEV-protected RPC"
        badge="DASH-K"
        status="stub"
      />
    ),
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
