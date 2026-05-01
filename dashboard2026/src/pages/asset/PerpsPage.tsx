import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { PlaceholderWidget } from "@/components/PlaceholderWidget";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { OrderForm } from "@/widgets/OrderForm";
import { PositionsPanel } from "@/widgets/PositionsPanel";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";

import { AssetPageShell } from "./AssetPageShell";

const SYMBOL = "BTC-PERP";

const ITEMS: GridItemSpec[] = [
  {
    i: "chart",
    x: 0,
    y: 0,
    w: 8,
    h: 12,
    minW: 4,
    minH: 6,
    render: () => <ChartPanel symbol={SYMBOL} />,
  },
  {
    i: "funding",
    x: 8,
    y: 0,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => (
      <PlaceholderWidget
        title="Funding Table"
        subtitle="per-venue · next-funding countdown · cumulative-funding PnL"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "liq",
    x: 8,
    y: 6,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => (
      <PlaceholderWidget
        title="Liquidation Heatmap"
        subtitle="open-interest bands · projected liq prices"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "order",
    x: 0,
    y: 12,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <OrderForm symbol={SYMBOL} />,
  },
  {
    i: "positions",
    x: 4,
    y: 12,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <PositionsPanel />,
  },
  {
    i: "sltp",
    x: 8,
    y: 12,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <SLTPBuilder form="perps" />,
  },
  {
    i: "coherence",
    x: 0,
    y: 19,
    w: 6,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <CoherencePanel />,
  },
  {
    i: "oracle",
    x: 6,
    y: 19,
    w: 6,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => (
      <PlaceholderWidget
        title="Per-venue Oracle"
        subtitle="oracle price vs exec price · divergence alarm"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
];

export function PerpsPage() {
  return (
    <AssetPageShell
      title="Perpetual Futures"
      asset="PERPS"
      description="Funding-rate, liquidation cascades, oracle-spread monitoring. Default widgets per PR-#2 spec §3.2 (Hyperliquid HIP-3 / dYdX / Drift parity)."
    >
      <AssetGrid storageKey="perps" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
