import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { PlaceholderWidget } from "@/components/PlaceholderWidget";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { DepthLadder } from "@/widgets/DepthLadder";
import { OrderForm } from "@/widgets/OrderForm";
import { PositionsPanel } from "@/widgets/PositionsPanel";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";

import { AssetPageShell } from "./AssetPageShell";

const SYMBOL = "EUR/USD";

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
    i: "session",
    x: 8,
    y: 0,
    w: 4,
    h: 4,
    minW: 3,
    minH: 3,
    render: () => (
      <PlaceholderWidget
        title="Session Clock"
        subtitle="Sydney · Tokyo · London · New York"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "depth",
    x: 8,
    y: 4,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => <DepthLadder symbol={SYMBOL} />,
  },
  {
    i: "order",
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
    render: () => <SLTPBuilder form="forex" />,
  },
  {
    i: "calendar",
    x: 0,
    y: 17,
    w: 6,
    h: 7,
    minW: 4,
    minH: 4,
    render: () => (
      <PlaceholderWidget
        title="Economic Calendar"
        subtitle="ForexFactory · TradingEconomics · auto-pause for FOMC/NFP/CPI"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "coherence",
    x: 6,
    y: 17,
    w: 6,
    h: 7,
    minW: 4,
    minH: 5,
    render: () => <CoherencePanel />,
  },
];

export function ForexPage() {
  return (
    <AssetPageShell
      title="Forex"
      asset="FOREX"
      description="Multi-broker FX surface (OANDA · IG · IBKR · MT4/MT5 bridge). Session-aware, calendar-gated. Default widgets per PR-#2 spec §3.5."
    >
      <AssetGrid storageKey="forex" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
