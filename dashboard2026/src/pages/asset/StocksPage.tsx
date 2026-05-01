import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { PlaceholderWidget } from "@/components/PlaceholderWidget";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { DepthLadder } from "@/widgets/DepthLadder";
import { NewsTicker } from "@/widgets/NewsTicker";
import { OrderForm } from "@/widgets/OrderForm";
import { PositionsPanel } from "@/widgets/PositionsPanel";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";
import { TimeAndSalesTape } from "@/widgets/TimeAndSalesTape";

import { AssetPageShell } from "./AssetPageShell";

const SYMBOL = "AAPL";

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
    i: "level2",
    x: 8,
    y: 0,
    w: 4,
    h: 5,
    minW: 3,
    minH: 4,
    render: () => <DepthLadder symbol={SYMBOL} />,
  },
  {
    i: "tape",
    x: 8,
    y: 5,
    w: 4,
    h: 5,
    minW: 3,
    minH: 4,
    render: () => <TimeAndSalesTape symbol={SYMBOL} />,
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
    render: () => <SLTPBuilder form="stocks" />,
  },
  {
    i: "options",
    x: 0,
    y: 17,
    w: 4,
    h: 7,
    minW: 3,
    minH: 4,
    render: () => (
      <PlaceholderWidget
        title="Options Chain"
        subtitle="calls/puts · IV skew · OI · greeks (ToS-style)"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "fundamentals",
    x: 4,
    y: 17,
    w: 4,
    h: 7,
    minW: 3,
    minH: 4,
    render: () => (
      <PlaceholderWidget
        title="Fundamentals"
        subtitle="P/E · P/B · FCF · debt · insider · institutional · short interest"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "earnings",
    x: 8,
    y: 17,
    w: 4,
    h: 7,
    minW: 3,
    minH: 4,
    render: () => <NewsTicker />,
  },
  {
    i: "coherence",
    x: 0,
    y: 24,
    w: 12,
    h: 7,
    minW: 4,
    minH: 5,
    render: () => <CoherencePanel />,
  },
];

export function StocksPage() {
  return (
    <AssetPageShell
      title="Stocks"
      asset="STOCKS"
      description="Equities surface (Alpaca · IBKR · Tradier · Schwab/ToS bridge). Tax-lot aware, options-chain aware. Default widgets per PR-#2 spec §3.6."
    >
      <AssetGrid storageKey="stocks" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
