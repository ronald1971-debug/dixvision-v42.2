import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { DepthLadder } from "@/widgets/DepthLadder";
import { NewsTicker } from "@/widgets/NewsTicker";
import { OrderForm } from "@/widgets/OrderForm";
import { PositionsPanel } from "@/widgets/PositionsPanel";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";
import { TimeAndSalesTape } from "@/widgets/TimeAndSalesTape";
import { AnalystRatings } from "@/widgets/stocks/AnalystRatings";
import { Fundamentals } from "@/widgets/stocks/Fundamentals";
import { InsiderTransactions } from "@/widgets/stocks/InsiderTransactions";
import { SectorHeatmap } from "@/widgets/stocks/SectorHeatmap";
import { ShortInterest } from "@/widgets/stocks/ShortInterest";

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
    i: "fundamentals",
    x: 0,
    y: 17,
    w: 6,
    h: 9,
    minW: 4,
    minH: 5,
    render: () => <Fundamentals symbol={SYMBOL} />,
  },
  {
    i: "ratings",
    x: 6,
    y: 17,
    w: 6,
    h: 9,
    minW: 4,
    minH: 5,
    render: () => <AnalystRatings symbol={SYMBOL} />,
  },
  {
    i: "insider",
    x: 0,
    y: 26,
    w: 6,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <InsiderTransactions symbol={SYMBOL} />,
  },
  {
    i: "short",
    x: 6,
    y: 26,
    w: 6,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <ShortInterest symbol={SYMBOL} />,
  },
  {
    i: "sectors",
    x: 0,
    y: 34,
    w: 8,
    h: 7,
    minW: 4,
    minH: 5,
    render: () => <SectorHeatmap />,
  },
  {
    i: "earnings",
    x: 8,
    y: 34,
    w: 4,
    h: 7,
    minW: 3,
    minH: 4,
    render: () => <NewsTicker />,
  },
  {
    i: "coherence",
    x: 0,
    y: 41,
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
      description="Equities surface (Alpaca · IBKR · Tradier · Schwab/ToS bridge). Pro pack: Fundamentals · AnalystRatings · InsiderTx · ShortInterest · SectorHeatmap. Tax-lot aware, options-chain aware."
    >
      <AssetGrid storageKey="stocks" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
