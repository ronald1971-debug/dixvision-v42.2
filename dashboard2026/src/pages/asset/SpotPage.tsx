import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { DepthLadder } from "@/widgets/DepthLadder";
import { NewsTicker } from "@/widgets/NewsTicker";
import { OrderForm } from "@/widgets/OrderForm";
import { PositionsPanel } from "@/widgets/PositionsPanel";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";
import { TimeAndSalesTape } from "@/widgets/TimeAndSalesTape";

import { AssetPageShell } from "./AssetPageShell";

const SYMBOL = "BTC/USDC";

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
    i: "depth",
    x: 8,
    y: 0,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => <DepthLadder symbol={SYMBOL} />,
  },
  {
    i: "tape",
    x: 8,
    y: 6,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => <TimeAndSalesTape symbol={SYMBOL} />,
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
    render: () => <SLTPBuilder form="spot" />,
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
    i: "news",
    x: 6,
    y: 19,
    w: 6,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <NewsTicker />,
  },
];

export function SpotPage() {
  return (
    <AssetPageShell
      title="Spot"
      asset="SPOT"
      description="Crypto / equity spot surface. Same SL/TP engine as every other form; default widgets follow PR-#2 spec §3.1."
    >
      <AssetGrid storageKey="spot" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
