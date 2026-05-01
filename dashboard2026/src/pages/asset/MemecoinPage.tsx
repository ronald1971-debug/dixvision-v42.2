import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { AlertsHub } from "@/widgets/AlertsHub";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { OrderForm } from "@/widgets/OrderForm";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";
import { TimeAndSalesTape } from "@/widgets/TimeAndSalesTape";
import { CopyLeaderboard } from "@/widgets/memecoin/CopyLeaderboard";
import { HolderConcentration } from "@/widgets/memecoin/HolderConcentration";
import { PairCard } from "@/widgets/memecoin/PairCard";
import { RugScore } from "@/widgets/memecoin/RugScore";
import { SignalTracker } from "@/widgets/memecoin/SignalTracker";
import { SniperQueue } from "@/widgets/memecoin/SniperQueue";

import { AssetPageShell } from "./AssetPageShell";

/**
 * Memecoin gets its own dashboard surface — separate route, dedicated
 * widget set — per the operator's explicit directive ("memecoin should
 * have its own dashboard as planned") and PR-#2 spec §3.4 (copy +
 * normal + sniper trio).
 */
const SYMBOL = "BONK/SOL";

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
    i: "pair-card",
    x: 8,
    y: 0,
    w: 4,
    h: 5,
    minW: 3,
    minH: 4,
    render: () => <PairCard />,
  },
  {
    i: "rug",
    x: 8,
    y: 5,
    w: 4,
    h: 5,
    minW: 3,
    minH: 4,
    render: () => <RugScore />,
  },
  {
    i: "holders",
    x: 0,
    y: 10,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => <HolderConcentration />,
  },
  {
    i: "tape",
    x: 4,
    y: 10,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => <TimeAndSalesTape symbol={SYMBOL} />,
  },
  {
    i: "sniper-queue",
    x: 8,
    y: 10,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => <SniperQueue />,
  },
  {
    i: "copy-leaders",
    x: 0,
    y: 16,
    w: 6,
    h: 7,
    minW: 4,
    minH: 4,
    render: () => <CopyLeaderboard />,
  },
  {
    i: "signal-tracker",
    x: 6,
    y: 16,
    w: 6,
    h: 7,
    minW: 4,
    minH: 4,
    render: () => <SignalTracker />,
  },
  {
    i: "order-copy",
    x: 0,
    y: 23,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <OrderForm symbol={SYMBOL} />,
  },
  {
    i: "sltp-normal",
    x: 4,
    y: 23,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <SLTPBuilder form="memecoin-normal" />,
  },
  {
    i: "sltp-sniper",
    x: 8,
    y: 23,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <SLTPBuilder form="memecoin-sniper" />,
  },
  {
    i: "coherence",
    x: 0,
    y: 30,
    w: 6,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <CoherencePanel />,
  },
  {
    i: "alerts",
    x: 6,
    y: 30,
    w: 6,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <AlertsHub />,
  },
];

export function MemecoinPage() {
  return (
    <AssetPageShell
      title="Memecoin"
      asset="MEMECOIN"
      description="Dedicated memecoin surface — copy + signal + sniper trio per PR-#2 spec §3.4. Holder distribution, rug score, dev-dump watchdog, sniper bundles, copy leaders all share the same governed kill-switch + SL/TP engine as every other form."
    >
      <AssetGrid storageKey="memecoin" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
