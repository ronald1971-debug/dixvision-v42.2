import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { PlaceholderWidget } from "@/components/PlaceholderWidget";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { OrderForm } from "@/widgets/OrderForm";
import { PositionsPanel } from "@/widgets/PositionsPanel";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";

import { AssetPageShell } from "./AssetPageShell";

const SYMBOL = "PUDGY/ETH";

const ITEMS: GridItemSpec[] = [
  {
    i: "floor",
    x: 0,
    y: 0,
    w: 8,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <ChartPanel symbol={SYMBOL} />,
  },
  {
    i: "trait-grid",
    x: 8,
    y: 0,
    w: 4,
    h: 8,
    minW: 3,
    minH: 5,
    render: () => (
      <PlaceholderWidget
        title="Trait-Floor Grid"
        subtitle="rarity-aware floors per trait"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "sweep",
    x: 0,
    y: 8,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => (
      <PlaceholderWidget
        title="Sweep Cart"
        subtitle="trait filters · multi-collection sweep"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "bid-ladder",
    x: 4,
    y: 8,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => (
      <PlaceholderWidget
        title="Collection-Bid Ladder"
        subtitle="bid at floor · floor-1% · floor-2% (Blur-style)"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "rarity",
    x: 8,
    y: 8,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    render: () => (
      <PlaceholderWidget
        title="Rarity Lens"
        subtitle="floors stratified by rarity band"
        badge="DASH-K"
        status="stub"
      />
    ),
  },
  {
    i: "order",
    x: 0,
    y: 14,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <OrderForm symbol={SYMBOL} />,
  },
  {
    i: "positions",
    x: 4,
    y: 14,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <PositionsPanel />,
  },
  {
    i: "sltp",
    x: 8,
    y: 14,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <SLTPBuilder form="nft" />,
  },
  {
    i: "coherence",
    x: 0,
    y: 21,
    w: 12,
    h: 7,
    minW: 4,
    minH: 5,
    render: () => <CoherencePanel />,
  },
];

export function NftPage() {
  return (
    <AssetPageShell
      title="NFT"
      asset="NFT"
      description="Cross-marketplace NFT surface (Blur · OpenSea Pro · Magic Eden · Tensor). Trait-aware floors, rarity bands, sweep cart. Default widgets per PR-#2 spec §3.7."
    >
      <AssetGrid storageKey="nft" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
