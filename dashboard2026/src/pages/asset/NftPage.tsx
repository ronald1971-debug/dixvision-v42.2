import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { OrderForm } from "@/widgets/OrderForm";
import { PositionsPanel } from "@/widgets/PositionsPanel";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";
import { BidLadder } from "@/widgets/nft/BidLadder";
import { CollectionVolume } from "@/widgets/nft/CollectionVolume";
import { RarityLens } from "@/widgets/nft/RarityLens";
import { SweepCart } from "@/widgets/nft/SweepCart";
import { TraitFloorGrid } from "@/widgets/nft/TraitFloorGrid";

import { AssetPageShell } from "./AssetPageShell";

const SYMBOL = "PUDGY/ETH";
const COLLECTION = "Pudgy";

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
    render: () => <TraitFloorGrid collection={COLLECTION} />,
  },
  {
    i: "sweep",
    x: 0,
    y: 8,
    w: 4,
    h: 8,
    minW: 3,
    minH: 5,
    render: () => <SweepCart collection={COLLECTION} />,
  },
  {
    i: "bid-ladder",
    x: 4,
    y: 8,
    w: 4,
    h: 8,
    minW: 3,
    minH: 5,
    render: () => <BidLadder collection={COLLECTION} />,
  },
  {
    i: "rarity",
    x: 8,
    y: 8,
    w: 4,
    h: 8,
    minW: 3,
    minH: 5,
    render: () => <RarityLens collection={COLLECTION} />,
  },
  {
    i: "order",
    x: 0,
    y: 16,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <OrderForm symbol={SYMBOL} />,
  },
  {
    i: "positions",
    x: 4,
    y: 16,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <PositionsPanel />,
  },
  {
    i: "sltp",
    x: 8,
    y: 16,
    w: 4,
    h: 7,
    minW: 3,
    minH: 5,
    render: () => <SLTPBuilder form="nft" />,
  },
  {
    i: "volume",
    x: 0,
    y: 23,
    w: 12,
    h: 8,
    minW: 4,
    minH: 5,
    render: () => <CollectionVolume />,
  },
  {
    i: "coherence",
    x: 0,
    y: 31,
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
      description="Cross-marketplace NFT surface (Blur · OpenSea Pro · Magic Eden · Tensor). Pro pack: TraitFloorGrid · SweepCart · BidLadder · RarityLens · CollectionVolume."
    >
      <AssetGrid storageKey="nft" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
