import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { ChartPanel } from "@/widgets/ChartPanel";
import { CoherencePanel } from "@/widgets/CoherencePanel";
import { DepthLadder } from "@/widgets/DepthLadder";
import { OrderForm } from "@/widgets/OrderForm";
import { PositionsPanel } from "@/widgets/PositionsPanel";
import { SLTPBuilder } from "@/widgets/SLTPBuilder";
import { CarryLadder } from "@/widgets/forex/CarryLadder";
import { CentralBankRates } from "@/widgets/forex/CentralBankRates";
import { CurrencyStrength } from "@/widgets/forex/CurrencyStrength";
import { EconomicCalendar } from "@/widgets/forex/EconomicCalendar";
import { PipCalc } from "@/widgets/forex/PipCalc";
import { SessionClock } from "@/widgets/forex/SessionClock";

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
    h: 5,
    minW: 3,
    minH: 4,
    render: () => <SessionClock />,
  },
  {
    i: "depth",
    x: 8,
    y: 5,
    w: 4,
    h: 5,
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
    render: () => <EconomicCalendar />,
  },
  {
    i: "cbrates",
    x: 6,
    y: 17,
    w: 6,
    h: 7,
    minW: 4,
    minH: 5,
    render: () => <CentralBankRates />,
  },
  {
    i: "carry",
    x: 0,
    y: 24,
    w: 4,
    h: 8,
    minW: 3,
    minH: 5,
    render: () => <CarryLadder />,
  },
  {
    i: "strength",
    x: 4,
    y: 24,
    w: 4,
    h: 8,
    minW: 3,
    minH: 5,
    render: () => <CurrencyStrength />,
  },
  {
    i: "pipcalc",
    x: 8,
    y: 24,
    w: 4,
    h: 8,
    minW: 3,
    minH: 6,
    render: () => <PipCalc />,
  },
  {
    i: "coherence",
    x: 0,
    y: 32,
    w: 12,
    h: 6,
    minW: 4,
    minH: 4,
    render: () => <CoherencePanel />,
  },
];

export function ForexPage() {
  return (
    <AssetPageShell
      title="Forex"
      asset="FOREX"
      description="Multi-broker FX surface (OANDA · IG · IBKR · MT4/MT5 bridge). Session-aware, calendar-gated. Pro pack: SessionClock · EconCal · CB rates · CarryLadder · PipCalc · CurrencyStrength."
    >
      <AssetGrid storageKey="forex" defaultItems={ITEMS} />
    </AssetPageShell>
  );
}
