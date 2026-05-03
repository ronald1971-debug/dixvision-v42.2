import { AssetGrid, type GridItemSpec } from "@/components/AssetGrid";
import { ChartPanel } from "@/widgets/ChartPanel";
import { ADXPanel } from "@/widgets/chart/ADXPanel";
import { ATRPanel } from "@/widgets/chart/ATRPanel";
import { ChartTypeSwitcher } from "@/widgets/chart/ChartTypeSwitcher";
import { DrawingToolsRail } from "@/widgets/chart/DrawingToolsRail";
import { MACDPanel } from "@/widgets/chart/MACDPanel";
import { RSIPanel } from "@/widgets/chart/RSIPanel";
import { StochasticPanel } from "@/widgets/chart/StochasticPanel";
import { VolumeProfile } from "@/widgets/chart/VolumeProfile";

const SYMBOL = "BTC/USDC";

const ITEMS: GridItemSpec[] = [
  {
    i: "tools",
    x: 0,
    y: 0,
    w: 2,
    h: 12,
    minW: 2,
    minH: 6,
    render: () => <DrawingToolsRail />,
  },
  {
    i: "chart",
    x: 2,
    y: 0,
    w: 7,
    h: 12,
    minW: 4,
    minH: 6,
    render: () => <ChartPanel symbol={SYMBOL} />,
  },
  {
    i: "vp",
    x: 9,
    y: 0,
    w: 3,
    h: 12,
    minW: 2,
    minH: 6,
    render: () => <VolumeProfile symbol={SYMBOL} />,
  },
  {
    i: "rsi",
    x: 0,
    y: 12,
    w: 3,
    h: 5,
    minW: 2,
    minH: 4,
    render: () => <RSIPanel symbol={SYMBOL} />,
  },
  {
    i: "macd",
    x: 3,
    y: 12,
    w: 3,
    h: 5,
    minW: 2,
    minH: 4,
    render: () => <MACDPanel symbol={SYMBOL} />,
  },
  {
    i: "stoch",
    x: 6,
    y: 12,
    w: 3,
    h: 5,
    minW: 2,
    minH: 4,
    render: () => <StochasticPanel symbol={SYMBOL} />,
  },
  {
    i: "atr",
    x: 9,
    y: 12,
    w: 3,
    h: 5,
    minW: 2,
    minH: 4,
    render: () => <ATRPanel symbol={SYMBOL} />,
  },
  {
    i: "adx",
    x: 0,
    y: 17,
    w: 6,
    h: 5,
    minW: 3,
    minH: 4,
    render: () => <ADXPanel symbol={SYMBOL} />,
  },
  {
    i: "type",
    x: 6,
    y: 17,
    w: 6,
    h: 5,
    minW: 3,
    minH: 4,
    render: () => <ChartTypeSwitcher />,
  },
];

export function ChartingPage() {
  return (
    <section className="flex h-full flex-col">
      <header className="mb-3">
        <h1 className="text-lg font-semibold tracking-tight">
          Charting{" "}
          <span className="ml-2 rounded border border-border bg-bg px-2 py-0.5 font-mono text-[11px] uppercase tracking-widest text-slate-400">
            {SYMBOL}
          </span>
        </h1>
        <p className="mt-1 text-xs text-slate-400">
          Indicator sub-panes (RSI · MACD · Stoch · ATR · ADX) + Volume
          Profile + drawing tools rail + chart-type switcher. The main
          ChartPanel keeps its EMA/VWAP overlays from the cockpit spec.
        </p>
      </header>
      <div className="flex-1 overflow-auto pb-6">
        <AssetGrid storageKey="charting-v1" defaultItems={ITEMS} />
      </div>
    </section>
  );
}
