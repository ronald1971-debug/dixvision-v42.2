import { useQuery } from "@tanstack/react-query";
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  createChart,
  createTextWatermark,
  type CandlestickData,
  type HistogramData,
  type IChartApi,
  type ISeriesApi,
  type ITextWatermarkPluginApi,
  type LineData,
  type Time,
} from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";

import { fetchMode } from "@/api/dashboard";

/**
 * Main chart panel — TradingView Lightweight Charts v5 (PR-#2 spec
 * §2). Apache 2.0, ~35 KB gzipped, tree-shakable. Every per-asset
 * page mounts this widget.
 *
 * Layers rendered:
 *   - Candlestick series (primary)
 *   - Volume histogram (overlay, separate price scale)
 *   - EMA(20) line indicator
 *   - VWAP line indicator (running, derived from candles)
 *   - Mode watermark — `PAPER` / `SHADOW` / `CANARY` painted in
 *     bright diagonal text so the operator never confuses modes.
 *
 * Indicator math is implemented client-side for now (deterministic,
 * cheap on small candle counts). When the SSE bridge lands the
 * candles + indicator overlays will arrive pre-computed from the
 * canonical event bus.
 */
const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"] as const;
type Timeframe = (typeof TIMEFRAMES)[number];

const INDICATORS = ["EMA(20)", "VWAP", "RSI(14)", "MACD"] as const;
type Indicator = (typeof INDICATORS)[number];

interface Candle {
  time: number; // unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ChartPanelProps {
  symbol: string;
  /** Default candle timeframe; operator can re-pick from the toolbar. */
  defaultTimeframe?: Timeframe;
}

function generateMockCandles(
  symbol: string,
  timeframe: Timeframe,
  count = 240,
): Candle[] {
  // Deterministic seed from symbol + timeframe so the panel stays
  // stable across re-mounts during development. Replaced by the SSE
  // bridge once `wave-realtime` lands.
  let seed = 0;
  for (const ch of `${symbol}:${timeframe}`) {
    seed = (seed * 31 + ch.charCodeAt(0)) >>> 0;
  }
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0xffffffff;
  };
  const stepSec: Record<Timeframe, number> = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
  };
  const step = stepSec[timeframe];
  const now = Math.floor(Date.now() / 1000);
  const start = now - step * count;
  let price = 100 + (seed % 50);
  const out: Candle[] = [];
  for (let i = 0; i < count; i += 1) {
    const drift = (rand() - 0.5) * price * 0.01;
    const open = price;
    const close = Math.max(0.0001, open + drift);
    const high = Math.max(open, close) * (1 + rand() * 0.005);
    const low = Math.min(open, close) * (1 - rand() * 0.005);
    const volume = Math.round(1000 + rand() * 9000);
    out.push({
      time: start + i * step,
      open,
      high,
      low,
      close,
      volume,
    });
    price = close;
  }
  return out;
}

function ema(values: number[], period: number): number[] {
  if (values.length === 0) return [];
  const k = 2 / (period + 1);
  const out = new Array<number>(values.length);
  out[0] = values[0];
  for (let i = 1; i < values.length; i += 1) {
    out[i] = values[i] * k + out[i - 1] * (1 - k);
  }
  return out;
}

function vwap(candles: Candle[]): number[] {
  let pv = 0;
  let v = 0;
  return candles.map((c) => {
    const typical = (c.high + c.low + c.close) / 3;
    pv += typical * c.volume;
    v += c.volume;
    return v > 0 ? pv / v : c.close;
  });
}

export function ChartPanel({
  symbol,
  defaultTimeframe = "5m",
}: ChartPanelProps) {
  const [timeframe, setTimeframe] = useState<Timeframe>(defaultTimeframe);
  const [active, setActive] = useState<Set<Indicator>>(
    new Set<Indicator>(["EMA(20)", "VWAP"]),
  );
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const emaRef = useRef<ISeriesApi<"Line"> | null>(null);
  const vwapRef = useRef<ISeriesApi<"Line"> | null>(null);
  const watermarkRef = useRef<ITextWatermarkPluginApi<Time> | null>(null);

  const { data: modeData } = useQuery({
    queryKey: ["dashboard", "mode"],
    queryFn: ({ signal }) => fetchMode(signal),
    refetchInterval: 2_000,
  });
  const mode = modeData?.current_mode ?? "SAFE";

  const candles = useMemo(
    () => generateMockCandles(symbol, timeframe),
    [symbol, timeframe],
  );

  // Mount the chart once.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const chart = createChart(el, {
      layout: {
        background: { color: "#0b0d12" },
        textColor: "#94a3b8",
      },
      grid: {
        horzLines: { color: "#1f2330" },
        vertLines: { color: "#1f2330" },
      },
      timeScale: { borderColor: "#1f2330", timeVisible: true },
      rightPriceScale: { borderColor: "#1f2330" },
      autoSize: true,
      crosshair: { mode: 1 },
    });
    const firstPane = chart.panes()[0];
    if (firstPane) {
      watermarkRef.current = createTextWatermark<Time>(firstPane, {
        horzAlign: "center",
        vertAlign: "center",
        lines: [
          {
            text: mode,
            color: watermarkColor(mode),
            fontSize: 56,
          },
        ],
      });
    }
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#3ddc84",
      downColor: "#ff5a5a",
      borderUpColor: "#3ddc84",
      borderDownColor: "#ff5a5a",
      wickUpColor: "#3ddc84",
      wickDownColor: "#ff5a5a",
    });
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceScaleId: "volume",
      priceFormat: { type: "volume" },
      color: "#3aa0ff",
    });
    chart
      .priceScale("volume")
      .applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
    const emaSeries = chart.addSeries(LineSeries, {
      color: "#ffaa3b",
      lineWidth: 2,
    });
    const vwapSeries = chart.addSeries(LineSeries, {
      color: "#3aa0ff",
      lineWidth: 2,
    });

    chartRef.current = chart;
    candleRef.current = candleSeries;
    volumeRef.current = volumeSeries;
    emaRef.current = emaSeries;
    vwapRef.current = vwapSeries;

    return () => {
      watermarkRef.current?.detach();
      watermarkRef.current = null;
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
      emaRef.current = null;
      vwapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push data + watermark on every dependency change.
  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleRef.current;
    const volumeSeries = volumeRef.current;
    const emaSeries = emaRef.current;
    const vwapSeries = vwapRef.current;
    if (!chart || !candleSeries || !volumeSeries || !emaSeries || !vwapSeries) {
      return;
    }
    watermarkRef.current?.applyOptions({
      lines: [
        {
          text: mode,
          color: watermarkColor(mode),
          fontSize: 56,
        },
      ],
    });
    const candleData: CandlestickData<Time>[] = candles.map((c) => ({
      time: c.time as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    const volumeData: HistogramData<Time>[] = candles.map((c) => ({
      time: c.time as Time,
      value: c.volume,
      color: c.close >= c.open ? "rgba(61,220,132,0.4)" : "rgba(255,90,90,0.4)",
    }));
    const closes = candles.map((c) => c.close);
    const emaValues = ema(closes, 20);
    const vwapValues = vwap(candles);
    const emaData: LineData<Time>[] = candles.map((c, i) => ({
      time: c.time as Time,
      value: emaValues[i],
    }));
    const vwapData: LineData<Time>[] = candles.map((c, i) => ({
      time: c.time as Time,
      value: vwapValues[i],
    }));
    candleSeries.setData(candleData);
    volumeSeries.setData(volumeData);
    emaSeries.setData(active.has("EMA(20)") ? emaData : []);
    vwapSeries.setData(active.has("VWAP") ? vwapData : []);
    chart.timeScale().fitContent();
  }, [candles, mode, active]);

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex items-baseline gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            {symbol}
          </h3>
          <span className="font-mono text-[10px] text-slate-500">
            lightweight-charts v5 · Apache-2.0 · mode watermark · {mode}
          </span>
        </div>
        <div className="flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              type="button"
              className={`rounded border px-1.5 py-0.5 ${
                tf === timeframe
                  ? "border-accent bg-accent text-bg"
                  : "border-border bg-bg text-slate-400 hover:text-accent"
              }`}
              onClick={() => setTimeframe(tf)}
              data-active={tf === timeframe ? "true" : "false"}
            >
              {tf}
            </button>
          ))}
        </div>
      </header>
      <div className="flex flex-wrap items-center gap-1 border-b border-border px-3 py-1 font-mono text-[10px] uppercase tracking-wider">
        <span className="text-slate-500">Indicators</span>
        {INDICATORS.map((ind) => (
          <button
            key={ind}
            type="button"
            className={`rounded border px-1.5 py-0.5 ${
              active.has(ind)
                ? "border-accent/60 bg-accent/15 text-accent"
                : "border-border bg-bg text-slate-500 hover:text-accent"
            }`}
            onClick={() => {
              setActive((prev) => {
                const next = new Set(prev);
                if (next.has(ind)) next.delete(ind);
                else next.add(ind);
                return next;
              });
            }}
            title={
              ind === "RSI(14)" || ind === "MACD"
                ? `${ind} pane lands with the SSE bridge — toggle is live, dedicated pane TBD`
                : `${ind} overlay`
            }
          >
            {ind}
          </button>
        ))}
      </div>
      <div ref={containerRef} className="flex-1" />
    </div>
  );
}

function watermarkColor(mode: string): string {
  switch (mode) {
    case "PAPER":
      return "rgba(58,160,255,0.08)";
    case "SHADOW":
      return "rgba(58,160,255,0.10)";
    case "CANARY":
      return "rgba(255,170,59,0.10)";
    case "LIVE":
      return "rgba(61,220,132,0.10)";
    case "AUTO":
      return "rgba(61,220,132,0.10)";
    case "LOCKED":
      return "rgba(255,90,90,0.12)";
    default:
      return "rgba(148,163,184,0.06)";
  }
}
