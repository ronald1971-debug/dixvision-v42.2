import {
  CandlestickSeries,
  createChart,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef } from "react";

/**
 * Lightweight wrapper around lightweight-charts. We don't have an OHLC
 * endpoint yet for memecoin pairs (Pump.fun streams trades, not bars)
 * so the chart synthesises a 1-minute candle series from a rolling
 * trade buffer. When the OHLC endpoint lands, swap `points` for the
 * real series — the surface stays identical.
 */
export type PricePoint = {
  ts: number; // ms epoch
  price: number;
};

export function PriceChart({
  points,
  height = 280,
}: {
  points: ReadonlyArray<PricePoint>;
  height?: number;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const chart = createChart(el, {
      layout: {
        background: { color: "rgba(0,0,0,0)" },
        textColor: "#9aa4b2",
        fontFamily: "JetBrains Mono, monospace",
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
      timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true },
      autoSize: true,
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#ef4444",
      borderUpColor: "#10b981",
      borderDownColor: "#ef4444",
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
    });
    chartRef.current = chart;
    seriesRef.current = series;
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    const bars = aggregateOneMinuteBars(points);
    series.setData(bars);
    chartRef.current?.timeScale().fitContent();
  }, [points]);

  return <div ref={containerRef} style={{ height }} className="w-full" />;
}

function aggregateOneMinuteBars(
  points: ReadonlyArray<PricePoint>,
): CandlestickData<UTCTimestamp>[] {
  if (points.length === 0) return [];
  const buckets = new Map<
    number,
    { o: number; h: number; l: number; c: number }
  >();
  for (const p of points) {
    const bucket = Math.floor(p.ts / 60_000) * 60_000;
    const existing = buckets.get(bucket);
    if (!existing) {
      buckets.set(bucket, { o: p.price, h: p.price, l: p.price, c: p.price });
    } else {
      existing.h = Math.max(existing.h, p.price);
      existing.l = Math.min(existing.l, p.price);
      existing.c = p.price;
    }
  }
  const out: CandlestickData<UTCTimestamp>[] = [];
  const keys = Array.from(buckets.keys()).sort((a, b) => a - b);
  for (const k of keys) {
    const b = buckets.get(k);
    if (!b) continue;
    out.push({
      time: Math.floor(k / 1000) as UTCTimestamp,
      open: b.o,
      high: b.h,
      low: b.l,
      close: b.c,
    });
  }
  return out;
}
