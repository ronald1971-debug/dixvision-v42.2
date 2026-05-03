import { useEffect, useMemo, useRef } from "react";

import { useEventStream } from "@/state/realtime";

interface Trade {
  side: "BUY" | "SELL" | string;
  price: number;
  size: number;
}

/**
 * Tier-2 order-flow widget — Cumulative Volume Delta.
 *
 * Maintains a running Σ(buy_size − sell_size) over the most recent
 * N ticks and renders the line as an area chart. Direction-of-travel
 * (slope of the last ~5% of points) is shown as a chip in the header.
 */
const SAMPLES = 600;

export function CVDChart() {
  const trades = useEventStream<Trade>("ticks", [], SAMPLES);
  const series = useMemo(() => {
    let cum = 0;
    return trades.map((t) => {
      cum += String(t.side).toUpperCase() === "BUY" ? t.size : -t.size;
      return cum;
    });
  }, [trades]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawCvd(ctx, canvas.width, canvas.height, series);
  }, [series]);

  const direction = useMemo(() => {
    if (series.length < 10) return "flat";
    const tail = Math.max(10, Math.floor(series.length * 0.05));
    const slope = series[series.length - 1] - series[series.length - tail];
    if (slope > 0) return "rising";
    if (slope < 0) return "falling";
    return "flat";
  }, [series]);

  const last = series[series.length - 1] ?? 0;

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Cumulative Volume Delta
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Σ(buy − sell) · last {series.length}/{SAMPLES} ticks
          </p>
        </div>
        <div className="flex items-baseline gap-2 font-mono text-[10px]">
          <span
            className={`rounded border px-1.5 py-0.5 ${
              direction === "rising"
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                : direction === "falling"
                  ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
                  : "border-slate-600/40 bg-slate-800/40 text-slate-400"
            }`}
          >
            {direction}
          </span>
          <span
            className={
              last >= 0 ? "text-emerald-300" : "text-rose-300"
            }
          >
            {last >= 0 ? "+" : ""}
            {last}
          </span>
        </div>
      </header>
      <div className="flex-1">
        <canvas
          ref={canvasRef}
          width={SAMPLES * 2}
          height={200}
          className="h-full w-full"
        />
      </div>
    </section>
  );
}

function drawCvd(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  series: number[],
): void {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0b0f1a";
  ctx.fillRect(0, 0, w, h);
  if (series.length < 2) return;

  const min = Math.min(0, ...series);
  const max = Math.max(0, ...series);
  const span = Math.max(1, max - min);
  const zeroY = h - ((0 - min) / span) * h;

  // zero baseline
  ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  ctx.lineTo(w, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);

  const stepX = w / Math.max(1, series.length - 1);

  // filled area
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  for (let i = 0; i < series.length; i++) {
    const y = h - ((series[i] - min) / span) * h;
    ctx.lineTo(i * stepX, y);
  }
  ctx.lineTo((series.length - 1) * stepX, zeroY);
  ctx.closePath();
  const last = series[series.length - 1];
  ctx.fillStyle =
    last >= 0 ? "rgba(61, 220, 132, 0.18)" : "rgba(255, 90, 90, 0.18)";
  ctx.fill();

  // line
  ctx.strokeStyle = last >= 0 ? "#3ddc84" : "#ff5a5a";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < series.length; i++) {
    const y = h - ((series[i] - min) / span) * h;
    if (i === 0) ctx.moveTo(0, y);
    else ctx.lineTo(i * stepX, y);
  }
  ctx.stroke();
}
