import { useEffect, useRef, useState } from "react";

import { useEventStream, useStreamState } from "@/state/realtime";

interface DepthLevel {
  price: number;
  size: number;
}
interface DepthSnapshot {
  bids: DepthLevel[];
  asks: DepthLevel[];
  mid: number;
}

/**
 * Tier-2 order-flow widget — Bookmap-style liquidity heatmap.
 *
 * Renders time × price × resting-size as a canvas heatmap. Each
 * column is a depth snapshot; intensity per cell is normalised to
 * the running max. Bid liquidity tints emerald, ask liquidity rose.
 * Mid-price is overlaid as a white line.
 *
 * Data: subscribes to the canonical SSE `depth` channel via
 * `useEventStream`. When the SSE bridge is in mock mode the heatmap
 * still draws so the structure is visible end-to-end.
 */
const HISTORY = 240; // columns
const PRICE_BUCKETS = 80; // rows

export function LiquidityHeatmap({ symbol = "BTC-USDT" }: { symbol?: string }) {
  const snapshots = useEventStream<DepthSnapshot>("depth", [], HISTORY);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{ price: number; size: number } | null>(
    null,
  );
  const stream = useStreamState();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || snapshots.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    draw(ctx, canvas.width, canvas.height, snapshots);
  }, [snapshots]);

  const last = snapshots[snapshots.length - 1];

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Liquidity heatmap · {symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            time × price × resting size · last {snapshots.length}/{HISTORY}{" "}
            snapshots
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${
            stream === "live"
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
              : stream === "mock"
                ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
                : "border-slate-600/40 bg-slate-800/40 text-slate-400"
          }`}
        >
          {stream}
        </span>
      </header>
      <div className="relative flex-1">
        <canvas
          ref={canvasRef}
          width={HISTORY * 4}
          height={PRICE_BUCKETS * 5}
          className="h-full w-full"
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const yPct = 1 - (e.clientY - rect.top) / rect.height;
            const last = snapshots[snapshots.length - 1];
            if (!last) return;
            const allLvls = [...last.bids, ...last.asks];
            if (allLvls.length === 0) return;
            const min = Math.min(...allLvls.map((l) => l.price));
            const max = Math.max(...allLvls.map((l) => l.price));
            const price = min + (max - min) * yPct;
            const closest = allLvls.reduce((a, b) =>
              Math.abs(a.price - price) < Math.abs(b.price - price) ? a : b,
            );
            setHover({ price: closest.price, size: closest.size });
          }}
          onMouseLeave={() => setHover(null)}
        />
        {hover && (
          <div className="pointer-events-none absolute right-2 top-2 rounded border border-border bg-bg/90 px-2 py-1 font-mono text-[10px] text-slate-300">
            {hover.price.toFixed(4)} · {hover.size}
          </div>
        )}
      </div>
      <footer className="flex items-baseline justify-between border-t border-border px-3 py-1 font-mono text-[11px] text-slate-400">
        <span>mid {last?.mid?.toFixed(4) ?? "—"}</span>
        <span className="text-slate-500">columns drift left over time</span>
      </footer>
    </section>
  );
}

function draw(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  snapshots: DepthSnapshot[],
): void {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0b0f1a";
  ctx.fillRect(0, 0, w, h);

  // global price min/max across the visible window
  let pmin = Infinity;
  let pmax = -Infinity;
  let smax = 0;
  for (const s of snapshots) {
    for (const l of [...s.bids, ...s.asks]) {
      if (l.price < pmin) pmin = l.price;
      if (l.price > pmax) pmax = l.price;
      if (l.size > smax) smax = l.size;
    }
  }
  if (!isFinite(pmin) || pmax <= pmin || smax === 0) return;
  const span = pmax - pmin;
  const colW = w / HISTORY;
  const cellH = h / PRICE_BUCKETS;

  for (let i = 0; i < snapshots.length; i++) {
    const s = snapshots[i];
    const x = (HISTORY - snapshots.length + i) * colW;
    for (const lvl of s.bids) {
      const y = h - ((lvl.price - pmin) / span) * h;
      const intensity = Math.min(1, lvl.size / smax);
      ctx.fillStyle = `rgba(61, 220, 132, ${intensity})`;
      ctx.fillRect(x, y - cellH / 2, Math.max(1, colW), Math.max(1, cellH));
    }
    for (const lvl of s.asks) {
      const y = h - ((lvl.price - pmin) / span) * h;
      const intensity = Math.min(1, lvl.size / smax);
      ctx.fillStyle = `rgba(255, 90, 90, ${intensity})`;
      ctx.fillRect(x, y - cellH / 2, Math.max(1, colW), Math.max(1, cellH));
    }
  }

  // mid-price line
  ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < snapshots.length; i++) {
    const s = snapshots[i];
    const x = (HISTORY - snapshots.length + i) * colW + colW / 2;
    const y = h - ((s.mid - pmin) / span) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
