import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { fetchPumpFunRecent, fetchRaydiumRecent } from "@/api/feeds";
import { HoldersPanel } from "@/components/HoldersPanel";
import { Panel } from "@/components/Panel";
import { PriceChart, type PricePoint } from "@/components/PriceChart";
import { RugScoreCard } from "@/components/RugScoreCard";
import { StatusPill } from "@/components/StatusPill";
import { TradeForm } from "@/components/TradeForm";
import { TxFeed } from "@/components/TxFeed";
import { useSelectedPair } from "@/state/pair";

function pickPrice(o: Record<string, unknown>): number | null {
  for (const k of ["price", "price_usd", "last", "p"]) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string") {
      const n = parseFloat(v);
      if (Number.isFinite(n)) return n;
    }
  }
  return null;
}

function pickTs(o: Record<string, unknown>): number {
  for (const k of ["ts", "time", "timestamp"]) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) {
      return v > 1e12 ? v : v * 1000;
    }
    if (typeof v === "string") {
      const d = new Date(v);
      if (!Number.isNaN(d.getTime())) return d.getTime();
    }
  }
  return Date.now();
}

export function PairExplorerPage() {
  const [pair] = useSelectedPair();
  const buf = useRef<PricePoint[]>([]);
  const [points, setPoints] = useState<PricePoint[]>([]);

  const pump = useQuery({
    queryKey: ["pumpfun", "recent"],
    queryFn: fetchPumpFunRecent,
    refetchInterval: 2_000,
  });
  const ray = useQuery({
    queryKey: ["raydium", "recent"],
    queryFn: fetchRaydiumRecent,
    refetchInterval: 5_000,
  });

  // Roll the price buffer from pump+ray feeds. Keep last 600 points.
  useEffect(() => {
    const incoming: PricePoint[] = [];
    for (const r of pump.data?.items ?? []) {
      const p = pickPrice(r);
      if (p != null) incoming.push({ ts: pickTs(r), price: p });
    }
    for (const r of ray.data?.items ?? []) {
      const p = pickPrice(r);
      if (p != null) incoming.push({ ts: pickTs(r), price: p });
    }
    if (incoming.length === 0) return;
    const merged = [...buf.current, ...incoming].sort((a, b) => a.ts - b.ts);
    const seen = new Set<number>();
    const dedup: PricePoint[] = [];
    for (const p of merged) {
      if (!seen.has(p.ts)) {
        seen.add(p.ts);
        dedup.push(p);
      }
    }
    buf.current = dedup.slice(-600);
    setPoints(buf.current);
  }, [pump.data, ray.data]);

  const lastPrice = points.length ? points[points.length - 1].price : null;

  return (
    <div className="grid h-full min-h-0 grid-cols-12 grid-rows-12 gap-2 p-2">
      <div className="col-span-9 row-span-7 min-h-0">
        <Panel
          title={`${pair.symbol} · ${pair.chain}`}
          right={
            <div className="flex items-center gap-2">
              {lastPrice != null && (
                <span className="font-mono text-text-primary">
                  ${lastPrice < 0.01 ? lastPrice.toExponential(3) : lastPrice.toFixed(6)}
                </span>
              )}
              <StatusPill tone={points.length ? "ok" : "neutral"}>
                {points.length} pts
              </StatusPill>
            </div>
          }
          bodyClassName="p-2"
        >
          <PriceChart points={points} height={320} />
        </Panel>
      </div>
      <div className="col-span-3 row-span-7 min-h-0">
        <RugScoreCard symbol={pair.symbol} />
      </div>
      <div className="col-span-3 row-span-5 min-h-0">
        <Panel title="Quick trade" bodyClassName="p-0">
          <TradeForm symbol={pair.symbol} chain={pair.chain} />
        </Panel>
      </div>
      <div className="col-span-5 row-span-5 min-h-0">
        <HoldersPanel />
      </div>
      <div className="col-span-4 row-span-5 min-h-0">
        <TxFeed />
      </div>
    </div>
  );
}
