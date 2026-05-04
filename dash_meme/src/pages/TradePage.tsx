import { useEffect, useRef, useState } from "react";

import { fetchPumpFunRecent } from "@/api/feeds";
import { useQuery } from "@tanstack/react-query";

import { Panel } from "@/components/Panel";
import { PriceChart, type PricePoint } from "@/components/PriceChart";
import { StatusPill } from "@/components/StatusPill";
import { TradeForm } from "@/components/TradeForm";
import { TxFeed } from "@/components/TxFeed";
import { useAutonomy } from "@/state/autonomy";
import { useSelectedPair } from "@/state/pair";

const AUTONOMY_BLURB: Record<string, string> = {
  manual:
    "Every intent requires a one-shot operator confirmation at the dashboard. Governance treats it as PAPER+ semantics with explicit operator authorization.",
  "semi-auto":
    "Operator pre-authorises within risk caps. Intents above caps fall back to manual approval at Governance.",
  "full-auto":
    "AUTO mode (per drift oracle). Operator attention relaxed within drift bounds; promotion gates still hash-anchored.",
};

function pickPrice(o: Record<string, unknown>): number | null {
  for (const k of ["price", "price_usd", "last"]) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}

function pickTs(o: Record<string, unknown>): number {
  for (const k of ["ts", "time", "timestamp"]) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) {
      return v > 1e12 ? v : v * 1000;
    }
  }
  return Date.now();
}

export function TradePage() {
  const [pair] = useSelectedPair();
  const [autonomy] = useAutonomy();
  const buf = useRef<PricePoint[]>([]);
  const [points, setPoints] = useState<PricePoint[]>([]);
  const q = useQuery({
    queryKey: ["pumpfun", "recent"],
    queryFn: fetchPumpFunRecent,
    refetchInterval: 2_000,
  });

  useEffect(() => {
    const incoming: PricePoint[] = [];
    for (const r of q.data?.recent ?? []) {
      const p = pickPrice(r);
      if (p != null) incoming.push({ ts: pickTs(r), price: p });
    }
    if (incoming.length === 0) return;
    buf.current = [...buf.current, ...incoming]
      .sort((a, b) => a.ts - b.ts)
      .slice(-600);
    setPoints([...buf.current]);
  }, [q.data]);

  return (
    <div className="grid h-full min-h-0 grid-cols-12 grid-rows-12 gap-2 p-2">
      <div className="col-span-9 row-span-8 min-h-0">
        <Panel
          title={`Trade · ${pair.symbol} · ${pair.chain}`}
          right={<StatusPill tone="info">{autonomy}</StatusPill>}
          bodyClassName="p-2"
        >
          <PriceChart points={points} height={400} />
          <div className="mt-2 rounded border border-hairline bg-surface-raised p-2 text-[11px] text-text-secondary">
            <span className="font-semibold text-text-primary">
              {autonomy.toUpperCase()}
            </span>{" "}
            — {AUTONOMY_BLURB[autonomy]}
          </div>
        </Panel>
      </div>
      <div className="col-span-3 row-span-12 min-h-0">
        <Panel title="Order entry" bodyClassName="p-0">
          <TradeForm symbol={pair.symbol} chain={pair.chain} />
        </Panel>
      </div>
      <div className="col-span-9 row-span-4 min-h-0">
        <TxFeed />
      </div>
    </div>
  );
}
