import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { fetchPumpFunRecent, fetchRaydiumRecent } from "@/api/feeds";
import { Panel } from "@/components/Panel";
import { StatusPill } from "@/components/StatusPill";

type Swap = {
  ts: number;
  symbol: string;
  side: "BUY" | "SELL" | "NEW";
  notional: number | null;
  price: number | null;
  source: "PUMP" | "RAY";
};

function pickNum(o: Record<string, unknown>, ...keys: string[]): number | null {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string") {
      const n = parseFloat(v);
      if (Number.isFinite(n)) return n;
    }
  }
  return null;
}

function pickStr(o: Record<string, unknown>, ...keys: string[]): string {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === "string" && v) return v;
  }
  return "";
}

function inferSide(o: Record<string, unknown>): "BUY" | "SELL" | "NEW" {
  const raw = (o.side ?? o.direction ?? o.kind ?? "").toString().toLowerCase();
  if (raw.includes("sell")) return "SELL";
  if (raw.includes("launch") || raw.includes("create") || raw.includes("new")) {
    return "NEW";
  }
  return "BUY";
}

export function BigSwapPage() {
  const [minNotional, setMinNotional] = useState(1_000);
  const pump = useQuery({
    queryKey: ["pumpfun", "recent"],
    queryFn: fetchPumpFunRecent,
    refetchInterval: 2_000,
  });
  const ray = useQuery({
    queryKey: ["raydium", "recent"],
    queryFn: fetchRaydiumRecent,
    refetchInterval: 4_000,
  });

  const swaps: ReadonlyArray<Swap> = useMemo(() => {
    const all: Swap[] = [];
    for (const r of pump.data?.recent ?? []) {
      all.push({
        ts: pickNum(r, "ts", "time", "timestamp") ?? Date.now(),
        symbol: pickStr(r, "symbol", "ticker", "name", "mint"),
        side: inferSide(r),
        notional: pickNum(r, "notional_usd", "size_usd", "amount_usd", "size"),
        price: pickNum(r, "price", "price_usd"),
        source: "PUMP",
      });
    }
    for (const r of ray.data?.recent ?? []) {
      all.push({
        ts: pickNum(r, "ts", "time", "timestamp") ?? Date.now(),
        symbol: pickStr(r, "symbol", "pair", "pool", "name"),
        side: inferSide(r),
        notional: pickNum(r, "notional_usd", "amount_usd", "vol_usd"),
        price: pickNum(r, "price", "price_usd"),
        source: "RAY",
      });
    }
    return all
      .filter((s) => s.notional == null || s.notional >= minNotional)
      .sort((a, b) => (b.notional ?? 0) - (a.notional ?? 0))
      .slice(0, 250);
  }, [pump.data, ray.data, minNotional]);

  return (
    <div className="h-full p-2">
      <Panel
        title="Big swaps"
        right={
          <div className="flex items-center gap-2 text-[10px]">
            <span className="text-text-secondary">min $</span>
            <input
              type="number"
              value={minNotional}
              onChange={(e) =>
                setMinNotional(Math.max(0, Number(e.target.value) || 0))
              }
              className="h-6 w-20 rounded border border-border bg-surface-raised px-1 text-right font-mono"
            />
            <StatusPill tone="info">{swaps.length}</StatusPill>
          </div>
        }
      >
        <table className="w-full font-mono text-[11px] tabular-nums">
          <thead className="sticky top-0 bg-surface text-text-secondary">
            <tr className="border-b border-hairline">
              <th className="px-2 py-1 text-left">Symbol</th>
              <th className="px-2 py-1 text-left">Side</th>
              <th className="px-2 py-1 text-right">Notional</th>
              <th className="px-2 py-1 text-right">Price</th>
              <th className="px-2 py-1 text-left">Src</th>
            </tr>
          </thead>
          <tbody>
            {swaps.length === 0 && (
              <tr>
                <td
                  colSpan={5}
                  className="px-2 py-3 text-center text-text-disabled"
                >
                  No swaps over ${minNotional.toLocaleString()} yet.
                </td>
              </tr>
            )}
            {swaps.map((s, i) => (
              <tr
                key={i}
                className={`dex-row ${
                  s.side === "BUY"
                    ? "dex-buy"
                    : s.side === "SELL"
                      ? "dex-sell"
                      : ""
                }`}
              >
                <td className="px-2 py-0.5 font-medium">{s.symbol || "?"}</td>
                <td className="px-2 py-0.5">{s.side}</td>
                <td className="px-2 py-0.5 text-right">
                  {s.notional == null ? "—" : `$${s.notional.toLocaleString()}`}
                </td>
                <td className="px-2 py-0.5 text-right">
                  {s.price == null
                    ? "—"
                    : s.price < 0.01
                      ? s.price.toExponential(2)
                      : s.price.toFixed(4)}
                </td>
                <td className="px-2 py-0.5 text-text-disabled">{s.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
