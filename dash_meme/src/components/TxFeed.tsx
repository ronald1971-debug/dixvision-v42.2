import { useQuery } from "@tanstack/react-query";

import { fetchPumpFunRecent, fetchRaydiumRecent } from "@/api/feeds";

import { Panel } from "./Panel";

type TxRow = {
  ts: string;
  side: "BUY" | "SELL" | "NEW";
  symbol: string;
  amount: string;
  price: string;
  source: "PUMP" | "RAY";
};

function fmtTs(v: unknown): string {
  if (typeof v === "number" && Number.isFinite(v)) {
    const d = new Date(v > 1e12 ? v : v * 1000);
    return d.toLocaleTimeString();
  }
  if (typeof v === "string") {
    const d = new Date(v);
    if (!Number.isNaN(d.getTime())) return d.toLocaleTimeString();
    return v;
  }
  return "—";
}

function pickStr(o: Record<string, unknown>, ...keys: string[]): string {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === "string" && v) return v;
    if (typeof v === "number" && Number.isFinite(v)) return String(v);
  }
  return "—";
}

function inferSide(o: Record<string, unknown>): "BUY" | "SELL" | "NEW" {
  const raw = (o.side ?? o.direction ?? o.kind ?? "").toString().toLowerCase();
  if (raw.includes("buy")) return "BUY";
  if (raw.includes("sell")) return "SELL";
  if (raw.includes("launch") || raw.includes("create") || raw.includes("new")) {
    return "NEW";
  }
  return "BUY";
}

export function TxFeed({ height = "100%" }: { height?: string | number }) {
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

  const rows: TxRow[] = [];
  for (const r of pump.data?.recent ?? []) {
    rows.push({
      ts: fmtTs(r.ts ?? r.time ?? r.timestamp),
      side: inferSide(r),
      symbol: pickStr(r, "symbol", "ticker", "name", "mint"),
      amount: pickStr(r, "amount", "size", "qty", "sol_amount"),
      price: pickStr(r, "price", "price_usd", "last"),
      source: "PUMP",
    });
  }
  for (const r of ray.data?.recent ?? []) {
    rows.push({
      ts: fmtTs(r.ts ?? r.time ?? r.timestamp),
      side: inferSide(r),
      symbol: pickStr(r, "symbol", "pair", "name", "pool"),
      amount: pickStr(r, "amount", "size", "liq_usd"),
      price: pickStr(r, "price", "price_usd"),
      source: "RAY",
    });
  }

  return (
    <Panel title="Live transactions" bodyClassName="text-xs">
      <div className="h-full" style={{ height }}>
        <table className="w-full table-fixed font-mono text-[11px] tabular-nums">
          <thead className="sticky top-0 bg-surface text-text-secondary">
            <tr className="border-b border-hairline">
              <th className="w-20 px-2 py-1 text-left">Time</th>
              <th className="w-14 px-2 py-1 text-left">Side</th>
              <th className="px-2 py-1 text-left">Symbol</th>
              <th className="w-24 px-2 py-1 text-right">Amount</th>
              <th className="w-24 px-2 py-1 text-right">Price</th>
              <th className="w-12 px-2 py-1 text-left">Src</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td
                  colSpan={6}
                  className="px-2 py-3 text-center text-text-disabled"
                >
                  No live trades yet…
                </td>
              </tr>
            )}
            {rows.slice(0, 200).map((r, idx) => (
              <tr
                key={idx}
                className={`dex-row ${
                  r.side === "BUY"
                    ? "dex-buy"
                    : r.side === "SELL"
                      ? "dex-sell"
                      : ""
                }`}
              >
                <td className="px-2 py-0.5 text-text-secondary">{r.ts}</td>
                <td className="px-2 py-0.5 font-semibold">{r.side}</td>
                <td className="truncate px-2 py-0.5">{r.symbol}</td>
                <td className="px-2 py-0.5 text-right">{r.amount}</td>
                <td className="px-2 py-0.5 text-right">{r.price}</td>
                <td className="px-2 py-0.5 text-text-disabled">{r.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
