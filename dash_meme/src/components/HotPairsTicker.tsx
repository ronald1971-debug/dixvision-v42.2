import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { fetchPumpFunRecent, fetchRaydiumRecent } from "@/api/feeds";

/**
 * DEXtools-style top scrolling ticker — concatenates the most recent
 * pump.fun launches and Raydium pool updates into a horizontally
 * scrolling tape. Real data, real refetch — no placeholders.
 */
type TickerRow = {
  symbol: string;
  source: "PUMP" | "RAY";
  raw: Record<string, unknown>;
};

function pickString(o: Record<string, unknown>, ...keys: string[]): string {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === "string" && v.length > 0) return v;
  }
  return "";
}

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

export function HotPairsTicker() {
  const pump = useQuery({
    queryKey: ["pumpfun", "recent"],
    queryFn: fetchPumpFunRecent,
    refetchInterval: 3_000,
  });
  const ray = useQuery({
    queryKey: ["raydium", "recent"],
    queryFn: fetchRaydiumRecent,
    refetchInterval: 5_000,
  });

  const rows: ReadonlyArray<TickerRow> = useMemo(() => {
    const out: TickerRow[] = [];
    if (pump.data?.items) {
      for (const r of pump.data.items.slice(0, 24)) {
        out.push({
          source: "PUMP",
          symbol: pickString(r, "symbol", "ticker", "name", "mint"),
          raw: r,
        });
      }
    }
    if (ray.data?.items) {
      for (const r of ray.data.items.slice(0, 24)) {
        out.push({
          source: "RAY",
          symbol: pickString(r, "symbol", "pair", "name", "pool"),
          raw: r,
        });
      }
    }
    return out;
  }, [pump.data, ray.data]);

  if (rows.length === 0) {
    return (
      <div className="flex h-7 items-center border-b border-border bg-surface px-3 text-xs text-text-disabled">
        No live pairs yet — check `/api/feeds/pumpfun/start` and
        `/api/feeds/raydium/start` are running.
      </div>
    );
  }

  return (
    <div className="dex-scroll flex h-7 shrink-0 items-center gap-3 overflow-x-auto whitespace-nowrap border-b border-border bg-surface px-3 text-xs">
      {rows.map((row, idx) => {
        const price = pickNum(row.raw, "price", "price_usd", "last");
        const change = pickNum(
          row.raw,
          "change_24h",
          "price_change_pct",
          "delta_24h",
        );
        const tone =
          change == null
            ? "text-text-secondary"
            : change >= 0
              ? "text-ok"
              : "text-danger";
        return (
          <span key={`${row.source}-${idx}`} className="flex items-center gap-1">
            <span className="text-text-disabled">{row.source}</span>
            <span className="font-medium">{row.symbol || "?"}</span>
            {price != null && (
              <span className="font-mono text-text-secondary">
                ${price < 0.01 ? price.toExponential(2) : price.toFixed(4)}
              </span>
            )}
            {change != null && (
              <span className={`font-mono ${tone}`}>
                {change >= 0 ? "+" : ""}
                {change.toFixed(2)}%
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}
