import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { fetchPumpFunRecent, fetchRaydiumRecent } from "@/api/feeds";
import { Panel } from "@/components/Panel";
import { StatusPill } from "@/components/StatusPill";
import { navigate } from "@/router";
import { useSelectedPair } from "@/state/pair";

type Row = {
  symbol: string;
  price: number | null;
  change: number | null;
  vol: number | null;
};

function pickStr(o: Record<string, unknown>, ...keys: string[]): string {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === "string" && v) return v;
  }
  return "";
}
function pickNum(o: Record<string, unknown>, ...keys: string[]): number | null {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}

export function StatsPage() {
  const [, setPair] = useSelectedPair();
  const pump = useQuery({
    queryKey: ["pumpfun", "recent"],
    queryFn: fetchPumpFunRecent,
    refetchInterval: 4_000,
  });
  const ray = useQuery({
    queryKey: ["raydium", "recent"],
    queryFn: fetchRaydiumRecent,
    refetchInterval: 6_000,
  });

  const rows: ReadonlyArray<Row> = useMemo(() => {
    const all: Row[] = [];
    for (const r of pump.data?.recent ?? []) {
      all.push({
        symbol: pickStr(r, "symbol", "ticker", "name", "mint"),
        price: pickNum(r, "price", "price_usd"),
        change: pickNum(r, "change_24h", "price_change_pct"),
        vol: pickNum(r, "volume_24h", "vol", "volume"),
      });
    }
    for (const r of ray.data?.recent ?? []) {
      all.push({
        symbol: pickStr(r, "symbol", "pair", "pool"),
        price: pickNum(r, "price", "price_usd"),
        change: pickNum(r, "change_24h", "price_change_pct"),
        vol: pickNum(r, "volume_24h", "vol_24h"),
      });
    }
    return all;
  }, [pump.data, ray.data]);

  const gainers = [...rows]
    .filter((r) => r.change != null)
    .sort((a, b) => (b.change ?? 0) - (a.change ?? 0))
    .slice(0, 25);
  const losers = [...rows]
    .filter((r) => r.change != null)
    .sort((a, b) => (a.change ?? 0) - (b.change ?? 0))
    .slice(0, 25);
  const hot = [...rows]
    .filter((r) => r.vol != null)
    .sort((a, b) => (b.vol ?? 0) - (a.vol ?? 0))
    .slice(0, 25);

  return (
    <div className="grid h-full grid-cols-3 gap-2 p-2">
      <BoardPanel title="Top gainers" tone="ok" rows={gainers} setPair={setPair} />
      <BoardPanel title="Top losers" tone="danger" rows={losers} setPair={setPair} />
      <BoardPanel title="Hot (volume)" tone="info" rows={hot} setPair={setPair} />
    </div>
  );
}

function BoardPanel({
  title,
  tone,
  rows,
  setPair,
}: {
  title: string;
  tone: "ok" | "danger" | "info";
  rows: ReadonlyArray<Row>;
  setPair: (p: { symbol: string; chain: string }) => void;
}) {
  return (
    <Panel
      title={title}
      right={<StatusPill tone={tone}>{rows.length}</StatusPill>}
    >
      <table className="w-full font-mono text-[11px] tabular-nums">
        <thead className="sticky top-0 bg-surface text-text-secondary">
          <tr className="border-b border-hairline">
            <th className="px-2 py-1 text-left">Symbol</th>
            <th className="px-2 py-1 text-right">Price</th>
            <th className="px-2 py-1 text-right">24h%</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td
                colSpan={3}
                className="px-2 py-3 text-center text-text-disabled"
              >
                No data.
              </td>
            </tr>
          )}
          {rows.map((r, i) => {
            const tone =
              r.change == null
                ? "text-text-secondary"
                : r.change >= 0
                  ? "text-ok"
                  : "text-danger";
            return (
              <tr
                key={r.symbol + i}
                className="dex-row cursor-pointer"
                onClick={() => {
                  setPair({ symbol: r.symbol, chain: "solana" });
                  navigate("explorer");
                }}
              >
                <td className="px-2 py-0.5">{r.symbol || "?"}</td>
                <td className="px-2 py-0.5 text-right">
                  {r.price == null
                    ? "—"
                    : r.price < 0.01
                      ? r.price.toExponential(2)
                      : r.price.toFixed(4)}
                </td>
                <td className={`px-2 py-0.5 text-right ${tone}`}>
                  {r.change == null
                    ? "—"
                    : `${r.change >= 0 ? "+" : ""}${r.change.toFixed(2)}%`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Panel>
  );
}
