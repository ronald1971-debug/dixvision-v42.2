import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { fetchRaydiumRecent } from "@/api/feeds";
import { Panel } from "@/components/Panel";
import { StatusPill } from "@/components/StatusPill";
import { navigate } from "@/router";
import { useSelectedPair } from "@/state/pair";

type SortKey = "liq" | "vol" | "fee" | "age";

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

export function PoolExplorerPage() {
  const [, setPair] = useSelectedPair();
  const [sort, setSort] = useState<SortKey>("liq");
  const q = useQuery({
    queryKey: ["raydium", "recent"],
    queryFn: fetchRaydiumRecent,
    refetchInterval: 5_000,
  });

  const rows = useMemo(() => {
    const raw = q.data?.items ?? [];
    // Normalize age to a duration in seconds. ``age_seconds`` and
    // ``age`` are already durations (lower = newer); ``ts`` is a Unix
    // timestamp (higher = newer) so we convert it to ``now - ts``
    // before sorting (Devin Review BUG_0002 on PR #181).
    const nowSec = Date.now() / 1000;
    const ageSeconds = (r: Record<string, unknown>): number | null => {
      const direct = pickNum(r, "age_seconds", "age");
      if (direct !== null) return direct;
      const ts = pickNum(r, "ts");
      if (ts === null) return null;
      // Heuristic: if value is too large to be seconds, assume ms.
      const tsSec = ts > 1e12 ? ts / 1000 : ts;
      return Math.max(0, nowSec - tsSec);
    };
    const mapped = raw.map((r) => ({
      symbol: pickStr(r, "symbol", "pair", "name", "pool"),
      pool: pickStr(r, "pool", "address", "id"),
      liq: pickNum(r, "liq_usd", "liquidity_usd", "liquidity"),
      vol: pickNum(r, "volume_24h", "vol_24h", "volume"),
      fee: pickNum(r, "fee_pct", "fee", "fees_24h"),
      age: ageSeconds(r),
      raw: r,
    }));
    const cmpDesc = (a: number | null, b: number | null) =>
      (b ?? -Infinity) - (a ?? -Infinity);
    const cmpAsc = (a: number | null, b: number | null) =>
      (a ?? Infinity) - (b ?? Infinity);
    return [...mapped].sort((a, b) => {
      if (sort === "liq") return cmpDesc(a.liq, b.liq);
      if (sort === "vol") return cmpDesc(a.vol, b.vol);
      if (sort === "fee") return cmpDesc(a.fee, b.fee);
      return cmpAsc(a.age, b.age); // newest first (lowest age)
    });
  }, [q.data, sort]);

  return (
    <div className="h-full p-2">
      <Panel
        title="Raydium pool explorer"
        right={
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide">
            {(["liq", "vol", "fee", "age"] as SortKey[]).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setSort(k)}
                className={
                  sort === k
                    ? "text-accent"
                    : "text-text-secondary hover:text-text-primary"
                }
              >
                {k}
              </button>
            ))}
            <StatusPill tone={q.isFetching ? "info" : "neutral"}>
              {rows.length}
            </StatusPill>
          </div>
        }
      >
        <table className="w-full font-mono text-[11px] tabular-nums">
          <thead className="sticky top-0 bg-surface text-text-secondary">
            <tr className="border-b border-hairline">
              <th className="px-2 py-1 text-left">Pool</th>
              <th className="px-2 py-1 text-right">Liquidity</th>
              <th className="px-2 py-1 text-right">Volume 24h</th>
              <th className="px-2 py-1 text-right">Fee</th>
              <th className="px-2 py-1 text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td
                  colSpan={5}
                  className="px-2 py-3 text-center text-text-disabled"
                >
                  No Raydium pools — start{" "}
                  <span className="font-mono">/api/feeds/raydium/start</span>.
                </td>
              </tr>
            )}
            {rows.map((r, i) => (
              <tr key={r.pool + i} className="dex-row">
                <td className="px-2 py-0.5">
                  <div className="font-medium text-text-primary">
                    {r.symbol || "?"}
                  </div>
                  {r.pool && (
                    <div className="text-text-disabled">
                      {r.pool.slice(0, 6)}…{r.pool.slice(-4)}
                    </div>
                  )}
                </td>
                <td className="px-2 py-0.5 text-right">
                  {r.liq == null ? "—" : `$${r.liq.toLocaleString()}`}
                </td>
                <td className="px-2 py-0.5 text-right">
                  {r.vol == null ? "—" : `$${r.vol.toLocaleString()}`}
                </td>
                <td className="px-2 py-0.5 text-right">
                  {r.fee == null ? "—" : `${r.fee.toFixed(3)}`}
                </td>
                <td className="px-2 py-0.5 text-right">
                  <button
                    type="button"
                    onClick={() => {
                      setPair({
                        symbol: r.symbol || "?",
                        chain: "solana",
                        poolId: r.pool,
                      });
                      navigate("explorer");
                    }}
                    className="rounded border border-accent px-2 py-0.5 text-accent hover:bg-[var(--accent-soft)]"
                  >
                    open
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
