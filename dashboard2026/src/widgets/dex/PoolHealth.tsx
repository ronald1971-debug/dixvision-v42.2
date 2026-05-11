/**
 * DEX widget — pool-health snapshot.
 *
 * Liquidity, 24h volume, and LP-concentration metrics for the
 * primary AMM pool backing the symbol. Polls
 * ``/api/dashboard/dex/pool_health?symbol=<sym>`` every 5 s; falls
 * back to a deterministic skeleton when the route 404s.
 */
import { useQuery } from "@tanstack/react-query";

import { apiUrl } from "@/api/base";
import { WidgetStatusChip } from "@/components/WidgetStatusChip";

interface LPHolder {
  address_short: string;
  pct: number;
}

interface PoolHealthSnapshot {
  symbol: string;
  liquidity_usd: number;
  volume_24h_usd: number;
  lp_count: number;
  hhi: number; // Herfindahl on LP shares (0..1)
  top_holders: LPHolder[];
  ts_iso: string;
}

const FALLBACK: PoolHealthSnapshot = {
  symbol: "SOL/USDC",
  liquidity_usd: 12_400_000,
  volume_24h_usd: 38_700_000,
  lp_count: 218,
  hhi: 0.31,
  top_holders: [
    { address_short: "8KQ4…b21Y", pct: 22.4 },
    { address_short: "FvR2…c9pL", pct: 14.1 },
    { address_short: "9wT7…aE3X", pct: 8.6 },
  ],
  ts_iso: new Date().toISOString(),
};

async function fetchPoolHealth(
  symbol: string,
  signal?: AbortSignal,
): Promise<{ snap: PoolHealthSnapshot; live: boolean }> {
  try {
    const res = await fetch(
      apiUrl(
        `/api/dashboard/dex/pool_health?symbol=${encodeURIComponent(symbol)}`,
      ),
      { signal },
    );
    if (!res.ok) throw new Error(`status ${res.status}`);
    return { snap: (await res.json()) as PoolHealthSnapshot, live: true };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    return { snap: { ...FALLBACK, symbol }, live: false };
  }
}

export function PoolHealth({ symbol = "SOL/USDC" }: { symbol?: string }) {
  const { data } = useQuery({
    queryKey: ["dashboard", "dex", "pool_health", symbol],
    queryFn: ({ signal }) => fetchPoolHealth(symbol, signal),
    refetchInterval: 5_000,
    initialData: { snap: { ...FALLBACK, symbol }, live: false },
  });
  const { snap, live } = data;
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Pool Health · {snap.symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            liquidity · 24h volume · LP concentration
          </p>
        </div>
        <WidgetStatusChip mode={live ? "live" : "mock"} />
      </header>
      <div className="flex-1 overflow-auto p-3 text-[12px]">
        <div className="grid grid-cols-2 gap-2 font-mono">
          <KV k="liquidity" v={fmtUsd(snap.liquidity_usd)} />
          <KV k="24h vol" v={fmtUsd(snap.volume_24h_usd)} />
          <KV k="LPs" v={String(snap.lp_count)} />
          <KV k="HHI" v={snap.hhi.toFixed(2)} />
        </div>
        <h4 className="mt-3 mb-1 font-mono text-[10px] uppercase tracking-wider text-slate-500">
          Top LPs
        </h4>
        <ul className="space-y-0.5 font-mono text-[11px]">
          {snap.top_holders.map((h) => (
            <li
              key={h.address_short}
              className="flex items-center justify-between text-slate-300"
            >
              <span>{h.address_short}</span>
              <span className="text-slate-400">{h.pct.toFixed(1)}%</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between rounded border border-border bg-bg/40 px-2 py-1 text-[11px]">
      <span className="text-slate-500">{k}</span>
      <span className="text-slate-200">{v}</span>
    </div>
  );
}

function fmtUsd(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}k`;
  return `$${n.toFixed(0)}`;
}
