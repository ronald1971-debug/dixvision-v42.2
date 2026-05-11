/**
 * Perps widget — liquidation heatmap.
 *
 * Bands of long / short open-interest projected onto their
 * mark-out liquidation prices, with the current price overlaid.
 * Polls ``/api/dashboard/perps/liquidations?symbol=<sym>`` every
 * 3 s; falls back to a deterministic skeleton when the route 404s.
 */
import { useQuery } from "@tanstack/react-query";

import { apiUrl } from "@/api/base";
import { WidgetStatusChip } from "@/components/WidgetStatusChip";

interface LiqBand {
  price: number;
  oi_long_usd: number;
  oi_short_usd: number;
}

interface LiqSnapshot {
  symbol: string;
  current_price: number;
  bands: LiqBand[];
  ts_iso: string;
}

function buildFallback(): LiqSnapshot {
  const current = 71_400;
  const bands: LiqBand[] = [];
  for (let i = -10; i <= 10; i++) {
    const price = current * (1 + i * 0.01);
    const dist = Math.abs(i);
    bands.push({
      price,
      oi_long_usd:
        i < 0 ? Math.max(0, 18_000_000 - dist * 1_500_000 + Math.cos(i) * 1e6) : 0,
      oi_short_usd:
        i > 0 ? Math.max(0, 16_000_000 - dist * 1_400_000 + Math.sin(i) * 1e6) : 0,
    });
  }
  return {
    symbol: "BTC-PERP",
    current_price: current,
    bands,
    ts_iso: new Date().toISOString(),
  };
}

const FALLBACK = buildFallback();

async function fetchLiqs(
  symbol: string,
  signal?: AbortSignal,
): Promise<{ snap: LiqSnapshot; live: boolean }> {
  try {
    const res = await fetch(
      apiUrl(
        `/api/dashboard/perps/liquidations?symbol=${encodeURIComponent(symbol)}`,
      ),
      { signal },
    );
    if (!res.ok) throw new Error(`status ${res.status}`);
    return { snap: (await res.json()) as LiqSnapshot, live: true };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    return { snap: { ...FALLBACK, symbol }, live: false };
  }
}

export function LiquidationMap({ symbol = "BTC-PERP" }: { symbol?: string }) {
  const { data } = useQuery({
    queryKey: ["dashboard", "perps", "liquidations", symbol],
    queryFn: ({ signal }) => fetchLiqs(symbol, signal),
    refetchInterval: 3_000,
    initialData: { snap: { ...FALLBACK, symbol }, live: false },
  });
  const { snap, live } = data;
  const maxOi = Math.max(
    1,
    ...snap.bands.map((b) => Math.max(b.oi_long_usd, b.oi_short_usd)),
  );
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Liquidation Heatmap · {snap.symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            open-interest bands · projected liq prices
          </p>
        </div>
        <WidgetStatusChip mode={live ? "live" : "mock"} />
      </header>
      <div className="flex-1 overflow-auto p-2 font-mono text-[10px]">
        <ul className="space-y-px">
          {[...snap.bands]
            .sort((a, b) => b.price - a.price)
            .map((b) => {
              const isCurrent =
                Math.abs(b.price - snap.current_price) <
                snap.current_price * 0.005;
              const longPct = (b.oi_long_usd / maxOi) * 100;
              const shortPct = (b.oi_short_usd / maxOi) * 100;
              return (
                <li
                  key={b.price}
                  className={`flex items-center gap-1 ${
                    isCurrent ? "font-bold text-slate-100" : "text-slate-400"
                  }`}
                >
                  <span className="w-16 shrink-0 tabular-nums">
                    {b.price.toFixed(0)}
                  </span>
                  <div className="relative flex h-3 flex-1">
                    <div
                      className="absolute right-1/2 h-full bg-emerald-500/50"
                      style={{ width: `${longPct / 2}%` }}
                    />
                    <div
                      className="absolute left-1/2 h-full bg-rose-500/50"
                      style={{ width: `${shortPct / 2}%` }}
                    />
                    {isCurrent && (
                      <div className="absolute left-1/2 h-full w-px -translate-x-1/2 bg-slate-100" />
                    )}
                  </div>
                </li>
              );
            })}
        </ul>
      </div>
      <footer className="flex items-baseline justify-between border-t border-border px-3 py-1 text-[10px] text-slate-500">
        <span>longs</span>
        <span className="text-slate-300">
          mark {snap.current_price.toFixed(0)}
        </span>
        <span>shorts</span>
      </footer>
    </div>
  );
}
