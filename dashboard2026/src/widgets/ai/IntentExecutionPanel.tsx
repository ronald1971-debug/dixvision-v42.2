import { useState } from "react";

/**
 * Tier-3 / E-track AI widget — Intent execution router.
 *
 * Surfaces an intent-based execution panel comparing UniswapX, CowSwap
 * and Across quotes for the same swap intent. Mirrors the 2026 trader
 * stack (Haiku / UniswapX / CowSwap / Across) the gap report flagged
 * as table stakes.
 *
 * Backend hooks (D-track):
 *   - UniswapX is wired in PR #155 via
 *     ``intelligence_engine.execution.uniswapx``.
 *   - CowSwap + Across will land in a follow-up adapter PR; this UI
 *     ships first so the surface is ready when the adapter arrives.
 *
 * Operator approval edge (INV-72) gates the actual sign+broadcast —
 * the "stage" button does not place a real order.
 */
interface RouterQuote {
  router: "UniswapX" | "CowSwap" | "Across";
  amount_out: number;
  est_gas_usd: number;
  fill_p: number;
  ttl_s: number;
  notes: string;
}

const QUOTES: RouterQuote[] = [
  {
    router: "UniswapX",
    amount_out: 1_004.21,
    est_gas_usd: 0,
    fill_p: 0.94,
    ttl_s: 30,
    notes: "Dutch auction · filler-paid gas · MEV-protected",
  },
  {
    router: "CowSwap",
    amount_out: 1_003.87,
    est_gas_usd: 0,
    fill_p: 0.91,
    ttl_s: 60,
    notes: "Batch auction · CoW · CIP-38 settlement",
  },
  {
    router: "Across",
    amount_out: 1_001.4,
    est_gas_usd: 4.2,
    fill_p: 0.99,
    ttl_s: 12,
    notes: "Cross-chain bridge · relayer-paid · UMA dispute window",
  },
];

export function IntentExecutionPanel() {
  const [tokenIn, setTokenIn] = useState("ETH");
  const [tokenOut, setTokenOut] = useState("USDC");
  const [amountIn, setAmountIn] = useState("0.5");
  const [staged, setStaged] = useState<RouterQuote["router"] | null>(null);

  const best = QUOTES.reduce((acc, q) =>
    q.amount_out > acc.amount_out ? q : acc,
  );

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Intent execution router
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          UniswapX · CowSwap · Across — one intent, three quotes,
          operator-approval-gated
        </p>
      </header>
      <div className="border-b border-border bg-bg/40 px-3 py-2">
        <div className="grid grid-cols-3 gap-2 font-mono text-[11px] text-slate-300">
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              token in
            </span>
            <input
              value={tokenIn}
              onChange={(e) => setTokenIn(e.target.value.toUpperCase())}
              className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              token out
            </span>
            <input
              value={tokenOut}
              onChange={(e) => setTokenOut(e.target.value.toUpperCase())}
              className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              amount in
            </span>
            <input
              value={amountIn}
              onChange={(e) => setAmountIn(e.target.value)}
              className="rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
            />
          </label>
        </div>
      </div>
      <ul className="flex-1 divide-y divide-border/40 overflow-auto">
        {QUOTES.map((q) => {
          const isBest = q.router === best.router;
          const isStaged = staged === q.router;
          return (
            <li
              key={q.router}
              className={`grid grid-cols-[1fr_auto] gap-2 px-3 py-2 font-mono text-[11px] text-slate-300 ${
                isBest ? "bg-emerald-500/5" : ""
              }`}
            >
              <div className="min-w-0">
                <div className="flex items-baseline gap-2">
                  <span className="font-semibold text-slate-200">
                    {q.router}
                  </span>
                  {isBest && (
                    <span className="rounded border border-emerald-500/40 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-emerald-300">
                      best
                    </span>
                  )}
                  <span className="ml-auto text-emerald-400">
                    {q.amount_out.toLocaleString()} {tokenOut}
                  </span>
                </div>
                <div className="mt-0.5 flex flex-wrap items-baseline gap-3 text-[10px] text-slate-500">
                  <span>gas {q.est_gas_usd.toFixed(2)} USD</span>
                  <span>fill p {(q.fill_p * 100).toFixed(0)}%</span>
                  <span>ttl {q.ttl_s}s</span>
                </div>
                <div className="mt-1 truncate text-[10px] text-slate-500">
                  {q.notes}
                </div>
              </div>
              <button
                type="button"
                onClick={() => setStaged(isStaged ? null : q.router)}
                className={`self-center rounded border px-2 py-0.5 text-[10px] uppercase tracking-wider ${
                  isStaged
                    ? "border-accent/40 bg-accent/10 text-accent"
                    : "border-border bg-bg/40 text-slate-400 hover:border-accent hover:text-accent"
                }`}
              >
                {isStaged ? "staged" : "stage"}
              </button>
            </li>
          );
        })}
      </ul>
      {staged && (
        <footer className="border-t border-border bg-bg/40 px-3 py-2 font-mono text-[10px] text-slate-500">
          staged via <span className="text-slate-300">{staged}</span> · awaiting
          approval-edge gate (INV-72)
        </footer>
      )}
    </section>
  );
}
