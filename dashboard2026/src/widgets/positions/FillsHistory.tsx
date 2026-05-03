/**
 * G-track widget — Fills history.
 *
 * Append-only ledger of executed child-fills with PnL tag.
 * Backend hook: ``GET /api/positions/fills?since=...`` reads from
 * the audit ledger projection (PR #64 DecisionTrace.children).
 */
interface Fill {
  id: string;
  ts: string;
  symbol: string;
  side: "BUY" | "SELL";
  qty: number;
  px: number;
  fee_bps: number;
  pnl_usd: number;
  venue: string;
}

const SEED: Fill[] = [
  { id: "f-12", ts: "16:42:18", symbol: "BTC-USDT", side: "SELL", qty: 0.25, px: 67_810, fee_bps: 1.0, pnl_usd: 432.10, venue: "binance" },
  { id: "f-11", ts: "16:38:54", symbol: "ETH-USDT", side: "BUY", qty: 1.2, px: 3_524, fee_bps: 1.0, pnl_usd: -22.40, venue: "binance" },
  { id: "f-10", ts: "16:30:12", symbol: "SOL-USDT", side: "SELL", qty: 50, px: 145.10, fee_bps: 1.2, pnl_usd: 142.50, venue: "kraken" },
  { id: "f-09", ts: "16:24:01", symbol: "BTC-USDT", side: "BUY", qty: 0.25, px: 67_640, fee_bps: 1.0, pnl_usd: 0, venue: "binance" },
  { id: "f-08", ts: "16:11:43", symbol: "AVAX-USDT", side: "BUY", qty: 100, px: 31.20, fee_bps: 1.2, pnl_usd: 0, venue: "okx" },
  { id: "f-07", ts: "15:58:29", symbol: "ETH-USDT", side: "SELL", qty: 2.5, px: 3_540, fee_bps: 1.0, pnl_usd: 318.00, venue: "binance" },
  { id: "f-06", ts: "15:42:11", symbol: "WIF-USDT", side: "SELL", qty: 12_500, px: 2.41, fee_bps: 4.0, pnl_usd: 1_212.50, venue: "binance" },
  { id: "f-05", ts: "15:30:00", symbol: "WIF-USDT", side: "BUY", qty: 12_500, px: 2.32, fee_bps: 4.0, pnl_usd: 0, venue: "binance" },
  { id: "f-04", ts: "15:14:22", symbol: "SOL-USDT", side: "BUY", qty: 50, px: 142.30, fee_bps: 1.2, pnl_usd: 0, venue: "kraken" },
  { id: "f-03", ts: "14:55:09", symbol: "MATIC-USDT", side: "BUY", qty: 5_000, px: 0.515, fee_bps: 1.5, pnl_usd: 0, venue: "binance" },
];

export function FillsHistory() {
  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Fills history
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          append-only · projected from audit ledger (PR #64)
        </p>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">time</th>
              <th className="px-3 py-1.5 text-left">symbol</th>
              <th className="px-3 py-1.5 text-left">side</th>
              <th className="px-3 py-1.5 text-right">qty</th>
              <th className="px-3 py-1.5 text-right">px</th>
              <th className="px-3 py-1.5 text-right">fee</th>
              <th className="px-3 py-1.5 text-right">pnl</th>
              <th className="px-3 py-1.5 text-left">venue</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {SEED.map((f) => (
              <tr key={f.id}>
                <td className="px-3 py-1 text-slate-500">{f.ts}</td>
                <td className="px-3 py-1 text-slate-200">{f.symbol}</td>
                <td
                  className={`px-3 py-1 ${
                    f.side === "BUY" ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {f.side}
                </td>
                <td className="px-3 py-1 text-right">
                  {f.qty.toLocaleString()}
                </td>
                <td className="px-3 py-1 text-right">
                  {f.px.toLocaleString()}
                </td>
                <td className="px-3 py-1 text-right text-slate-500">
                  {f.fee_bps.toFixed(1)} bps
                </td>
                <td
                  className={`px-3 py-1 text-right ${
                    f.pnl_usd > 0
                      ? "text-emerald-400"
                      : f.pnl_usd < 0
                        ? "text-rose-400"
                        : "text-slate-500"
                  }`}
                >
                  {f.pnl_usd === 0
                    ? "—"
                    : `${f.pnl_usd > 0 ? "+" : ""}${f.pnl_usd.toFixed(2)}`}
                </td>
                <td className="px-3 py-1 text-slate-400">{f.venue}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
