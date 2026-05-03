import { useEffect, useState } from "react";

/**
 * G-track widget — Open orders panel.
 *
 * Lists working / partially-filled orders with cancel buttons.
 * Backend hook: ``GET /api/positions/open_orders`` reads from
 * ``execution_engine.order_book``; ``DELETE /api/positions/orders/{id}``
 * issues a cancel intent through the operator-approval edge.
 */
type Status = "WORKING" | "PARTIAL" | "PENDING";

interface OpenOrder {
  id: string;
  symbol: string;
  side: "BUY" | "SELL";
  type: "LMT" | "STP" | "STP-LMT" | "MKT";
  qty: number;
  filled: number;
  px: number;
  status: Status;
  age_s: number;
  venue: string;
}

const SEED: OpenOrder[] = [
  { id: "o-1", symbol: "BTC-USDT", side: "BUY", type: "LMT", qty: 0.5, filled: 0, px: 67_400, status: "WORKING", age_s: 42, venue: "binance" },
  { id: "o-2", symbol: "ETH-USDT", side: "SELL", type: "LMT", qty: 4, filled: 1.2, px: 3_530, status: "PARTIAL", age_s: 188, venue: "binance" },
  { id: "o-3", symbol: "SOL-USDT", side: "BUY", type: "STP-LMT", qty: 80, filled: 0, px: 143.20, status: "WORKING", age_s: 612, venue: "kraken" },
  { id: "o-4", symbol: "AVAX-USDT", side: "BUY", type: "LMT", qty: 200, filled: 0, px: 31.50, status: "WORKING", age_s: 1_204, venue: "okx" },
  { id: "o-5", symbol: "MATIC-USDT", side: "SELL", type: "STP", qty: 5_000, filled: 0, px: 0.512, status: "PENDING", age_s: 6, venue: "binance" },
];

export function OpenOrdersPanel() {
  const [rows, setRows] = useState<OpenOrder[]>(SEED);

  useEffect(() => {
    const t = setInterval(() => {
      setRows((prev) =>
        prev.map((r) => ({ ...r, age_s: r.age_s + 5 })),
      );
    }, 5000);
    return () => clearInterval(t);
  }, []);

  const cancel = (id: string) =>
    setRows((prev) => prev.filter((r) => r.id !== id));

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Open orders
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          {rows.length} working · cancel issues approval-edge intent
        </p>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">symbol</th>
              <th className="px-3 py-1.5 text-left">side</th>
              <th className="px-3 py-1.5 text-left">type</th>
              <th className="px-3 py-1.5 text-right">qty</th>
              <th className="px-3 py-1.5 text-right">filled</th>
              <th className="px-3 py-1.5 text-right">px</th>
              <th className="px-3 py-1.5 text-left">venue</th>
              <th className="px-3 py-1.5 text-right">age</th>
              <th className="px-3 py-1.5"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {rows.map((r) => (
              <tr key={r.id}>
                <td className="px-3 py-1 text-slate-200">{r.symbol}</td>
                <td
                  className={`px-3 py-1 ${
                    r.side === "BUY" ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {r.side}
                </td>
                <td className="px-3 py-1 text-slate-400">{r.type}</td>
                <td className="px-3 py-1 text-right">{r.qty}</td>
                <td className="px-3 py-1 text-right text-slate-400">
                  {r.filled > 0 ? r.filled : "—"}
                </td>
                <td className="px-3 py-1 text-right">{r.px.toLocaleString()}</td>
                <td className="px-3 py-1 text-slate-400">{r.venue}</td>
                <td className="px-3 py-1 text-right text-slate-500">
                  {r.age_s < 60 ? `${r.age_s}s` : `${Math.floor(r.age_s / 60)}m`}
                </td>
                <td className="px-3 py-1 text-right">
                  <button
                    type="button"
                    onClick={() => cancel(r.id)}
                    className="rounded border border-border bg-bg/40 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-slate-500 hover:border-rose-500/40 hover:text-rose-400"
                  >
                    cancel
                  </button>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td
                  colSpan={9}
                  className="px-3 py-6 text-center text-[11px] text-slate-500"
                >
                  no working orders
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
