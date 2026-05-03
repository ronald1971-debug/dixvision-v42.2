import { useEffect, useState } from "react";

/**
 * Tier-5 on-chain widget — Open-interest matrix.
 *
 * Aggregate perp open interest by exchange × symbol with 24h
 * delta. Rising OI alongside a price move is canonically
 * trend-confirming; rising OI against price is a squeeze setup.
 *
 * Real wiring through the Coalesce / Coinglass adapter (filed).
 * Mock drifts deterministically.
 */
const VENUES = ["Binance", "Bybit", "OKX", "HL", "dYdX"];
const SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "DOGE"];

interface Cell {
  oi_usd: number;
  d24h_pct: number;
}

function seedMatrix(): Cell[][] {
  const rows: Cell[][] = [];
  for (let i = 0; i < SYMBOLS.length; i += 1) {
    const row: Cell[] = [];
    for (let j = 0; j < VENUES.length; j += 1) {
      row.push({
        oi_usd: 200_000_000 + ((i * 7 + j * 13) % 9) * 380_000_000,
        d24h_pct: ((i + j) % 7) * 0.004 - 0.012,
      });
    }
    rows.push(row);
  }
  return rows;
}

export function OpenInterestMatrix() {
  const [matrix, setMatrix] = useState<Cell[][]>(() => seedMatrix());

  useEffect(() => {
    const id = setInterval(() => {
      setMatrix((prev) =>
        prev.map((row, i) =>
          row.map((c, j) => {
            const drift = Math.sin(Date.now() / 6_500 + i + j * 7) * 0.001;
            return {
              oi_usd: c.oi_usd * (1 + drift),
              d24h_pct: c.d24h_pct + drift * 0.5,
            };
          }),
        ),
      );
    }, 4_000);
    return () => clearInterval(id);
  }, []);

  const totalsBySymbol = matrix.map((row) =>
    row.reduce((s, c) => s + c.oi_usd, 0),
  );

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Open interest matrix
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          perp OI by venue × symbol · cell shows OI / 24h Δ
        </p>
      </header>
      <div className="flex-1 overflow-auto p-2">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="text-[9px] uppercase tracking-wider text-slate-500">
              <th className="px-2 py-1 text-left">sym</th>
              {VENUES.map((v) => (
                <th key={v} className="px-2 py-1 text-right">
                  {v}
                </th>
              ))}
              <th className="px-2 py-1 text-right">Σ</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {SYMBOLS.map((sym, i) => (
              <tr key={sym} className="border-t border-border/40">
                <td className="px-2 py-1 font-semibold text-slate-200">
                  {sym}
                </td>
                {matrix[i].map((c, j) => {
                  const tone =
                    c.d24h_pct >= 0.005
                      ? "text-emerald-300"
                      : c.d24h_pct <= -0.005
                        ? "text-rose-300"
                        : "text-slate-400";
                  return (
                    <td key={j} className={`px-2 py-1 text-right ${tone}`}>
                      <div>${(c.oi_usd / 1_000_000_000).toFixed(2)}B</div>
                      <div className="text-[9px]">
                        {(c.d24h_pct * 100).toFixed(2)}%
                      </div>
                    </td>
                  );
                })}
                <td className="px-2 py-1 text-right text-slate-300">
                  ${(totalsBySymbol[i] / 1_000_000_000).toFixed(2)}B
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
