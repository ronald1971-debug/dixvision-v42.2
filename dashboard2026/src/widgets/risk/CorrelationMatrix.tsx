import { useEffect, useState } from "react";

/**
 * Tier-6 risk widget — Correlation matrix.
 *
 * 30-day rolling pairwise Pearson correlation across the
 * portfolio's reference symbols. Real wiring drives this off
 * the price-history adapter; mock seeds to plausible values
 * and drifts via a sine so the panel feels alive.
 */
const SYMS = ["BTC", "ETH", "SOL", "GOLD", "DXY", "NDX", "BONK"];

function seedMatrix(): number[][] {
  // Hand-tuned plausible 30d correlations.
  return [
    [1.0, 0.84, 0.71, 0.05, -0.36, 0.42, 0.58],
    [0.84, 1.0, 0.78, 0.04, -0.32, 0.45, 0.61],
    [0.71, 0.78, 1.0, 0.02, -0.28, 0.41, 0.74],
    [0.05, 0.04, 0.02, 1.0, -0.41, 0.07, 0.01],
    [-0.36, -0.32, -0.28, -0.41, 1.0, -0.55, -0.18],
    [0.42, 0.45, 0.41, 0.07, -0.55, 1.0, 0.34],
    [0.58, 0.61, 0.74, 0.01, -0.18, 0.34, 1.0],
  ];
}

function tone(c: number): string {
  if (c >= 0.85) return "bg-emerald-500/40 text-emerald-100";
  if (c >= 0.6) return "bg-emerald-500/25 text-emerald-200";
  if (c >= 0.3) return "bg-emerald-500/10 text-emerald-300";
  if (c >= -0.3) return "text-slate-300";
  if (c >= -0.6) return "bg-rose-500/10 text-rose-300";
  return "bg-rose-500/25 text-rose-200";
}

export function CorrelationMatrix() {
  const [m, setM] = useState<number[][]>(() => seedMatrix());

  useEffect(() => {
    const id = setInterval(() => {
      setM((prev) =>
        prev.map((row, i) =>
          row.map((v, j) => {
            if (i === j) return 1;
            const drift = Math.sin(Date.now() / 9_000 + i * 3 + j) * 0.012;
            const next = Math.max(-0.99, Math.min(0.99, v + drift));
            return next;
          }),
        ),
      );
    }, 4_500);
    return () => clearInterval(id);
  }, []);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Correlation matrix
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          rolling 30d Pearson · drifting in real time
        </p>
      </header>
      <div className="flex-1 overflow-auto p-2">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="text-[9px] uppercase tracking-wider text-slate-500">
              <th className="px-2 py-1" />
              {SYMS.map((s) => (
                <th key={s} className="px-2 py-1 text-right">
                  {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono">
            {SYMS.map((row, i) => (
              <tr key={row} className="border-t border-border/40">
                <td className="px-2 py-1 font-semibold text-slate-200">
                  {row}
                </td>
                {SYMS.map((_, j) => (
                  <td
                    key={j}
                    className={`px-2 py-1 text-right ${tone(m[i][j])}`}
                  >
                    {m[i][j].toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
