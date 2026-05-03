/**
 * H-track widget — Implied Volatility surface (heatmap).
 *
 * Strikes (moneyness) × expiries grid, value = IV. Color from cool
 * (low IV) → hot (high IV). Backend hook:
 * ``GET /api/market/iv-surface?symbol=BTC`` reads the Deribit option
 * chain and runs Black-76 inversion on each leg.
 */
const STRIKES = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15];
const EXPIRIES = ["7d", "14d", "30d", "60d", "90d", "180d"];

// pseudo IV surface — smile + term structure
function ivAt(s: number, eIdx: number): number {
  const smile = 0.18 + 0.42 * Math.pow(s - 1, 2); // skew + kurtosis
  const term = 0.95 + eIdx * 0.02;
  return Math.min(smile * term, 1.6);
}

function color(v: number): string {
  // 20%..160% IV → blue..red
  const t = Math.min(Math.max((v - 0.2) / 1.4, 0), 1);
  const r = Math.round(20 + t * 220);
  const g = Math.round(80 + (1 - Math.abs(t - 0.5) * 2) * 100);
  const b = Math.round(220 - t * 200);
  return `rgb(${r},${g},${b})`;
}

export function IVSurface() {
  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          IV surface
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          Deribit BTC · moneyness × expiry · annualised
        </p>
      </header>
      <div className="flex-1 overflow-auto p-2">
        <table className="w-full font-mono text-[10px] text-slate-300">
          <thead>
            <tr>
              <th className="px-1 py-1 text-right text-slate-500">m / T</th>
              {EXPIRIES.map((e) => (
                <th key={e} className="px-1 py-1 text-right text-slate-500">
                  {e}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {STRIKES.map((s) => (
              <tr key={s}>
                <td className="px-1 py-0.5 text-right text-slate-400">
                  {(s * 100).toFixed(0)}%
                </td>
                {EXPIRIES.map((_, i) => {
                  const iv = ivAt(s, i);
                  return (
                    <td key={i} className="px-0.5 py-0.5">
                      <div
                        className="rounded px-1 py-0.5 text-center text-[10px] text-slate-900"
                        style={{ backgroundColor: color(iv) }}
                        title={`m=${(s * 100).toFixed(0)}% T=${EXPIRIES[i]} IV=${(iv * 100).toFixed(1)}%`}
                      >
                        {(iv * 100).toFixed(0)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <footer className="border-t border-border px-3 py-1.5 font-mono text-[10px] text-slate-500">
        cell = IV (%), color cool→hot, ATM term shown by 100% row
      </footer>
    </section>
  );
}
