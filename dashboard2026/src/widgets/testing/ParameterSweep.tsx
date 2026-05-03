/**
 * I-track widget — Parameter sweep heatmap.
 *
 * Visualises a 2-D grid of strategy parameters with the resulting
 * Sharpe ratio per cell. Dark cells = poor, bright cells = best.
 * The "best" cell is outlined in emerald.
 *
 * Backend hook: ``GET /api/testing/sweep?strategy=&p1=&p1_grid=&p2=&p2_grid=``
 * runs the canonical historical replay across the cartesian product
 * and returns the metric matrix.
 */
const P1_LABEL = "lookback (bars)";
const P1_VALUES = [5, 10, 15, 20, 30, 50, 75, 100];
const P2_LABEL = "z-threshold";
const P2_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25];

function sharpeAt(p1Idx: number, p2Idx: number): number {
  const peak1 = 3;
  const peak2 = 2;
  const d1 = (p1Idx - peak1) / 4;
  const d2 = (p2Idx - peak2) / 3;
  const base = 2.4 * Math.exp(-(d1 * d1 + d2 * d2));
  const ridge = 0.4 * Math.cos(p1Idx + p2Idx);
  return Math.max(-0.5, base + ridge - 0.1);
}

function cellColor(v: number, lo: number, hi: number): string {
  const t = Math.min(Math.max((v - lo) / (hi - lo), 0), 1);
  if (t < 0.4) {
    const k = t / 0.4;
    return `rgba(244, 63, 94, ${0.25 + 0.4 * (1 - k)})`;
  }
  if (t < 0.7) {
    const k = (t - 0.4) / 0.3;
    return `rgba(245, 158, 11, ${0.4 + 0.2 * k})`;
  }
  const k = (t - 0.7) / 0.3;
  return `rgba(16, 185, 129, ${0.45 + 0.4 * k})`;
}

export function ParameterSweep() {
  const grid = P1_VALUES.map((_, i) => P2_VALUES.map((_, j) => sharpeAt(i, j)));
  const flat = grid.flat();
  const lo = Math.min(...flat);
  const hi = Math.max(...flat);
  let best = { i: 0, j: 0, v: -Infinity };
  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[i].length; j++) {
      if (grid[i][j] > best.v) best = { i, j, v: grid[i][j] };
    }
  }

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2 flex items-baseline justify-between">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Parameter sweep
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            strat_meanrev_v3 · BTC-USDT · 90d · cell = Sharpe
          </p>
        </div>
        <div className="font-mono text-[10px]">
          <span className="text-slate-500">best</span>{" "}
          <span className="text-emerald-400">
            {P1_LABEL.split(" ")[0]}={P1_VALUES[best.i]} · z={P2_VALUES[best.j].toFixed(2)} · S={best.v.toFixed(2)}
          </span>
        </div>
      </header>
      <div className="flex-1 overflow-auto p-3">
        <table className="w-full font-mono text-[10px] text-slate-300">
          <thead>
            <tr>
              <th className="px-1 py-0.5 text-right text-slate-500">{P1_LABEL} \ {P2_LABEL}</th>
              {P2_VALUES.map((v) => (
                <th key={v} className="px-1 py-0.5 text-right text-slate-500">{v.toFixed(2)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {P1_VALUES.map((p1, i) => (
              <tr key={p1}>
                <td className="px-1 py-0.5 text-right text-slate-400">{p1}</td>
                {P2_VALUES.map((p2, j) => {
                  const v = grid[i][j];
                  const isBest = i === best.i && j === best.j;
                  return (
                    <td key={p2} className="px-0.5 py-0.5">
                      <div
                        className={`rounded px-1 py-0.5 text-center text-[10px] ${
                          isBest ? "ring-2 ring-emerald-400 text-slate-900" : "text-slate-900"
                        }`}
                        style={{ backgroundColor: cellColor(v, lo, hi) }}
                        title={`${P1_LABEL}=${p1} ${P2_LABEL}=${p2} Sharpe=${v.toFixed(2)}`}
                      >
                        {v.toFixed(2)}
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
        red = poor · amber = mid · emerald = best · ring = global max
      </footer>
    </section>
  );
}
