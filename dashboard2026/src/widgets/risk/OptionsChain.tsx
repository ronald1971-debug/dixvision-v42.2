import { useMemo, useState } from "react";

/**
 * Tier-6 risk widget — Options chain.
 *
 * Calls / puts grid keyed on strike, with mid IV, mid price,
 * Δ, Γ, Θ, Vega per side. Reference price is fixed for the
 * mock; in the live wiring this comes from the spot adapter.
 *
 * Pricing uses a flat Black-Scholes style closed-form on
 * deterministic flat-vol so the panel feels alive without an
 * options data feed. Real chain data lives in the venue
 * adapter (Deribit / OKX options; CME for index options).
 */
type Side = "C" | "P";

interface Row {
  strike: number;
  call: Greeks;
  put: Greeks;
}

interface Greeks {
  iv: number;
  mid: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  oi: number;
}

const SPOT = 67_400;
const T_DAYS = 21;
const FLAT_IV = 0.62;

function ndf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function ncdf(x: number): number {
  // Abramowitz & Stegun 26.2.17 — adequate for mock pricing
  const k = 1 / (1 + 0.2316419 * Math.abs(x));
  const w =
    1 -
    ndf(x) *
      k *
      (0.319381530 +
        k *
          (-0.356563782 +
            k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));
  return x >= 0 ? w : 1 - w;
}

function bs(side: Side, S: number, K: number, T: number, sigma: number): Greeks {
  const sqrtT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT);
  const d2 = d1 - sigma * sqrtT;
  const call_px = S * ncdf(d1) - K * ncdf(d2);
  const put_px = K * ncdf(-d2) - S * ncdf(-d1);
  const mid = side === "C" ? call_px : put_px;
  const delta = side === "C" ? ncdf(d1) : ncdf(d1) - 1;
  const gamma = ndf(d1) / (S * sigma * sqrtT);
  const theta = (-S * ndf(d1) * sigma) / (2 * sqrtT) / 365;
  const vega = (S * ndf(d1) * sqrtT) / 100;
  const oi = 200 + ((Math.abs(K - S) | 0) % 1300);
  return {
    iv: sigma,
    mid,
    delta,
    gamma,
    theta,
    vega,
    oi,
  };
}

export function OptionsChain() {
  const [expiry, setExpiry] = useState(T_DAYS);
  const rows: Row[] = useMemo(() => {
    const out: Row[] = [];
    const T = expiry / 365;
    for (let i = -6; i <= 6; i += 1) {
      const strike = Math.round((SPOT + i * 1_000) / 100) * 100;
      out.push({
        strike,
        call: bs("C", SPOT, strike, T, FLAT_IV),
        put: bs("P", SPOT, strike, T, FLAT_IV),
      });
    }
    return out;
  }, [expiry]);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Options chain · BTC
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            spot ${SPOT.toLocaleString()} · flat IV {(FLAT_IV * 100).toFixed(0)}
            %
          </p>
        </div>
        <label className="flex items-center gap-2 text-[10px] text-slate-400">
          expiry
          <select
            value={expiry}
            onChange={(e) => setExpiry(parseInt(e.target.value, 10))}
            className="rounded border border-border bg-bg/40 px-1.5 py-0.5 text-[11px] text-slate-200"
          >
            {[7, 14, 21, 28, 60, 90].map((d) => (
              <option key={d} value={d}>
                {d}d
              </option>
            ))}
          </select>
        </label>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[10px]">
          <thead className="sticky top-0 bg-surface text-[9px] uppercase tracking-wider text-slate-500">
            <tr className="text-right">
              <th className="px-2 py-1 text-right text-emerald-300/70" colSpan={4}>
                CALLS
              </th>
              <th className="bg-bg/40 px-2 py-1 text-center">strike</th>
              <th className="px-2 py-1 text-rose-300/70" colSpan={4}>
                PUTS
              </th>
            </tr>
            <tr className="text-right">
              <th className="px-2 py-1">mid</th>
              <th className="px-2 py-1">Δ</th>
              <th className="px-2 py-1">Γ</th>
              <th className="px-2 py-1">Θ</th>
              <th className="bg-bg/40 px-2 py-1 text-center">K</th>
              <th className="px-2 py-1">mid</th>
              <th className="px-2 py-1">Δ</th>
              <th className="px-2 py-1">Γ</th>
              <th className="px-2 py-1">Θ</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((r) => {
              const itmCall = r.strike < SPOT;
              const itmPut = r.strike > SPOT;
              return (
                <tr
                  key={r.strike}
                  className="border-t border-border/40 text-right"
                >
                  <td
                    className={`px-2 py-1 ${itmCall ? "text-slate-200" : "text-slate-500"}`}
                  >
                    {r.call.mid.toFixed(0)}
                  </td>
                  <td className="px-2 py-1 text-slate-400">
                    {r.call.delta.toFixed(2)}
                  </td>
                  <td className="px-2 py-1 text-slate-500">
                    {r.call.gamma.toFixed(5)}
                  </td>
                  <td className="px-2 py-1 text-slate-500">
                    {r.call.theta.toFixed(1)}
                  </td>
                  <td className="bg-bg/40 px-2 py-1 text-center font-semibold text-slate-200">
                    {r.strike.toLocaleString()}
                  </td>
                  <td
                    className={`px-2 py-1 ${itmPut ? "text-slate-200" : "text-slate-500"}`}
                  >
                    {r.put.mid.toFixed(0)}
                  </td>
                  <td className="px-2 py-1 text-slate-400">
                    {r.put.delta.toFixed(2)}
                  </td>
                  <td className="px-2 py-1 text-slate-500">
                    {r.put.gamma.toFixed(5)}
                  </td>
                  <td className="px-2 py-1 text-slate-500">
                    {r.put.theta.toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
