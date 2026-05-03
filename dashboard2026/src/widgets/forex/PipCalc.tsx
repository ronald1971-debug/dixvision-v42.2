import { useMemo, useState } from "react";

const PAIRS = [
  "EUR/USD",
  "GBP/USD",
  "USD/JPY",
  "AUD/USD",
  "USD/CHF",
  "USD/CAD",
  "NZD/USD",
  "EUR/JPY",
  "GBP/JPY",
] as const;

const RATE: Record<string, number> = {
  "EUR/USD": 1.0824,
  "GBP/USD": 1.2641,
  "USD/JPY": 154.32,
  "AUD/USD": 0.6512,
  "USD/CHF": 0.9082,
  "USD/CAD": 1.3712,
  "NZD/USD": 0.5921,
  "EUR/JPY": 167.05,
  "GBP/JPY": 195.12,
};

const LOT_STANDARD = 100_000;

function pipSize(pair: string): number {
  return pair.endsWith("JPY") ? 0.01 : 0.0001;
}

/** Returns how many USD one unit of `ccy` is worth, using direct or inverse cross. */
function ccyToUsd(ccy: string): number {
  if (ccy === "USD") return 1;
  const direct = RATE[`${ccy}/USD`];
  if (direct !== undefined) return direct;
  const inverse = RATE[`USD/${ccy}`];
  if (inverse !== undefined && inverse !== 0) return 1 / inverse;
  return 1;
}

export function PipCalc() {
  const [pair, setPair] = useState<string>("EUR/USD");
  const [lots, setLots] = useState<number>(1);
  const [account, setAccount] = useState<"USD" | "EUR" | "GBP">("USD");

  const result = useMemo(() => {
    const ps = pipSize(pair);
    const units = lots * LOT_STANDARD;
    const quote = pair.split("/")[1];
    const rate = RATE[pair] ?? 1;
    // pip value in quote ccy
    const pipQuote = ps * units;
    // convert to account ccy via USD as bridge.
    //   quote -> USD: multiply by ccyToUsd(quote)
    //   USD   -> account: divide by ccyToUsd(account)
    let pipAccount = pipQuote;
    if (quote !== account) {
      const quoteToUsd = ccyToUsd(quote);
      const accountToUsd = ccyToUsd(account);
      pipAccount = (pipQuote * quoteToUsd) / accountToUsd;
    }
    return {
      pipQuote,
      pipAccount,
      quote,
      rate,
      units,
    };
  }, [pair, lots, account]);

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Pip Calculator
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          standard lot · pip value × position size
        </p>
      </header>
      <div className="flex-1 space-y-3 px-3 py-3 text-[11px]">
        <label className="block">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">pair</span>
          <select
            value={pair}
            onChange={(e) => setPair(e.target.value)}
            className="mt-0.5 w-full rounded border border-border bg-slate-900/60 px-2 py-1 font-mono text-slate-200"
          >
            {PAIRS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">lots (standard)</span>
          <input
            type="number"
            step={0.01}
            min={0}
            value={lots}
            onChange={(e) => setLots(Number(e.target.value) || 0)}
            className="mt-0.5 w-full rounded border border-border bg-slate-900/60 px-2 py-1 font-mono text-slate-200"
          />
        </label>
        <label className="block">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">account ccy</span>
          <select
            value={account}
            onChange={(e) => setAccount(e.target.value as "USD" | "EUR" | "GBP")}
            className="mt-0.5 w-full rounded border border-border bg-slate-900/60 px-2 py-1 font-mono text-slate-200"
          >
            <option value="USD">USD</option>
            <option value="EUR">EUR</option>
            <option value="GBP">GBP</option>
          </select>
        </label>
        <div className="space-y-1 rounded border border-border bg-slate-900/40 px-2 py-2 font-mono">
          <Row label="units" value={result.units.toLocaleString()} />
          <Row label={`rate (${pair})`} value={result.rate.toFixed(4)} />
          <Row label={`pip value (${result.quote})`} value={result.pipQuote.toFixed(2)} />
          <Row
            label={`pip value (${account})`}
            value={result.pipAccount.toFixed(2)}
            tone="emerald"
          />
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{label}</span>
      <span
        className={`text-[11px] ${tone === "emerald" ? "text-emerald-300" : "text-slate-200"}`}
      >
        {value}
      </span>
    </div>
  );
}
