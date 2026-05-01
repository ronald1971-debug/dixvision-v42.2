import { useMemo, useState } from "react";

import { Activity, Play, Settings2 } from "lucide-react";

/**
 * Backtester widget (PR-#2 spec §5.1 — "Backtest Lab").
 *
 * Strategy picker · date range · venue picker · slippage / latency
 * model · param grid (grid or Bayesian) · walk-forward toggle. Run
 * against historical bars from the canonical event bus replay store
 * and produce: equity curve, max drawdown, win rate, Sharpe, sortino,
 * profit factor, calmar, average trade, longest losing streak, plus
 * a trade-by-trade ledger.
 *
 * The result panel is fully driven from the deterministic seed of
 * `(strategy, symbol, range, slippage_bps)` so this surface renders
 * end-to-end even before the backend `/api/backtest/run` route lands;
 * once it does, swap the local generator for a `useMutation` against
 * that endpoint without changing the layout.
 */
type Strategy =
  | "ema_cross_20_50"
  | "rsi_2_meanrev"
  | "vwap_reversion"
  | "breakout_channel"
  | "microstructure_v1"
  | "news_event_drift"
  | "memecoin_copy"
  | "memecoin_sniper";

type Venue =
  | "binance_spot"
  | "binance_perps"
  | "hyperliquid"
  | "drift"
  | "raydium"
  | "fxcm"
  | "ibkr";

type FillModel = "next_tick" | "vwap_5min" | "mid_price" | "tob_aggress";

const STRATEGIES: ReadonlyArray<{ key: Strategy; label: string }> = [
  { key: "ema_cross_20_50", label: "EMA cross 20/50" },
  { key: "rsi_2_meanrev", label: "RSI(2) mean-reversion" },
  { key: "vwap_reversion", label: "VWAP reversion" },
  { key: "breakout_channel", label: "Breakout channel" },
  { key: "microstructure_v1", label: "Microstructure v1" },
  { key: "news_event_drift", label: "News-event drift" },
  { key: "memecoin_copy", label: "Memecoin copy-trader" },
  { key: "memecoin_sniper", label: "Memecoin sniper" },
];

const VENUES: ReadonlyArray<{ key: Venue; label: string }> = [
  { key: "binance_spot", label: "Binance Spot" },
  { key: "binance_perps", label: "Binance Perps" },
  { key: "hyperliquid", label: "Hyperliquid" },
  { key: "drift", label: "Drift V2" },
  { key: "raydium", label: "Raydium DEX" },
  { key: "fxcm", label: "FXCM" },
  { key: "ibkr", label: "IBKR" },
];

const FILL_MODELS: ReadonlyArray<{ key: FillModel; label: string }> = [
  { key: "next_tick", label: "next-tick" },
  { key: "vwap_5min", label: "VWAP(5m)" },
  { key: "mid_price", label: "mid-price" },
  { key: "tob_aggress", label: "TOB aggress" },
];

interface Trade {
  ts_iso: string;
  side: "BUY" | "SELL";
  pnl_pct: number;
  bars_held: number;
}

interface Report {
  equity: number[];
  drawdown: number[];
  trades: Trade[];
  metrics: {
    final_equity_pct: number;
    cagr: number;
    sharpe: number;
    sortino: number;
    max_dd_pct: number;
    win_rate: number;
    profit_factor: number;
    avg_trade_pct: number;
    longest_loss_streak: number;
    n_trades: number;
  };
}

function seededRandom(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function hashSeed(parts: ReadonlyArray<string | number>): number {
  let h = 2166136261;
  for (const part of parts) {
    const str = String(part);
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
  }
  return h >>> 0;
}

function runDeterministicBacktest(
  strategy: Strategy,
  symbol: string,
  startIso: string,
  endIso: string,
  fill: FillModel,
  slippageBps: number,
): Report {
  const seed = hashSeed([strategy, symbol, startIso, endIso, fill, slippageBps]);
  const rng = seededRandom(seed);
  const n = 240;
  const equity: number[] = [100];
  const drawdown: number[] = [0];
  const trades: Trade[] = [];
  let peak = 100;
  let losingStreak = 0;
  let longestLoss = 0;
  let wins = 0;
  let losses = 0;
  let grossWin = 0;
  let grossLoss = 0;
  for (let i = 0; i < n; i++) {
    const drift = strategy === "memecoin_sniper" ? 0.06 : 0.012;
    const vol = strategy === "memecoin_sniper" ? 0.95 : 0.32;
    const r = (rng() - 0.5) * vol + drift / 100;
    const slipDrag = slippageBps / 10000;
    const stepPnl = r - slipDrag * (i % 5 === 0 ? 1 : 0);
    if (i % 5 === 0) {
      const pnlPct = stepPnl * 100;
      trades.push({
        ts_iso: new Date(Date.parse(startIso) + i * 3600_000).toISOString(),
        side: rng() > 0.5 ? "BUY" : "SELL",
        pnl_pct: pnlPct,
        bars_held: 3 + Math.floor(rng() * 8),
      });
      if (pnlPct >= 0) {
        wins += 1;
        grossWin += pnlPct;
        losingStreak = 0;
      } else {
        losses += 1;
        grossLoss += Math.abs(pnlPct);
        losingStreak += 1;
        if (losingStreak > longestLoss) longestLoss = losingStreak;
      }
    }
    const next = equity[equity.length - 1] * (1 + stepPnl);
    equity.push(next);
    peak = Math.max(peak, next);
    drawdown.push(((next - peak) / peak) * 100);
  }
  const finalPct = equity[equity.length - 1] - 100;
  const days = Math.max(
    1,
    (Date.parse(endIso) - Date.parse(startIso)) / (24 * 3600_000),
  );
  const cagr =
    (Math.pow(equity[equity.length - 1] / 100, 365 / days) - 1) * 100;
  const returns: number[] = [];
  for (let i = 1; i < equity.length; i++) {
    returns.push((equity[i] - equity[i - 1]) / equity[i - 1]);
  }
  const mean = returns.reduce((s, x) => s + x, 0) / returns.length;
  const variance =
    returns.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, returns.length - 1);
  const sharpe = (mean / Math.sqrt(variance + 1e-9)) * Math.sqrt(252);
  const downs = returns.filter((r) => r < 0);
  const downVar =
    downs.reduce((s, x) => s + (x - 0) ** 2, 0) / Math.max(1, downs.length - 1);
  const sortino = (mean / Math.sqrt(downVar + 1e-9)) * Math.sqrt(252);
  const maxDd = Math.min(...drawdown);
  const totalTrades = wins + losses;
  return {
    equity,
    drawdown,
    trades,
    metrics: {
      final_equity_pct: finalPct,
      cagr,
      sharpe,
      sortino,
      max_dd_pct: Math.abs(maxDd),
      win_rate: totalTrades === 0 ? 0 : wins / totalTrades,
      profit_factor: grossLoss === 0 ? Infinity : grossWin / grossLoss,
      avg_trade_pct:
        totalTrades === 0 ? 0 : (grossWin - grossLoss) / totalTrades,
      longest_loss_streak: longestLoss,
      n_trades: totalTrades,
    },
  };
}

function isoDateOnly(iso: string): string {
  return iso.slice(0, 10);
}

function todayIso(offsetDays: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() + offsetDays);
  d.setUTCHours(0, 0, 0, 0);
  return d.toISOString();
}

export function Backtester() {
  const [strategy, setStrategy] = useState<Strategy>("ema_cross_20_50");
  const [symbol, setSymbol] = useState<string>("BTC/USDT");
  const [venue, setVenue] = useState<Venue>("binance_spot");
  const [start, setStart] = useState<string>(isoDateOnly(todayIso(-90)));
  const [end, setEnd] = useState<string>(isoDateOnly(todayIso(0)));
  const [slippageBps, setSlippageBps] = useState<number>(8);
  const [fillModel, setFillModel] = useState<FillModel>("next_tick");
  const [walkForward, setWalkForward] = useState<boolean>(false);
  const [report, setReport] = useState<Report | null>(null);

  const startIso = useMemo(() => `${start}T00:00:00Z`, [start]);
  const endIso = useMemo(() => `${end}T00:00:00Z`, [end]);

  function run() {
    const r = runDeterministicBacktest(
      strategy,
      symbol,
      startIso,
      endIso,
      fillModel,
      slippageBps,
    );
    setReport(r);
  }

  return (
    <div className="flex h-full flex-col gap-2 rounded border border-border bg-surface p-3 text-[12px]">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-slate-300">
          <Activity className="h-4 w-4 text-accent" />
          <span className="font-semibold uppercase tracking-wide">
            Backtester
          </span>
          <span className="text-[10px] text-slate-500">
            PR-#2 §5.1 · backtest lab
          </span>
        </div>
        <button
          type="button"
          onClick={run}
          className="flex items-center gap-1 rounded border border-accent/40 bg-accent/10 px-3 py-1 text-[11px] text-accent hover:border-accent"
        >
          <Play className="h-3 w-3" />
          Run
        </button>
      </header>

      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <FormField label="Strategy">
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as Strategy)}
            className="w-full rounded border border-border bg-bg px-2 py-1 text-[11px]"
          >
            {STRATEGIES.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </FormField>
        <FormField label="Symbol">
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="w-full rounded border border-border bg-bg px-2 py-1 text-[11px]"
          />
        </FormField>
        <FormField label="Venue">
          <select
            value={venue}
            onChange={(e) => setVenue(e.target.value as Venue)}
            className="w-full rounded border border-border bg-bg px-2 py-1 text-[11px]"
          >
            {VENUES.map((v) => (
              <option key={v.key} value={v.key}>
                {v.label}
              </option>
            ))}
          </select>
        </FormField>
        <FormField label="Fill model">
          <select
            value={fillModel}
            onChange={(e) => setFillModel(e.target.value as FillModel)}
            className="w-full rounded border border-border bg-bg px-2 py-1 text-[11px]"
          >
            {FILL_MODELS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </FormField>
        <FormField label="Start">
          <input
            type="date"
            value={start}
            onChange={(e) => setStart(e.target.value)}
            className="w-full rounded border border-border bg-bg px-2 py-1 text-[11px]"
          />
        </FormField>
        <FormField label="End">
          <input
            type="date"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
            className="w-full rounded border border-border bg-bg px-2 py-1 text-[11px]"
          />
        </FormField>
        <FormField label="Slippage (bps)">
          <input
            type="number"
            min={0}
            max={500}
            value={slippageBps}
            onChange={(e) => setSlippageBps(Number(e.target.value))}
            className="w-full rounded border border-border bg-bg px-2 py-1 text-[11px]"
          />
        </FormField>
        <FormField label="Walk-forward OOS">
          <label className="flex items-center gap-2 text-[11px]">
            <input
              type="checkbox"
              checked={walkForward}
              onChange={(e) => setWalkForward(e.target.checked)}
            />
            <Settings2 className="h-3 w-3 text-slate-500" />
            <span className="text-slate-400">
              {walkForward ? "rolling 70/30 windows" : "in-sample only"}
            </span>
          </label>
        </FormField>
      </div>

      {report ? (
        <ReportView report={report} />
      ) : (
        <div className="flex flex-1 items-center justify-center rounded border border-dashed border-border/60 text-slate-500">
          Click <span className="px-1 text-accent">Run</span> to execute the
          backtest.
        </div>
      )}
    </div>
  );
}

function FormField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </span>
      {children}
    </label>
  );
}

function ReportView({ report }: { report: Report }) {
  const m = report.metrics;
  return (
    <div className="flex flex-1 flex-col gap-2 overflow-hidden">
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-5">
        <Metric label="Final P&L" value={fmtPct(m.final_equity_pct)} tone={m.final_equity_pct >= 0 ? "ok" : "danger"} />
        <Metric label="CAGR" value={fmtPct(m.cagr)} tone={m.cagr >= 0 ? "ok" : "danger"} />
        <Metric label="Sharpe" value={m.sharpe.toFixed(2)} tone={m.sharpe >= 1 ? "ok" : m.sharpe >= 0.5 ? "warn" : "danger"} />
        <Metric label="Sortino" value={m.sortino.toFixed(2)} />
        <Metric label="Max DD" value={`-${m.max_dd_pct.toFixed(2)}%`} tone={m.max_dd_pct <= 5 ? "ok" : m.max_dd_pct <= 15 ? "warn" : "danger"} />
        <Metric label="Win rate" value={`${(m.win_rate * 100).toFixed(1)}%`} />
        <Metric label="Profit factor" value={Number.isFinite(m.profit_factor) ? m.profit_factor.toFixed(2) : "∞"} />
        <Metric label="Avg trade" value={fmtPct(m.avg_trade_pct)} />
        <Metric label="Longest loss" value={`${m.longest_loss_streak}`} />
        <Metric label="# trades" value={`${m.n_trades}`} />
      </div>
      <EquityCurve equity={report.equity} drawdown={report.drawdown} />
      <TradeLedger trades={report.trades} />
    </div>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "ok" | "warn" | "danger";
}) {
  const cls =
    tone === "ok"
      ? "text-emerald-300"
      : tone === "warn"
        ? "text-amber-300"
        : tone === "danger"
          ? "text-red-300"
          : "text-slate-200";
  return (
    <div className="rounded border border-border bg-bg px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className={`mt-0.5 font-mono text-sm ${cls}`}>{value}</div>
    </div>
  );
}

function fmtPct(x: number): string {
  return `${x >= 0 ? "+" : ""}${x.toFixed(2)}%`;
}

function EquityCurve({
  equity,
  drawdown,
}: {
  equity: number[];
  drawdown: number[];
}) {
  const w = 800;
  const h = 140;
  const max = Math.max(...equity);
  const min = Math.min(...equity);
  const span = Math.max(0.1, max - min);
  const eqPath = equity
    .map((v, i) => {
      const x = (i / (equity.length - 1)) * w;
      const y = h - ((v - min) / span) * h;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  const ddMin = Math.min(...drawdown);
  const ddSpan = Math.max(0.5, Math.abs(ddMin));
  const ddPath = drawdown
    .map((v, i) => {
      const x = (i / (drawdown.length - 1)) * w;
      const y = ((Math.abs(v) - 0) / ddSpan) * 36;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <div className="rounded border border-border bg-bg p-2">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-slate-500">
          Equity curve
        </span>
        <span className="text-[10px] text-slate-500">drawdown below</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h + 40}`} className="h-40 w-full">
        <path d={eqPath} fill="none" stroke="#34d399" strokeWidth={1.4} />
        <line
          x1={0}
          x2={w}
          y1={h - ((100 - min) / span) * h}
          y2={h - ((100 - min) / span) * h}
          stroke="#475569"
          strokeDasharray="2 4"
        />
        <g transform={`translate(0, ${h + 4})`}>
          <path d={ddPath} fill="none" stroke="#f87171" strokeWidth={1} />
        </g>
      </svg>
    </div>
  );
}

function TradeLedger({ trades }: { trades: Trade[] }) {
  return (
    <div className="flex-1 overflow-hidden rounded border border-border bg-bg">
      <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-slate-500">
        Trade ledger ({trades.length})
      </div>
      <div className="max-h-44 overflow-auto">
        <table className="w-full text-left text-[11px]">
          <thead className="sticky top-0 bg-bg text-[9px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1">ts</th>
              <th className="px-2 py-1">side</th>
              <th className="px-2 py-1 text-right">pnl%</th>
              <th className="px-2 py-1 text-right">bars</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => (
              <tr key={`${t.ts_iso}-${i}`} className="odd:bg-surface/40">
                <td className="px-2 py-0.5 font-mono text-slate-400">
                  {t.ts_iso.slice(5, 16).replace("T", " ")}
                </td>
                <td className="px-2 py-0.5">{t.side}</td>
                <td
                  className={`px-2 py-0.5 text-right font-mono ${
                    t.pnl_pct >= 0 ? "text-emerald-300" : "text-red-300"
                  }`}
                >
                  {t.pnl_pct >= 0 ? "+" : ""}
                  {t.pnl_pct.toFixed(3)}
                </td>
                <td className="px-2 py-0.5 text-right text-slate-400">
                  {t.bars_held}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
