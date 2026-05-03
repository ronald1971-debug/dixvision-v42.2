import { useEffect, useState } from "react";

import { Activity } from "lucide-react";

/**
 * Tier-8 testing widget — Forward (paper) tester.
 *
 * The 30-day SHADOW countdown that gates SHADOW → CANARY promotion.
 * Same engine as live execution, but `mode_effect_table` blocks broker
 * dispatch — only fills are simulated against the live tape so the
 * strategy's edge is measured under real microstructure.
 *
 * Live wiring sources:
 *   - Days elapsed / total: `governance_engine.strategy_registry.get_window`
 *   - Live Sharpe / floor: `evaluation.shadow_aggregator.live_metrics`
 *   - Hazard tally / ceiling: `system_engine.hazard_monitor.window_count`
 *   - Pass / fail badge: drift_oracle composite + promotion-gates rule
 *
 * Today the panel renders a deterministic mock that drifts forward so
 * the operator can see the FSM live; once `/api/governance/shadow_status`
 * lands, replace the timer + mock with `useSWR(...)` against it.
 */
interface ShadowMetrics {
  days_elapsed: number;
  days_total: number;
  live_sharpe: number;
  sharpe_floor: number;
  hazard_count: number;
  hazard_ceiling: number;
  fill_rate: number;
  win_rate: number;
  avg_slippage_bps: number;
  trades: number;
}

const SEED: ShadowMetrics = {
  days_elapsed: 18,
  days_total: 30,
  live_sharpe: 1.42,
  sharpe_floor: 0.9,
  hazard_count: 3,
  hazard_ceiling: 8,
  fill_rate: 0.94,
  win_rate: 0.58,
  avg_slippage_bps: 4.6,
  trades: 1248,
};

function fmtPct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function tone(passing: boolean) {
  return passing
    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
    : "border-rose-500/40 bg-rose-500/10 text-rose-300";
}

export function ForwardTester() {
  const [m, setM] = useState<ShadowMetrics>(SEED);

  useEffect(() => {
    const id = setInterval(() => {
      setM((prev) => {
        const drift = Math.sin(Date.now() / 8_000) * 0.04;
        const next: ShadowMetrics = {
          ...prev,
          live_sharpe: Math.max(0, prev.live_sharpe + drift * 0.05),
          fill_rate: Math.min(0.999, Math.max(0.7, prev.fill_rate + drift * 0.005)),
          win_rate: Math.min(0.99, Math.max(0.4, prev.win_rate + drift * 0.003)),
          avg_slippage_bps: Math.max(0.5, prev.avg_slippage_bps + drift * 0.2),
        };
        return next;
      });
    }, 4_000);
    return () => clearInterval(id);
  }, []);

  const sharpePass = m.live_sharpe >= m.sharpe_floor;
  const hazardPass = m.hazard_count <= m.hazard_ceiling;
  const dayPass = m.days_elapsed >= m.days_total;
  const wouldPromote = sharpePass && hazardPass && dayPass;
  const pct = Math.min(1, m.days_elapsed / m.days_total);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 text-accent" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Forward (paper) tester
          </h3>
        </div>
        <span className={`rounded border px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest ${tone(wouldPromote)}`}>
          {wouldPromote ? "would promote" : "blocked"}
        </span>
      </header>
      <div className="flex flex-col gap-3 p-3">
        <div>
          <div className="mb-1 flex items-baseline justify-between text-[10px] uppercase tracking-widest text-slate-500">
            <span>SHADOW window</span>
            <span className="font-mono text-slate-300">
              {m.days_elapsed}/{m.days_total} days
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded bg-bg">
            <div
              className="h-full bg-accent"
              style={{ width: `${pct * 100}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs md:grid-cols-3">
          <Metric
            label="Live Sharpe"
            value={m.live_sharpe.toFixed(2)}
            sub={`floor ${m.sharpe_floor.toFixed(2)}`}
            ok={sharpePass}
          />
          <Metric
            label="Hazards"
            value={String(m.hazard_count)}
            sub={`ceil ${m.hazard_ceiling}`}
            ok={hazardPass}
          />
          <Metric label="Trades" value={String(m.trades)} />
          <Metric label="Win rate" value={fmtPct(m.win_rate)} />
          <Metric label="Fill rate" value={fmtPct(m.fill_rate)} />
          <Metric label="Slippage" value={`${m.avg_slippage_bps.toFixed(1)} bps`} />
        </div>

        <p className="border-t border-border pt-2 text-[10px] leading-relaxed text-slate-500">
          Promotion is gated by all three rails — Sharpe ≥ floor,
          hazards ≤ ceiling, and full window elapsed. The drift oracle
          adds a fourth continuous-AUTO gate.
        </p>
      </div>
    </section>
  );
}

interface MetricProps {
  label: string;
  value: string;
  sub?: string;
  ok?: boolean;
}

function Metric({ label, value, sub, ok }: MetricProps) {
  const okTone =
    ok === undefined
      ? "text-slate-100"
      : ok
        ? "text-emerald-300"
        : "text-rose-300";
  return (
    <div className="rounded border border-border bg-bg px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-widest text-slate-500">
        {label}
      </div>
      <div className={`font-mono text-base ${okTone}`}>{value}</div>
      {sub && (
        <div className="font-mono text-[10px] text-slate-500">{sub}</div>
      )}
    </div>
  );
}
