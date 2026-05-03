import { useEffect, useMemo, useRef, useState } from "react";

import { useEventStream } from "@/state/realtime";

interface Trade {
  side: "BUY" | "SELL" | string;
  price: number;
  size: number;
  venue?: string;
}

interface FlowEvent {
  id: string;
  kind: "SWEEP" | "ICEBERG" | "BLOCK";
  side: "BUY" | "SELL";
  price: number;
  total_size: number;
  ts: number;
  detail: string;
}

/**
 * Tier-2 order-flow widget — Sweep / Iceberg / Block monitor.
 *
 * Heuristic detector running purely in the client:
 *   - SWEEP   : ≥3 same-side trades within 750 ms summing to ≥3× the
 *               recent average size
 *   - ICEBERG : ≥4 same-side trades at the same price level within
 *               2 s, individually within 1.4× the recent average
 *               (refilled liquidity)
 *   - BLOCK   : single trade ≥6× the recent average size
 *
 * These are detection signals only — the widget surfaces them to the
 * operator; canonical detection lives in `system_engine` once the
 * server-side detector ships.
 */
const ROLLING_AVG_N = 80;
const SWEEP_WINDOW_MS = 750;
const SWEEP_TRADE_COUNT = 3;
const SWEEP_MULT = 3;
const ICEBERG_WINDOW_MS = 2000;
const ICEBERG_TRADE_COUNT = 4;
const ICEBERG_MAX_MULT = 1.4;
const BLOCK_MULT = 6;
const KEEP_EVENTS = 30;

export function SweepIcebergMonitor() {
  const trades = useEventStream<Trade>("ticks", [], 200);
  const [events, setEvents] = useState<FlowEvent[]>([]);
  const prevTrades = useRef<Trade[]>([]);
  const tradeBuffer = useRef<{ trade: Trade; ts: number }[]>([]);

  // rolling average size — derived from the trade window
  const avgSize = useMemo(() => {
    if (trades.length === 0) return 1;
    const sample = trades.slice(-ROLLING_AVG_N);
    return sample.reduce((s, t) => s + t.size, 0) / sample.length;
  }, [trades]);

  useEffect(() => {
    // Identify newly-arrived trades by reference. `useEventStream` slides a
    // fixed-size window so `trades.length` plateaus at the cap once full —
    // tracking by index is unsafe. Instead, find the position in `trades`
    // immediately after the last item we saw on the previous render.
    const prev = prevTrades.current;
    prevTrades.current = trades;
    const lastSeen = prev.length > 0 ? prev[prev.length - 1] : undefined;
    let startIdx = 0;
    if (lastSeen !== undefined) {
      const found = trades.lastIndexOf(lastSeen);
      startIdx = found >= 0 ? found + 1 : 0;
    }
    if (startIdx >= trades.length) return;
    const now = Date.now();
    const fresh = trades.slice(startIdx).map((trade) => ({
      trade,
      ts: now,
    }));
    tradeBuffer.current = [...tradeBuffer.current, ...fresh].filter(
      (e) => now - e.ts < 5000,
    );

    const detected: FlowEvent[] = [];
    for (const incoming of fresh) {
      const t = incoming.trade;
      const side = (String(t.side).toUpperCase() === "BUY"
        ? "BUY"
        : "SELL") as FlowEvent["side"];

      // BLOCK
      if (t.size >= avgSize * BLOCK_MULT) {
        detected.push({
          id: `${incoming.ts}-${detected.length}`,
          kind: "BLOCK",
          side,
          price: t.price,
          total_size: t.size,
          ts: incoming.ts,
          detail: `single ${t.size} ≥ ${BLOCK_MULT}× avg ${avgSize.toFixed(0)}`,
        });
        continue;
      }

      // SWEEP — same-side burst within window
      const window = tradeBuffer.current.filter(
        (e) =>
          incoming.ts - e.ts < SWEEP_WINDOW_MS &&
          String(e.trade.side).toUpperCase() === side,
      );
      const totalWindow = window.reduce((s, e) => s + e.trade.size, 0);
      if (
        window.length >= SWEEP_TRADE_COUNT &&
        totalWindow >= avgSize * SWEEP_MULT
      ) {
        detected.push({
          id: `${incoming.ts}-${detected.length}`,
          kind: "SWEEP",
          side,
          price: t.price,
          total_size: totalWindow,
          ts: incoming.ts,
          detail: `${window.length} same-side trades in ${SWEEP_WINDOW_MS}ms · Σ ${totalWindow}`,
        });
      }

      // ICEBERG — same price level repeatedly hit, sizes ≈ avg
      const samePrice = tradeBuffer.current.filter(
        (e) =>
          incoming.ts - e.ts < ICEBERG_WINDOW_MS &&
          Math.abs(e.trade.price - t.price) < 0.01 &&
          String(e.trade.side).toUpperCase() === side &&
          e.trade.size <= avgSize * ICEBERG_MAX_MULT,
      );
      if (samePrice.length >= ICEBERG_TRADE_COUNT) {
        detected.push({
          id: `${incoming.ts}-${detected.length}`,
          kind: "ICEBERG",
          side,
          price: t.price,
          total_size: samePrice.reduce((s, e) => s + e.trade.size, 0),
          ts: incoming.ts,
          detail: `${samePrice.length} refills @ ${t.price.toFixed(4)} in ${ICEBERG_WINDOW_MS}ms`,
        });
      }
    }

    if (detected.length > 0) {
      setEvents((prev) => [...detected.reverse(), ...prev].slice(0, KEEP_EVENTS));
    }
  }, [trades, avgSize]);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Sweep / Iceberg / Block
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          rolling avg size {avgSize.toFixed(0)} · {events.length} detected
        </p>
      </header>
      <ul className="flex-1 overflow-auto divide-y divide-border/40 text-[11px]">
        {events.length === 0 && (
          <li className="px-3 py-2 text-slate-500">
            no flow anomalies in window
          </li>
        )}
        {events.map((e) => (
          <li key={e.id} className="flex items-baseline gap-2 px-3 py-1">
            <KindChip kind={e.kind} />
            <SideChip side={e.side} />
            <span className="font-mono text-slate-300">
              {e.price.toFixed(4)}
            </span>
            <span className="font-mono text-slate-400">{e.total_size}</span>
            <span className="ml-auto text-[10px] text-slate-500">
              {e.detail}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function KindChip({ kind }: { kind: FlowEvent["kind"] }) {
  const cls =
    kind === "SWEEP"
      ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
      : kind === "ICEBERG"
        ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
        : "border-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-300";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider ${cls}`}
    >
      {kind}
    </span>
  );
}

function SideChip({ side }: { side: FlowEvent["side"] }) {
  const cls =
    side === "BUY"
      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
      : "border-rose-500/40 bg-rose-500/10 text-rose-300";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider ${cls}`}
    >
      {side}
    </span>
  );
}
