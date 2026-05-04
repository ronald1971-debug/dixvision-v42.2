import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { fetchPumpFunRecent } from "@/api/feeds";
import { Panel } from "@/components/Panel";
import { PriceChart, type PricePoint } from "@/components/PriceChart";

const STORAGE_KEY = "dixmeme.multichart.symbols";
const DEFAULT_SYMBOLS = ["BONK/SOL", "WIF/SOL", "POPCAT/SOL", "MOG/SOL"];

function readStored(): string[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.every((x) => typeof x === "string")) {
        return parsed.slice(0, 4);
      }
    }
  } catch {
    // ignore
  }
  return [...DEFAULT_SYMBOLS];
}

function pickPrice(o: Record<string, unknown>): number | null {
  for (const k of ["price", "price_usd", "last"]) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}

function pickTs(o: Record<string, unknown>): number {
  for (const k of ["ts", "time", "timestamp"]) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) {
      return v > 1e12 ? v : v * 1000;
    }
  }
  return Date.now();
}

function pickSym(o: Record<string, unknown>): string {
  for (const k of ["symbol", "ticker", "name", "mint"]) {
    const v = o[k];
    if (typeof v === "string" && v) return v;
  }
  return "";
}

export function MultichartPage() {
  const [symbols, setSymbols] = useState<string[]>(readStored);
  const buffers = useRef<Record<string, PricePoint[]>>({});
  const [tick, setTick] = useState(0);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(symbols));
    } catch {
      // ignore
    }
  }, [symbols]);

  const q = useQuery({
    queryKey: ["pumpfun", "recent"],
    queryFn: fetchPumpFunRecent,
    refetchInterval: 2_500,
  });

  useEffect(() => {
    const recent = q.data?.items ?? [];
    // Group incoming points by symbol first, then merge+dedup each
    // symbol's buffer once at the end (Devin Review BUG_0002 on PR
    // #181 — overlapping 2.5s refetches were padding the per-symbol
    // 300-point caps with duplicates, shrinking effective history).
    const incoming: Record<string, PricePoint[]> = {};
    for (const r of recent) {
      const sym = pickSym(r);
      const price = pickPrice(r);
      if (!sym || price == null) continue;
      const matched = symbols.find(
        (s) => s.toUpperCase().split("/")[0] === sym.toUpperCase(),
      );
      if (!matched) continue;
      (incoming[matched] ??= []).push({ ts: pickTs(r), price });
    }
    let changed = false;
    for (const [matched, points] of Object.entries(incoming)) {
      const merged = [...(buffers.current[matched] ?? []), ...points].sort(
        (a, b) => a.ts - b.ts,
      );
      const seen = new Set<number>();
      const dedup: PricePoint[] = [];
      for (const p of merged) {
        if (!seen.has(p.ts)) {
          seen.add(p.ts);
          dedup.push(p);
        }
      }
      buffers.current[matched] = dedup.slice(-300);
      changed = true;
    }
    if (changed) setTick((t) => t + 1);
  }, [q.data, symbols]);

  return (
    <div className="grid h-full grid-cols-2 grid-rows-2 gap-2 p-2">
      {symbols.map((sym, i) => (
        <Panel
          key={i}
          title={sym}
          right={
            <input
              defaultValue={sym}
              onBlur={(e) => {
                const v = e.target.value.trim();
                if (!v || v === sym) return;
                const next = [...symbols];
                next[i] = v;
                setSymbols(next);
              }}
              className="h-6 w-32 rounded border border-border bg-surface-raised px-1 text-right font-mono text-[11px]"
            />
          }
          bodyClassName="p-2"
        >
          <PriceChart
            points={buffers.current[sym] ?? []}
            height={Math.max(200, Math.floor(window.innerHeight / 3))}
          />
          <div className="mt-1 text-right text-[10px] text-text-disabled">
            {(buffers.current[sym]?.length ?? 0)} pts · tick {tick}
          </div>
        </Panel>
      ))}
    </div>
  );
}
