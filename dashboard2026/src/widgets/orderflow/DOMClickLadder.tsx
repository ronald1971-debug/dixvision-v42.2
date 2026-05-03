import { useRef, useState } from "react";

import { useLatestEvent } from "@/state/realtime";

interface DepthLevel {
  price: number;
  size: number;
}
interface DepthSnapshot {
  bids: DepthLevel[];
  asks: DepthLevel[];
  mid: number;
}

type Side = "BUY" | "SELL";

interface PendingOrder {
  id: string;
  side: Side;
  price: number;
  size: number;
  ts: number;
  status: "STAGED" | "BLOCKED";
  reason: string;
}

/**
 * Tier-2 order-flow widget — DOM-click ladder.
 *
 * Click any price level to stage a limit order at that price; the
 * default size is configurable in the header. Orders go to the
 * STAGED tray — the operator-approval edge (INV-72) is the
 * authoritative gate before execution. This widget never sends
 * orders directly to the execution engine; it stages intent.
 */
const DEFAULT_SIZE = 100;

export function DOMClickLadder({ symbol = "BTC-USDT" }: { symbol?: string }) {
  const snapshot = useLatestEvent<DepthSnapshot>("depth");
  const [size, setSize] = useState(DEFAULT_SIZE);
  const [pending, setPending] = useState<PendingOrder[]>([]);
  const seq = useRef(0);

  const bids = snapshot?.bids ?? [];
  const asks = snapshot?.asks ?? [];
  const maxLvl = Math.max(
    1,
    ...bids.map((l) => l.size),
    ...asks.map((l) => l.size),
  );

  const stage = (side: Side, price: number) => {
    const ts = Date.now();
    seq.current += 1;
    const id = `${ts}-${seq.current}`;
    if (size <= 0) {
      const blocked: PendingOrder = {
        id,
        side,
        price,
        size,
        ts,
        status: "BLOCKED",
        reason: "size must be > 0",
      };
      setPending((p) => [blocked, ...p].slice(0, 12));
      return;
    }
    const staged: PendingOrder = {
      id,
      side,
      price,
      size,
      ts,
      status: "STAGED",
      reason: "awaits operator-approval edge",
    };
    setPending((p) => [staged, ...p].slice(0, 12));
  };

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            DOM ladder · {symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            click a price to stage a limit · approval edge gates execution
          </p>
        </div>
        <label className="flex items-baseline gap-1.5 font-mono text-[10px] text-slate-300">
          <span className="text-slate-500">size</span>
          <input
            type="number"
            value={size}
            min={1}
            onChange={(e) => setSize(Number(e.target.value) || 0)}
            className="w-20 rounded border border-border bg-bg/40 px-1.5 py-0.5 text-right text-slate-200 focus:border-accent focus:outline-none"
          />
        </label>
      </header>
      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-auto border-r border-border">
          <div className="sticky top-0 z-10 border-b border-border bg-surface px-2 py-1 text-[9px] uppercase tracking-wider text-emerald-400">
            BIDS · click to BUY
          </div>
          {bids.length === 0 ? (
            <Empty label="bids" />
          ) : (
            bids.map((lvl) => (
              <Row
                key={`b-${lvl.price}`}
                lvl={lvl}
                maxLvl={maxLvl}
                side="BUY"
                onClick={() => stage("BUY", lvl.price)}
              />
            ))
          )}
        </div>
        <div className="flex-1 overflow-auto">
          <div className="sticky top-0 z-10 border-b border-border bg-surface px-2 py-1 text-right text-[9px] uppercase tracking-wider text-rose-400">
            ASKS · click to SELL
          </div>
          {asks.length === 0 ? (
            <Empty label="asks" />
          ) : (
            asks.map((lvl) => (
              <Row
                key={`a-${lvl.price}`}
                lvl={lvl}
                maxLvl={maxLvl}
                side="SELL"
                onClick={() => stage("SELL", lvl.price)}
              />
            ))
          )}
        </div>
      </div>
      <footer className="max-h-32 overflow-auto border-t border-border bg-bg/40 text-[10px]">
        <div className="border-b border-border px-3 py-1 text-[9px] uppercase tracking-wider text-slate-500">
          staged orders · {pending.length}
        </div>
        {pending.length === 0 ? (
          <p className="px-3 py-1 text-slate-500">
            no staged orders — click a level
          </p>
        ) : (
          <ul className="divide-y divide-border/40 font-mono">
            {pending.map((o) => (
              <li
                key={o.id}
                className="flex items-baseline gap-2 px-3 py-0.5"
              >
                <span
                  className={
                    o.side === "BUY"
                      ? "text-emerald-300"
                      : "text-rose-300"
                  }
                >
                  {o.side}
                </span>
                <span className="text-slate-300">{o.price.toFixed(4)}</span>
                <span className="text-slate-400">× {o.size}</span>
                <span
                  className={`ml-auto rounded border px-1 py-0.5 text-[9px] ${
                    o.status === "STAGED"
                      ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
                      : "border-rose-500/40 bg-rose-500/10 text-rose-300"
                  }`}
                >
                  {o.status}
                </span>
                <span className="text-slate-500">{o.reason}</span>
              </li>
            ))}
          </ul>
        )}
      </footer>
    </section>
  );
}

function Row({
  lvl,
  maxLvl,
  side,
  onClick,
}: {
  lvl: DepthLevel;
  maxLvl: number;
  side: Side;
  onClick: () => void;
}) {
  const fill = Math.min(100, (lvl.size / maxLvl) * 100);
  const tint =
    side === "BUY" ? "rgba(61, 220, 132, 0.18)" : "rgba(255, 90, 90, 0.18)";
  const fg = side === "BUY" ? "text-emerald-300" : "text-rose-300";
  const align = side === "BUY" ? "to left" : "to right";
  return (
    <button
      type="button"
      onClick={onClick}
      className="relative flex w-full items-center justify-between px-2 py-0.5 text-left font-mono text-[11px] transition hover:bg-white/5"
      style={{
        backgroundImage: `linear-gradient(${align}, ${tint} ${fill}%, transparent ${fill}%)`,
      }}
    >
      <span className={fg}>{lvl.price.toFixed(4)}</span>
      <span className="text-slate-300">{lvl.size}</span>
    </button>
  );
}

function Empty({ label }: { label: string }) {
  return (
    <div className="grid h-full place-items-center text-[11px] text-slate-600">
      no {label} (waiting for SSE bridge)
    </div>
  );
}
