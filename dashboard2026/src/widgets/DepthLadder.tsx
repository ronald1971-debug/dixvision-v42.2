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

export function DepthLadder({ symbol }: { symbol: string }) {
  const snapshot = useLatestEvent<DepthSnapshot>("depth");
  const bids = snapshot?.bids ?? [];
  const asks = snapshot?.asks ?? [];
  const maxSize = Math.max(
    1,
    ...bids.map((b) => b.size),
    ...asks.map((a) => a.size),
  );
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Depth · {symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            L2 ladder · per-venue toggle pending wiring
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          live
        </span>
      </header>
      <div className="grid flex-1 grid-cols-2 overflow-hidden text-[11px]">
        <div className="overflow-auto border-r border-border">
          {bids.map((lvl, i) => (
            <Row
              key={`bid-${i}`}
              level={lvl}
              maxSize={maxSize}
              side="bid"
            />
          ))}
          {bids.length === 0 && <Empty label="bids" />}
        </div>
        <div className="overflow-auto">
          {asks.map((lvl, i) => (
            <Row
              key={`ask-${i}`}
              level={lvl}
              maxSize={maxSize}
              side="ask"
            />
          ))}
          {asks.length === 0 && <Empty label="asks" />}
        </div>
      </div>
      <footer className="border-t border-border px-3 py-1 font-mono text-[11px] text-slate-400">
        mid {snapshot?.mid?.toFixed(4) ?? "—"}
      </footer>
    </div>
  );
}

function Row({
  level,
  maxSize,
  side,
}: {
  level: DepthLevel;
  maxSize: number;
  side: "bid" | "ask";
}) {
  const fill = Math.min(100, Math.round((level.size / maxSize) * 100));
  const tint = side === "bid" ? "rgba(61,220,132,0.18)" : "rgba(255,90,90,0.18)";
  const fg = side === "bid" ? "text-emerald-300" : "text-red-300";
  return (
    <div
      className="relative flex items-center justify-between px-2 py-0.5 font-mono"
      style={{
        backgroundImage: `linear-gradient(${
          side === "bid" ? "to left" : "to right"
        }, ${tint} ${fill}%, transparent ${fill}%)`,
      }}
    >
      <span className={fg}>{level.price.toFixed(4)}</span>
      <span className="text-slate-300">{level.size}</span>
    </div>
  );
}

function Empty({ label }: { label: string }) {
  return (
    <div className="grid h-full place-items-center text-[11px] text-slate-600">
      no {label} (waiting for SSE bridge)
    </div>
  );
}
