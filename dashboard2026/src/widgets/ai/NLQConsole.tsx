import { useState } from "react";

/**
 * Tier-3 AI widget — Natural-language query console.
 *
 * Operator types intent in plain English; a deterministic parser
 * extracts a structured `ParsedIntent` (action / symbol / side /
 * trigger / size). The intent is staged — it never reaches
 * `execution_engine` directly. The operator-approval edge (INV-72)
 * is the authoritative gate before any execution.
 *
 * Real NLP is delegated to Indira via the registry-driven chat
 * adapter (PR #82). This widget surfaces what the parser saw and
 * lets the operator confirm or reject before approval-edge dispatch.
 */
type Action = "BUY" | "SELL" | "ALERT" | "STRATEGY" | "UNPARSED";

interface ParsedIntent {
  raw: string;
  action: Action;
  symbol?: string;
  side?: "BUY" | "SELL";
  trigger?: string;
  size?: string;
  reason: string;
}

const SYMBOL_RE = /\b([A-Z]{2,5})(?:[-/]?(?:USDT|USD|USDC))?\b/;
const PRICE_RE = /\$?\s?([0-9]{1,7}(?:\.[0-9]+)?)\b/;
const SIZE_RE = /\b(\d+(?:\.\d+)?)\s?(USDT|USD|coins|shares|sol|btc|eth)?\b/i;

function parse(raw: string): ParsedIntent {
  const lc = raw.toLowerCase();
  const symMatch = raw.match(SYMBOL_RE);
  const symbol = symMatch ? symMatch[1] : undefined;

  let action: Action = "UNPARSED";
  let side: "BUY" | "SELL" | undefined;
  if (/\bbuy\b|\blong\b/.test(lc)) {
    action = "BUY";
    side = "BUY";
  } else if (/\bsell\b|\bshort\b|\bclose\b/.test(lc)) {
    action = "SELL";
    side = "SELL";
  } else if (/\balert\b|\bnotify\b|\btell me\b/.test(lc)) {
    action = "ALERT";
  } else if (/\bstrategy\b|\bbacktest\b|\bforward test\b/.test(lc)) {
    action = "STRATEGY";
  }

  let trigger: string | undefined;
  const dropMatch = lc.match(/(?:drops? below|under)\s*\$?\s?([\d.]+)/);
  const riseMatch = lc.match(/(?:rises? above|over)\s*\$?\s?([\d.]+)/);
  if (dropMatch) trigger = `< ${dropMatch[1]}`;
  else if (riseMatch) trigger = `> ${riseMatch[1]}`;
  else {
    const p = raw.match(PRICE_RE);
    if (p && /\bat\b/.test(lc)) trigger = `@ ${p[1]}`;
  }

  const sz = raw.match(SIZE_RE);
  const size = sz ? `${sz[1]}${sz[2] ? " " + sz[2] : ""}` : undefined;

  let reason = "parsed locally · awaiting Indira semantic refinement";
  if (action === "UNPARSED")
    reason = "no actionable verb detected · please rephrase";
  else if (!symbol)
    reason = "verb detected but no symbol · add a ticker";

  return { raw, action, symbol, side, trigger, size, reason };
}

const EXAMPLES = [
  "Buy AAPL if it drops below $170 with 5% stop",
  "Short SOL 25 coins at 178",
  "Alert me when BTC rises above 70000",
  "Run backtest of CVD-divergence strategy on ETH last 30 days",
];

export function NLQConsole() {
  const [text, setText] = useState("");
  const [staged, setStaged] = useState<ParsedIntent[]>([]);
  const submit = () => {
    if (!text.trim()) return;
    const intent = parse(text);
    setStaged((prev) => [intent, ...prev].slice(0, 8));
    setText("");
  };

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          NLQ console
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          natural-language → structured intent · staged · approval-edge gates
          execution
        </p>
      </header>
      <div className="flex flex-1 flex-col gap-2 overflow-auto p-3 text-[12px]">
        <div className="flex gap-2">
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            placeholder="Buy AAPL if it drops below $170 with 5% stop"
            className="flex-1 rounded border border-border bg-bg/40 px-2 py-1 font-mono text-[11px] text-slate-200 focus:border-accent focus:outline-none"
          />
          <button
            type="button"
            onClick={submit}
            className="rounded border border-accent/40 bg-accent/10 px-2 py-1 text-[11px] uppercase tracking-wider text-accent hover:bg-accent/20"
          >
            Stage
          </button>
        </div>
        <div className="flex flex-wrap gap-1">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => setText(ex)}
              className="rounded border border-border bg-bg/40 px-2 py-0.5 text-[10px] text-slate-400 hover:border-accent hover:text-accent"
            >
              {ex}
            </button>
          ))}
        </div>
        <div>
          <h4 className="mb-1 font-mono text-[10px] uppercase tracking-wider text-slate-500">
            staged intents · {staged.length}
          </h4>
          {staged.length === 0 ? (
            <p className="text-[11px] text-slate-500">no staged intents yet</p>
          ) : (
            <ul className="divide-y divide-border/40 rounded border border-border">
              {staged.map((s, i) => (
                <li
                  key={`${s.raw}-${i}`}
                  className="px-2 py-1.5 font-mono text-[11px] text-slate-300"
                >
                  <div className="flex items-baseline justify-between">
                    <span
                      className={
                        s.action === "UNPARSED"
                          ? "text-rose-400"
                          : "text-accent"
                      }
                    >
                      {s.action}
                      {s.symbol ? ` · ${s.symbol}` : ""}
                      {s.side ? ` · ${s.side}` : ""}
                      {s.trigger ? ` · ${s.trigger}` : ""}
                      {s.size ? ` · ${s.size}` : ""}
                    </span>
                    <span className="text-[10px] uppercase text-slate-500">
                      staged
                    </span>
                  </div>
                  <p className="mt-1 text-[10px] text-slate-500">"{s.raw}"</p>
                  <p className="text-[10px] text-slate-500">{s.reason}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
