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

const SYMBOL_RE = /\b([A-Z]{2,5})(?:[-/]?(?:USDT|USD|USDC))?\b/g;
const PAIRED_SYMBOL_RE = /\b([A-Z]{2,5})[-/](?:USDT|USD|USDC)\b/;
const CONTEXTUAL_SYMBOL_RE = /\b(?:on|of|for|in)\s+([A-Z]{2,5})\b/;
const SIZE_RE = /\b(\d+(?:\.\d+)?)\s?(USDT|USD|coins|shares|sol|btc|eth)?\b/i;

// Common uppercase tokens that look like tickers but are technical
// indicators, generic acronyms, or units. These are skipped when the
// regex first-matches them, so the parser falls through to the next
// candidate or to a contextual / paired match instead.
const NON_SYMBOL_TOKENS: ReadonlySet<string> = new Set([
  "RSI",
  "MACD",
  "CVD",
  "ATR",
  "ADX",
  "VWAP",
  "EMA",
  "SMA",
  "OBV",
  "ETF",
  "ATH",
  "ATL",
  "BLS",
  "CPI",
  "PPI",
  "GDP",
  "FOMC",
  "FED",
  "ECB",
  "BOJ",
  "OI",
  "TVL",
  "PnL",
  "PNL",
  "TP",
  "SL",
  "DCA",
  "TWAP",
  "POV",
  // Action verbs that parse() itself recognises — needed because the
  // operator may type all-caps in a trading terminal (e.g. "BUY AAPL").
  "BUY",
  "SELL",
  "SHORT",
  "LONG",
  "ALERT",
  "CLOSE",
  "RUN",
  "NOTIFY",
]);

function extractSymbol(raw: string): string | undefined {
  // Prefer an explicit trading pair (e.g. ETH-USDT, BTC/USD) — the most
  // unambiguous signal that a token is a real symbol.
  const paired = raw.match(PAIRED_SYMBOL_RE);
  if (paired) return paired[1];

  // Next prefer a contextual cue ("... on ETH", "... of SOL").
  const contextual = raw.match(CONTEXTUAL_SYMBOL_RE);
  if (contextual && !NON_SYMBOL_TOKENS.has(contextual[1])) {
    return contextual[1];
  }

  // Finally fall back to the first uppercase 2-5 letter token that
  // is not a known non-symbol acronym (RSI, CVD, ETF, ...).
  for (const m of raw.matchAll(SYMBOL_RE)) {
    const candidate = m[1];
    if (!NON_SYMBOL_TOKENS.has(candidate)) return candidate;
  }
  return undefined;
}

function parse(raw: string): ParsedIntent {
  const lc = raw.toLowerCase();
  const symbol = extractSymbol(raw);

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
  let triggerMatch: RegExpMatchArray | null = null;
  const dropMatch = lc.match(/(?:drops? below|under)\s*\$?\s?([\d.]+)/);
  const riseMatch = lc.match(/(?:rises? above|over)\s*\$?\s?([\d.]+)/);
  if (dropMatch) {
    trigger = `< ${dropMatch[1]}`;
    triggerMatch = dropMatch;
  } else if (riseMatch) {
    trigger = `> ${riseMatch[1]}`;
    triggerMatch = riseMatch;
  } else {
    // Anchor the price extraction to the number that immediately
    // follows "at" — otherwise PRICE_RE greedily matches the first
    // number in the string, which for "Short SOL 25 coins at 178"
    // is the size (25), not the price (178).
    const atMatch = lc.match(/\bat\s+\$?\s?([\d.]+)/);
    if (atMatch) {
      trigger = `@ ${atMatch[1]}`;
      triggerMatch = atMatch;
    }
  }

  // Strip the trigger-matched substring before extracting size, so that
  // the trigger price ("under $170") is not re-captured as a trade size.
  // SIZE_RE runs against the original (mixed-case) raw string; the
  // trigger match indices are in `lc`, but `lc` and `raw` are the same
  // length (toLowerCase preserves indices), so the slice indices align.
  let sizeSource = raw;
  if (triggerMatch && triggerMatch.index !== undefined) {
    const start = triggerMatch.index;
    const end = start + triggerMatch[0].length;
    sizeSource = raw.slice(0, start) + raw.slice(end);
  }
  // Trade size is meaningful only for BUY/SELL intents. For ALERT and
  // STRATEGY, any incidental number (e.g. "last 30 days") is not a size.
  const sz = action === "BUY" || action === "SELL" ? sizeSource.match(SIZE_RE) : null;
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
