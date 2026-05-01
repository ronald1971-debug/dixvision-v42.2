import { useState } from "react";

/**
 * Indira chat (Widget #7 — Trading Intelligence) per PR-#2 spec §0.
 *
 * Conversational interface for natural-language interaction with
 * Indira. Supports:
 *   - "Explain this signal" — pulls latest DecisionTrace
 *   - "Propose trade with risk parameters" — emits proposal through
 *     operator-approval edge
 *   - "Counterfactual: what if I had X" — forks BeliefState
 *   - "Adjust constraints for this memecoin" — validates against
 *     ConstraintEngine before suggesting
 *
 * Every proposal is routed through Governance approval edge before
 * any side-effecting action lands. Responses include structured
 * "Why" layer tied to BeliefState + DecisionTrace.
 */
interface ChatMessage {
  id: string;
  role: "operator" | "indira" | "governance";
  text: string;
  ts_iso: string;
  trace_id?: string;
}

const SUGGESTIONS = [
  "Explain the last signal",
  "Propose a trade for SOL/USDC",
  "Counterfactual: what if I had not exited at TP1?",
  "Adjust SL for current memecoin position",
];

export function IndiraChat() {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);

  async function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed || pending) return;
    const operatorMsg: ChatMessage = {
      id: `op-${Date.now()}`,
      role: "operator",
      text: trimmed,
      ts_iso: new Date().toISOString(),
    };
    setHistory((h) => [...h, operatorMsg]);
    setDraft("");
    setPending(true);
    try {
      const res = await fetch("/api/cognitive/chat/turn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: "indira", text: trimmed }),
      });
      const body = (await res.json()) as {
        text?: string;
        trace_id?: string;
        approval_required?: boolean;
      };
      const indiraMsg: ChatMessage = {
        id: `id-${Date.now()}`,
        role: "indira",
        text:
          body.text ??
          "(no response — backend may not be wired; check `/api/cognitive/chat/status`)",
        ts_iso: new Date().toISOString(),
        trace_id: body.trace_id,
      };
      setHistory((h) => [...h, indiraMsg]);
      if (body.approval_required) {
        setHistory((h) => [
          ...h,
          {
            id: `gov-${Date.now()}`,
            role: "governance",
            text: "Approval gate engaged — proposal pending operator click on Operator tab.",
            ts_iso: new Date().toISOString(),
          },
        ]);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setHistory((h) => [
        ...h,
        {
          id: `err-${Date.now()}`,
          role: "indira",
          text: `(network error — ${msg})`,
          ts_iso: new Date().toISOString(),
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface text-sm">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Indira · Trading Intelligence
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            #7 · BeliefState + DecisionTrace · proposals routed through
            operator-approval edge
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          PR-#2 §0
        </span>
      </header>
      <div className="flex-1 space-y-2 overflow-auto p-3">
        {history.length === 0 && (
          <div className="grid h-full place-items-center text-center text-xs text-slate-500">
            <div className="space-y-2">
              <div className="font-mono uppercase tracking-wider">
                ask Indira
              </div>
              <div className="flex flex-col gap-1">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => send(s)}
                    className="rounded border border-border bg-bg px-2 py-1 text-left text-[11px] text-slate-300 hover:text-accent"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        {history.map((m) => (
          <ChatBubble key={m.id} message={m} />
        ))}
      </div>
      <form
        className="flex items-center gap-1 border-t border-border p-2"
        onSubmit={(e) => {
          e.preventDefault();
          void send(draft);
        }}
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask Indira a trading-intelligence question…"
          className="flex-1 rounded border border-border bg-bg px-2 py-1 text-[12px] focus:border-accent focus:outline-none"
        />
        <button
          type="submit"
          disabled={pending}
          className="rounded border border-accent/60 bg-accent/15 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-accent disabled:opacity-50"
        >
          send
        </button>
      </form>
    </div>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const tone =
    message.role === "operator"
      ? "border-accent/40 bg-accent/10 text-slate-100"
      : message.role === "indira"
        ? "border-emerald-500/40 bg-emerald-500/10 text-slate-100"
        : "border-amber-500/40 bg-amber-500/10 text-amber-200";
  return (
    <div className={`rounded border px-2 py-1.5 ${tone}`}>
      <div className="flex items-baseline justify-between font-mono text-[10px] uppercase tracking-wider opacity-70">
        <span>{message.role}</span>
        <span>
          {new Date(message.ts_iso).toLocaleTimeString()}
          {message.trace_id ? ` · ${message.trace_id.slice(0, 8)}` : ""}
        </span>
      </div>
      <div className="whitespace-pre-wrap text-[12px]">{message.text}</div>
    </div>
  );
}
