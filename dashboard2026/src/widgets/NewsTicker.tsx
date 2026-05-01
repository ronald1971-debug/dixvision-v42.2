import { useEventStream } from "@/state/realtime";

interface NewsItem {
  source: string;
  title: string;
  sentiment: number;
}

export function NewsTicker() {
  const items = useEventStream<NewsItem>("news", [], 30);
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            News + Sentiment
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            CoinDesk live + sentiment scorer · X/TG hooks pending Wave-05
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          live
        </span>
      </header>
      <div className="flex-1 overflow-auto">
        <ul className="divide-y divide-border">
          {[...items].reverse().map((n, i) => (
            <li
              key={i}
              className="flex items-baseline gap-2 px-3 py-1.5 text-[12px]"
            >
              <SentimentDot value={n.sentiment} />
              <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
                {n.source}
              </span>
              <span className="flex-1 text-slate-200">{n.title}</span>
              <span className="font-mono text-[11px] text-slate-400">
                {n.sentiment >= 0 ? "+" : ""}
                {n.sentiment.toFixed(2)}
              </span>
            </li>
          ))}
          {items.length === 0 && (
            <li className="px-3 py-3 text-center text-[11px] text-slate-600">
              waiting for news bus (CoinDesk · Wave-04.5)
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}

function SentimentDot({ value }: { value: number }) {
  const tone =
    value > 0.2
      ? "bg-emerald-400"
      : value < -0.2
        ? "bg-red-400"
        : "bg-amber-300";
  return <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${tone}`} />;
}
