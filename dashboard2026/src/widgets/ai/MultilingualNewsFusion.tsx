import { useEffect, useMemo, useState } from "react";

/**
 * Tier-3 / E-track AI widget — Multilingual news fusion.
 *
 * Fuses non-English news sources (Reuters JP, Nikkei, Caixin, Korea
 * Economic Daily, Handelsblatt, Les Échos) into one unified stream
 * alongside the CoinDesk EN feed (already wired through
 * ``CoinDeskRSSPump``). Each item carries a language tag, a
 * translated headline (server-side translation; mock here), and a
 * sentiment score the projection layer (PR #118) uses for damping.
 *
 * Backend hook: ``GET /api/news/multilingual`` reads from the same
 * ``NewsKnowledgeIndex`` that ``ui/server.STATE.news_index`` owns
 * (D4 wiring) once the ML translation gateway is provisioned.
 */
type Lang = "EN" | "JA" | "ZH" | "KO" | "DE" | "FR";

interface FusedItem {
  ts: number;
  source: string;
  lang: Lang;
  original: string;
  translated: string;
  sentiment: number;
}

const SEED: FusedItem[] = [
  {
    ts: Date.now() - 90_000,
    source: "REUTERS_JP",
    lang: "JA",
    original: "日銀、政策金利を据え置き",
    translated: "BOJ holds policy rate steady",
    sentiment: 0.05,
  },
  {
    ts: Date.now() - 130_000,
    source: "NIKKEI",
    lang: "JA",
    original: "ソニーグループ、AI投資を加速",
    translated: "Sony Group accelerates AI investment",
    sentiment: 0.32,
  },
  {
    ts: Date.now() - 200_000,
    source: "CAIXIN",
    lang: "ZH",
    original: "中国央行下调存款准备金率",
    translated: "PBOC cuts reserve requirement ratio",
    sentiment: 0.41,
  },
  {
    ts: Date.now() - 260_000,
    source: "KED",
    lang: "KO",
    original: "삼성전자, HBM4 양산 일정 발표",
    translated: "Samsung announces HBM4 mass-production schedule",
    sentiment: 0.27,
  },
  {
    ts: Date.now() - 340_000,
    source: "HANDELSBLATT",
    lang: "DE",
    original: "EZB-Sitzung: Lagarde signalisiert Pause",
    translated: "ECB meeting: Lagarde signals pause",
    sentiment: 0.0,
  },
  {
    ts: Date.now() - 410_000,
    source: "LES_ECHOS",
    lang: "FR",
    original: "TotalEnergies relève son dividende",
    translated: "TotalEnergies raises dividend",
    sentiment: 0.18,
  },
  {
    ts: Date.now() - 480_000,
    source: "COINDESK",
    lang: "EN",
    original: "Spot ETH ETF inflows hit record",
    translated: "Spot ETH ETF inflows hit record",
    sentiment: 0.55,
  },
];

const LANG_PILL: Record<Lang, string> = {
  EN: "border-slate-500/40 text-slate-300",
  JA: "border-rose-500/40 text-rose-300",
  ZH: "border-amber-500/40 text-amber-300",
  KO: "border-sky-500/40 text-sky-300",
  DE: "border-emerald-500/40 text-emerald-300",
  FR: "border-indigo-500/40 text-indigo-300",
};

export function MultilingualNewsFusion() {
  const [items, setItems] = useState<FusedItem[]>(SEED);
  const [filter, setFilter] = useState<Lang | "ALL">("ALL");

  useEffect(() => {
    const id = setInterval(() => {
      setItems((prev) =>
        prev.map((it) => ({
          ...it,
          sentiment: Math.max(
            -1,
            Math.min(
              1,
              it.sentiment + (Math.sin(Date.now() / 6_000 + it.ts) - 0.5) * 0.04,
            ),
          ),
        })),
      );
    }, 5_000);
    return () => clearInterval(id);
  }, []);

  const filtered = useMemo(
    () =>
      items
        .filter((it) => filter === "ALL" || it.lang === filter)
        .sort((a, b) => b.ts - a.ts),
    [items, filter],
  );

  const langs: (Lang | "ALL")[] = ["ALL", "EN", "JA", "ZH", "KO", "DE", "FR"];

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Multilingual news fusion
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          fused non-EN sources · translated · sentiment-scored
        </p>
        <div className="mt-2 flex flex-wrap gap-1">
          {langs.map((l) => (
            <button
              key={l}
              type="button"
              onClick={() => setFilter(l)}
              className={`rounded border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ${
                filter === l
                  ? "border-accent bg-accent/15 text-accent"
                  : "border-border bg-bg/40 text-slate-400 hover:border-accent hover:text-accent"
              }`}
            >
              {l}
            </button>
          ))}
        </div>
      </header>
      <ul className="flex-1 divide-y divide-border/40 overflow-auto">
        {filtered.map((it) => {
          const ago = Math.max(1, Math.floor((Date.now() - it.ts) / 1000));
          return (
            <li
              key={`${it.source}-${it.ts}`}
              className="px-3 py-2 font-mono text-[11px] text-slate-300"
            >
              <div className="flex items-baseline gap-2">
                <span
                  className={`shrink-0 rounded border px-1.5 py-0.5 text-[9px] uppercase tracking-wider ${LANG_PILL[it.lang]}`}
                >
                  {it.lang}
                </span>
                <span className="truncate text-[10px] text-slate-500">
                  {it.source}
                </span>
                <span className="ml-auto shrink-0 text-[10px] text-slate-500">
                  {ago}s
                </span>
              </div>
              {it.lang !== "EN" && (
                <div className="mt-1 truncate text-slate-400" title={it.original}>
                  {it.original}
                </div>
              )}
              <div className="mt-1 truncate text-slate-200" title={it.translated}>
                {it.translated}
              </div>
              <div className="mt-1 flex items-baseline gap-2 text-[10px] text-slate-500">
                <span>sentiment</span>
                <span
                  className={
                    it.sentiment > 0.05
                      ? "text-emerald-400"
                      : it.sentiment < -0.05
                        ? "text-rose-400"
                        : "text-slate-400"
                  }
                >
                  {it.sentiment >= 0 ? "+" : ""}
                  {it.sentiment.toFixed(2)}
                </span>
              </div>
            </li>
          );
        })}
        {filtered.length === 0 && (
          <li className="px-3 py-4 text-center font-mono text-[11px] text-slate-500">
            no items match the {filter} filter
          </li>
        )}
      </ul>
    </section>
  );
}
