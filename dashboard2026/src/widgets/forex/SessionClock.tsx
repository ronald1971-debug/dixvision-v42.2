import { useEffect, useState } from "react";

interface Session {
  id: string;
  city: string;
  openUtc: number; // hour (UTC)
  closeUtc: number; // hour (UTC)
  tone: string;
}

const SESSIONS: Session[] = [
  { id: "sydney", city: "Sydney", openUtc: 22, closeUtc: 7, tone: "amber" },
  { id: "tokyo", city: "Tokyo", openUtc: 0, closeUtc: 9, tone: "rose" },
  { id: "london", city: "London", openUtc: 7, closeUtc: 16, tone: "sky" },
  { id: "newyork", city: "New York", openUtc: 12, closeUtc: 21, tone: "emerald" },
];

const TONE_CLASS: Record<string, { bar: string; pill: string }> = {
  amber: { bar: "bg-amber-500/40", pill: "bg-amber-500/20 text-amber-300 border-amber-500/40" },
  rose: { bar: "bg-rose-500/40", pill: "bg-rose-500/20 text-rose-300 border-rose-500/40" },
  sky: { bar: "bg-sky-500/40", pill: "bg-sky-500/20 text-sky-300 border-sky-500/40" },
  emerald: {
    bar: "bg-emerald-500/40",
    pill: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  },
};

function isOpen(s: Session, hourUtc: number): boolean {
  if (s.openUtc < s.closeUtc) {
    return hourUtc >= s.openUtc && hourUtc < s.closeUtc;
  }
  return hourUtc >= s.openUtc || hourUtc < s.closeUtc;
}

function nextChange(s: Session, hourUtc: number): { label: string; hours: number } {
  if (isOpen(s, hourUtc)) {
    const close = s.closeUtc;
    const hours = (close - hourUtc + 24) % 24 || 24;
    return { label: "closes in", hours };
  }
  const open = s.openUtc;
  const hours = (open - hourUtc + 24) % 24 || 24;
  return { label: "opens in", hours };
}

export function SessionClock() {
  const [hourUtc, setHourUtc] = useState<number>(() => new Date().getUTCHours());

  useEffect(() => {
    const id = window.setInterval(() => {
      setHourUtc(new Date().getUTCHours());
    }, 60_000);
    return () => window.clearInterval(id);
  }, []);

  const openCount = SESSIONS.filter((s) => isOpen(s, hourUtc)).length;

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Session Clock
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Sydney · Tokyo · London · New York · {hourUtc.toString().padStart(2, "0")}:00 UTC
          </p>
        </div>
        <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[11px] text-emerald-300">
          {openCount} open
        </span>
      </header>
      <div className="flex-1 overflow-auto px-3 py-2">
        <div className="mb-2 grid grid-cols-24 gap-px text-[8px] text-slate-600">
          {Array.from({ length: 24 }, (_, h) => (
            <div
              key={h}
              className={`text-center ${h === hourUtc ? "text-emerald-300" : ""}`}
            >
              {h % 6 === 0 ? h : ""}
            </div>
          ))}
        </div>
        <div className="space-y-1.5">
          {SESSIONS.map((s) => {
            const tone = TONE_CLASS[s.tone];
            const open = isOpen(s, hourUtc);
            const change = nextChange(s, hourUtc);
            return (
              <div key={s.id}>
                <div className="mb-0.5 flex items-baseline justify-between text-[11px]">
                  <div className="flex items-center gap-2">
                    <span className="text-slate-200">{s.city}</span>
                    <span
                      className={`rounded border px-1 py-px text-[9px] uppercase ${
                        open
                          ? tone.pill
                          : "border-slate-600/40 bg-slate-700/30 text-slate-500"
                      }`}
                    >
                      {open ? "live" : "off"}
                    </span>
                  </div>
                  <span className="font-mono text-[10px] text-slate-500">
                    {change.label} {change.hours}h
                  </span>
                </div>
                <div className="grid h-2 grid-cols-24 gap-px overflow-hidden rounded bg-slate-800/40">
                  {Array.from({ length: 24 }, (_, h) => (
                    <div
                      key={h}
                      className={
                        isOpen(s, h)
                          ? tone.bar
                          : h === hourUtc
                            ? "bg-slate-600/40"
                            : ""
                      }
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
