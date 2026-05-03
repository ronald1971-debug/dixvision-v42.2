import { dismissToast, useToasts, type ToastTone } from "@/state/toast";

const TONE_CLS: Record<ToastTone, string> = {
  info: "border-accent/40 bg-accent/10 text-accent",
  success: "border-ok/40 bg-ok/10 text-ok",
  warn: "border-warn/40 bg-warn/10 text-warn",
  danger: "border-danger/40 bg-danger/10 text-danger",
};

export function ToastHost() {
  const toasts = useToasts();
  if (toasts.length === 0) return null;
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex max-w-sm flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          role="status"
          className={`pointer-events-auto rounded border px-3 py-2 text-xs shadow-lg ${TONE_CLS[t.tone]}`}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1">
              <div className="font-medium leading-tight">{t.message}</div>
              {t.hint && (
                <div className="mt-0.5 font-mono text-[10px] opacity-70">
                  {t.hint}
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={() => dismissToast(t.id)}
              className="text-[10px] uppercase tracking-wider opacity-70 hover:opacity-100"
            >
              ×
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
