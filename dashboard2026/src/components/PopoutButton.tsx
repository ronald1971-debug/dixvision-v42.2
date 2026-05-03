import { openPopout } from "@/state/popout";
import { pushToast } from "@/state/toast";
import type { Route } from "@/router";

/**
 * J-track tiny "↗ pop-out" button anchored on a page header.
 *
 * Clicking opens the same route in a new chromeless window so the
 * operator can drag it onto a second monitor. If the popup is
 * blocked by the browser, a toast is surfaced explaining why.
 */
export function PopoutButton({ route }: { route: Route }) {
  return (
    <button
      type="button"
      onClick={() => {
        const ok = openPopout(route);
        if (!ok) {
          pushToast("Pop-out blocked", {
            tone: "warn",
            hint: "allow popups for /dash2/ in your browser settings",
          });
        }
      }}
      className="rounded border border-border px-2 py-0.5 text-[10px] uppercase tracking-wider text-slate-300 hover:border-accent/50"
      title="Open this page in a separate window"
    >
      ↗ pop-out
    </button>
  );
}
