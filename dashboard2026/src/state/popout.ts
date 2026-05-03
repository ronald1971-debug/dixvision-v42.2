/**
 * J-track pop-out window helpers.
 *
 * The cockpit shell renders ``#/popout/<route>`` chromeless (no
 * sidebar, no top-ribbons) so an operator can detach a route into
 * its own browser window — useful for multi-monitor setups. The
 * pop-out window is a normal SPA mount; it shares localStorage with
 * the parent so preferences/hotkeys are coherent.
 */
import type { Route } from "@/router";

export function openPopout(route: Route): boolean {
  if (typeof window === "undefined") return false;
  const url = `${window.location.pathname}#/popout/${route}`;
  const features =
    "popup=yes,width=1200,height=820,menubar=no,toolbar=no,location=no";
  const w = window.open(url, `dix-popout-${route}`, features);
  return w !== null;
}

export function isPopoutHash(hash: string): boolean {
  return hash.startsWith("#/popout/");
}

export function parsePopoutHash(hash: string): string | null {
  if (!isPopoutHash(hash)) return null;
  const rest = hash.slice("#/popout/".length);
  return rest.split(/[/?]/)[0] || null;
}
