import {
  Activity,
  Banknote,
  BarChart3,
  Coins,
  Copy as CopyIcon,
  LayoutGrid,
  LineChart,
  ListOrdered,
  Sparkles,
  Target,
  Wallet,
} from "lucide-react";

import { navigate, type Route } from "@/router";

type NavItem = {
  id: Route;
  label: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
};

const ITEMS: ReadonlyArray<NavItem> = [
  { id: "explorer", label: "Pair Explorer", icon: LineChart },
  { id: "pools", label: "Pool Explorer", icon: LayoutGrid },
  { id: "bigswap", label: "Big Swaps", icon: Activity },
  { id: "multichart", label: "Multichart", icon: BarChart3 },
  { id: "trade", label: "Trade", icon: ListOrdered },
  { id: "copy", label: "Copy Trading", icon: CopyIcon },
  { id: "sniper", label: "Sniper", icon: Target },
  { id: "multiswap", label: "Multiswap", icon: Coins },
  { id: "wallet", label: "Wallet", icon: Wallet },
  { id: "stats", label: "Stats", icon: Sparkles },
];

export function Sidebar({
  active,
  collapsed,
}: {
  active: Route;
  collapsed: boolean;
}) {
  return (
    <aside
      className={`flex h-full shrink-0 flex-col gap-1 border-r border-border bg-surface ${
        collapsed ? "w-12" : "w-48"
      }`}
    >
      <div className="flex h-12 items-center justify-center border-b border-border px-2">
        {!collapsed ? (
          <span className="text-sm font-semibold tracking-wide text-accent">
            DIX&nbsp;MEME
          </span>
        ) : (
          <Banknote size={18} className="text-accent" />
        )}
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 px-1 py-2">
        {ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = item.id === active;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => navigate(item.id)}
              className={`flex items-center gap-2 rounded px-2 py-1.5 text-sm transition-colors ${
                isActive
                  ? "bg-[var(--accent-soft)] text-accent"
                  : "text-text-secondary hover:bg-surface-raised hover:text-text-primary"
              }`}
              title={item.label}
            >
              <Icon size={16} className="shrink-0" />
              {!collapsed && <span className="truncate">{item.label}</span>}
            </button>
          );
        })}
      </nav>
      <div className="border-t border-border px-2 py-2 text-[10px] text-text-disabled">
        {collapsed ? "v0.1" : "DIX MEME · v0.1"}
      </div>
    </aside>
  );
}
