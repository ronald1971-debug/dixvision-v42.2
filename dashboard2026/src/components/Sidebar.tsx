import {
  Activity,
  Banknote,
  BarChart3,
  Bot,
  Brain,
  CandlestickChart,
  CheckSquare,
  ChevronLeft,
  ChevronRight,
  Coins,
  Gauge,
  Image as ImageIcon,
  KeyRound,
  Layers,
  LineChart,
  MessageSquare,
  Puzzle,
  Rocket,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";
import { type ComponentType } from "react";

import {
  ASSET_ROUTE_LIST,
  SYSTEM_ROUTE_LIST,
  type AssetRoute,
  type Route,
  type SystemRoute,
} from "@/router";

interface NavItem<R extends Route> {
  key: R;
  label: string;
  href: string;
  icon: ComponentType<{ className?: string }>;
}

const ASSET_NAV: Record<AssetRoute, NavItem<AssetRoute>> = {
  spot: { key: "spot", label: "Spot", href: "#/spot", icon: BarChart3 },
  perps: { key: "perps", label: "Perps", href: "#/perps", icon: Activity },
  dex: { key: "dex", label: "DEX", href: "#/dex", icon: Layers },
  memecoin: {
    key: "memecoin",
    label: "Memecoin",
    href: "#/memecoin",
    icon: Rocket,
  },
  forex: { key: "forex", label: "Forex", href: "#/forex", icon: Banknote },
  stocks: { key: "stocks", label: "Stocks", href: "#/stocks", icon: LineChart },
  nft: { key: "nft", label: "NFT", href: "#/nft", icon: ImageIcon },
};

const SYSTEM_NAV: Record<SystemRoute, NavItem<SystemRoute>> = {
  operator: {
    key: "operator",
    label: "Operator",
    href: "#/operator",
    icon: Bot,
  },
  credentials: {
    key: "credentials",
    label: "Credentials",
    href: "#/credentials",
    icon: KeyRound,
  },
  chat: { key: "chat", label: "Chat", href: "#/chat", icon: MessageSquare },
  indira: {
    key: "indira",
    label: "Indira learn",
    href: "#/indira",
    icon: Brain,
  },
  dyon: { key: "dyon", label: "Dyon learn", href: "#/dyon", icon: Wrench },
  testing: {
    key: "testing",
    label: "Testing & Eval",
    href: "#/testing",
    icon: CheckSquare,
  },
  onchain: {
    key: "onchain",
    label: "On-chain",
    href: "#/onchain",
    icon: Coins,
  },
  ai: {
    key: "ai",
    label: "AI · ASKB",
    href: "#/ai",
    icon: Sparkles,
  },
  orderflow: {
    key: "orderflow",
    label: "Order Flow",
    href: "#/orderflow",
    icon: CandlestickChart,
  },
  governance: {
    key: "governance",
    label: "Governance",
    href: "#/governance",
    icon: ShieldCheck,
  },
  risk: {
    key: "risk",
    label: "Risk",
    href: "#/risk",
    icon: Gauge,
  },
  charting: {
    key: "charting",
    label: "Charting",
    href: "#/charting",
    icon: LineChart,
  },
};

export interface SidebarProps {
  active: Route;
  collapsed: boolean;
  onToggle: () => void;
}

export function Sidebar({ active, collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={`flex flex-col border-r border-border bg-surface transition-[width] duration-200 ${
        collapsed ? "w-12" : "w-56"
      }`}
      aria-label="primary navigation"
      data-testid="sidebar"
    >
      <button
        type="button"
        onClick={onToggle}
        className="m-1 flex h-9 items-center justify-center rounded border border-transparent text-slate-400 hover:border-border hover:text-accent"
        aria-label={collapsed ? "expand sidebar" : "collapse sidebar"}
        title={collapsed ? "expand" : "collapse"}
      >
        {collapsed ? (
          <ChevronRight className="h-4 w-4" />
        ) : (
          <ChevronLeft className="h-4 w-4" />
        )}
      </button>

      <SidebarSection title="Assets" collapsed={collapsed}>
        {ASSET_ROUTE_LIST.map((key) => (
          <SidebarLink
            key={key}
            item={ASSET_NAV[key]}
            isActive={active === key}
            collapsed={collapsed}
          />
        ))}
      </SidebarSection>

      <SidebarSection title="System" collapsed={collapsed}>
        {SYSTEM_ROUTE_LIST.map((key) => (
          <SidebarLink
            key={key}
            item={SYSTEM_NAV[key]}
            isActive={active === key}
            collapsed={collapsed}
          />
        ))}
      </SidebarSection>

      <SidebarSection title="Plugins" collapsed={collapsed}>
        <PluginPlaceholder collapsed={collapsed} />
      </SidebarSection>
    </aside>
  );
}

function SidebarSection({
  title,
  collapsed,
  children,
}: {
  title: string;
  collapsed: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-2 border-t border-border pt-2">
      {!collapsed && (
        <div className="px-3 pb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
          {title}
        </div>
      )}
      <ul className="flex flex-col">{children}</ul>
    </div>
  );
}

function SidebarLink<R extends Route>({
  item,
  isActive,
  collapsed,
}: {
  item: NavItem<R>;
  isActive: boolean;
  collapsed: boolean;
}) {
  const Icon = item.icon;
  return (
    <li>
      <a
        href={item.href}
        title={item.label}
        aria-current={isActive ? "page" : undefined}
        className={`mx-1 my-0.5 flex items-center gap-2 rounded px-2 py-1.5 text-sm transition-colors ${
          isActive
            ? "bg-accent/10 text-accent"
            : "text-slate-300 hover:bg-bg hover:text-accent"
        }`}
        data-testid={`sidebar-link-${item.key}`}
      >
        <Icon className="h-4 w-4 shrink-0" />
        {!collapsed && <span className="truncate">{item.label}</span>}
      </a>
    </li>
  );
}

function PluginPlaceholder({ collapsed }: { collapsed: boolean }) {
  if (collapsed) {
    return (
      <li
        className="mx-1 my-0.5 flex items-center justify-center px-2 py-1.5 text-slate-600"
        title="Plugin slots — operator-installable widgets land here"
      >
        <Puzzle className="h-4 w-4" />
      </li>
    );
  }
  return (
    <li className="mx-1 my-0.5 rounded border border-dashed border-border bg-bg/40 px-2 py-2 text-[11px] text-slate-500">
      <div className="flex items-center gap-2">
        <Coins className="h-4 w-4" />
        <span>plugin slot</span>
      </div>
      <p className="mt-1 leading-snug text-slate-600">
        Operator-installed widgets mount here. Lifecycle through{" "}
        <code className="text-slate-400">/api/dashboard/action/lifecycle</code>.
      </p>
    </li>
  );
}
