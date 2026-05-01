import { type ReactNode, useMemo, useState } from "react";
import {
  GridLayout,
  type Layout,
  type LayoutItem,
  useContainerWidth,
} from "react-grid-layout";

export interface GridItemSpec {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
  minW?: number;
  minH?: number;
  static?: boolean;
  render: () => ReactNode;
}

export interface AssetGridProps {
  /** Storage namespace; layouts are persisted under this key. */
  storageKey: string;
  /** Default layout if no override is found in localStorage. */
  defaultItems: GridItemSpec[];
  /** Number of columns in the grid. */
  cols?: number;
  /** Row height in px. */
  rowHeight?: number;
}

const STORAGE_PREFIX = "dixvision.dash2026.layout.";

function readLayout(key: string): LayoutItem[] | null {
  try {
    const raw = window.localStorage.getItem(`${STORAGE_PREFIX}${key}`);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as LayoutItem[];
    if (!Array.isArray(parsed)) return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeLayout(key: string, layout: Layout): void {
  try {
    window.localStorage.setItem(
      `${STORAGE_PREFIX}${key}`,
      JSON.stringify(layout),
    );
  } catch {
    // Quota / privacy mode — silently ignore.
  }
}

/**
 * `react-grid-layout` v2 wrapper that powers every per-asset surface.
 * Layouts are persisted per storage key so operators can rearrange a
 * dashboard once and have the change stick across reloads. The grid
 * is draggable + resizable; widget content is delegated to each
 * item's `render()`.
 */
export function AssetGrid({
  storageKey,
  defaultItems,
  cols = 12,
  rowHeight = 30,
}: AssetGridProps) {
  const initial = useMemo<LayoutItem[]>(
    () =>
      defaultItems.map((item) => ({
        i: item.i,
        x: item.x,
        y: item.y,
        w: item.w,
        h: item.h,
        minW: item.minW,
        minH: item.minH,
        static: item.static,
      })),
    [defaultItems],
  );

  const [layout, setLayout] = useState<LayoutItem[]>(() => {
    const persisted = readLayout(storageKey);
    return persisted ?? initial;
  });

  const { width, containerRef, mounted } = useContainerWidth();

  return (
    <div ref={containerRef} className="w-full">
      {mounted ? (
        <GridLayout
          width={width}
          layout={layout}
          gridConfig={{ cols, rowHeight, margin: [8, 8], containerPadding: [0, 0] }}
          dragConfig={{ handle: ".widget-drag-handle" }}
          onLayoutChange={(next: Layout) => {
            setLayout([...next]);
            writeLayout(storageKey, next);
          }}
        >
          {defaultItems.map((item) => (
            <div key={item.i} className="overflow-hidden">
              <div className="widget-drag-handle h-full w-full">
                {item.render()}
              </div>
            </div>
          ))}
        </GridLayout>
      ) : null}
    </div>
  );
}
