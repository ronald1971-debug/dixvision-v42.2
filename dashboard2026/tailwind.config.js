/** @type {import('tailwindcss').Config} */

/*
 * dash-pro.B.0 — colors resolve through CSS variables defined in
 * src/theme/tokens.css. Each token is a space-separated RGB triplet
 * so Tailwind opacity utilities such as `bg-surface/40` keep working.
 *
 * Theme switching is a one-attribute mutation on <html data-theme="…">
 * (see src/theme/tokens.css). No bespoke `bg-bg` / `bg-surface`
 * override CSS is required anymore — the variables drive everything.
 */
const withVar = (name) => `rgb(var(${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: withVar("--bg"),
        surface: withVar("--surface"),
        "surface-raised": withVar("--surface-raised"),
        "surface-overlay": withVar("--surface-overlay"),
        border: withVar("--border"),
        hairline: withVar("--hairline"),
        accent: withVar("--accent"),
        ok: withVar("--ok"),
        warn: withVar("--warn"),
        danger: withVar("--danger"),
        info: withVar("--info"),
        "text-primary": withVar("--text-primary"),
        "text-secondary": withVar("--text-secondary"),
        "text-disabled": withVar("--text-disabled"),
      },
      fontFamily: {
        display: ["var(--font-display)"],
        mono: ["var(--font-mono)"],
      },
    },
  },
  plugins: [],
};
