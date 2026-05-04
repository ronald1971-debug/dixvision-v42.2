/** @type {import('tailwindcss').Config} */
// DIX MEME consumes the same theme tokens as dashboard2026. Both apps
// reference CSS custom properties (declared in `src/theme/tokens.css`)
// through the `rgb(var(--name) / <alpha-value>)` bridge so theme
// switching is a one-attribute mutation on `<html data-theme="…">`.
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
