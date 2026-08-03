/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // OpenShorts dark/red theme — values MUST mirror tokens.css. They are
        // kept literal (not var()) so Tailwind alpha modifiers like bg-brass/10
        // still compile. That duplication is a trap: changing only tokens.css
        // leaves every `bg-paper*` utility on the old colour, which is exactly
        // how the "pitch black" theme shipped still rendering #101010 grey.
        // Touch both files together.
        paper: "rgb(0 0 0 / <alpha-value>)",
        paper2: "rgb(7 7 7 / <alpha-value>)",
        paper3: "rgb(19 19 19 / <alpha-value>)",
        ink: "rgb(250 250 250 / <alpha-value>)",
        ink2: "rgb(228 228 228 / <alpha-value>)",
        muted: "rgb(163 163 163 / <alpha-value>)",
        // Semantic colours were drifting from tokens.css: `ok` was a neon
        // #00FF7F here while every SVG used var(--color-ok) = #22c55e, and
        // `brass` was #FF0F0F against accent #ef4444. Two palettes rendering
        // side by side in the same view is most of what read as "cheap".
        // These now mirror tokens.css exactly — change both together.
        brass: "rgb(239 68 68 / <alpha-value>)",     // --color-accent
        brassink: "rgb(255 255 255 / <alpha-value>)",
        coral: "rgb(248 113 113 / <alpha-value>)",
        ok: "rgb(34 197 94 / <alpha-value>)",        // --color-ok
        warn: "rgb(245 158 11 / <alpha-value>)",     // --color-warn
        danger: "rgb(220 38 38 / <alpha-value>)",    // --color-danger
        // legacy aliases so untouched files degrade gracefully
        background: "rgb(0 0 0 / <alpha-value>)",
        surface: "rgb(7 7 7 / <alpha-value>)",
        primary: "rgb(239 68 68 / <alpha-value>)",
        accent: "rgb(239 68 68 / <alpha-value>)",
      },
      fontFamily: {
        display: "var(--font-display)",
        body: "var(--font-body)",
        sans: "var(--font-body)",
        serif: "var(--font-display)",
        mono: "var(--font-mono)",
      },
      borderColor: {
        rule: "var(--color-rule)",
        rule2: "var(--color-rule-2)",
      },
      borderRadius: {
        card: "var(--radius-card)",
        input: "var(--radius-input)",
      },
      fontSize: {
        micro: ["10.5px", { letterSpacing: "0.10em" }],
      },
      transitionTimingFunction: {
        out: "var(--ease-out)",
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade': 'fadeIn 0.4s var(--ease-out)',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        // Travelling sheen used to mark the pipeline stage that is running
        // right now (see TelemetryGrid's per-stage bars).
        shimmer: {
          '0%': { transform: 'translateX(-120%)' },
          '100%': { transform: 'translateX(420%)' },
        },
      },
    },
  },
  plugins: [],
}
