# Wilfred — how the brand is built in the app

> **Showcase note, not a spec.** This records *how* the forest/copper brand was
> implemented in the Wilfred Django app (Tailwind v4 + Flowbite v4) during the
> 2026-06 reskin. The brand reference (the rest of this folder) remains the source
> of truth for *values*; this page just shows the wiring. It does not constrain
> future work.

## Approach: re-skin Flowbite, don't fork it

The app uses Flowbite's **semantic utility classes** (`bg-brand`, `text-body`,
`text-heading`, `bg-neutral-primary`, `border-default`, `bg-warning-soft`, …),
which resolve through Flowbite CSS variables. The brand is delivered by
**redeclaring those variables with brand values** in `static/src/input.css` —
not by linking this folder's `styles.css` (its variable names differ and its
global element rules would fight Flowbite's preflight).

Order in `input.css`:
1. Font `@import`s (Source Serif 4, Hanken Grotesk) + Maple Mono `@font-face`.
2. `@import "flowbite/src/themes/default"` — the variable surface we override.
3. Override `@theme {}` (light) — wins by source order.
4. Literal `.dark {}` — dark overrides (NOT `@theme`).
5. `@import "tailwindcss"` + Flowbite plugin/source + `@layer base`.

## Brand role → Flowbite variable

| Brand role | Flowbite var(s) | Light | Dark |
|---|---|---|---|
| Primary fill | `--color-brand`, `-strong` (hover) | forest-700 / -600 | **copper-500 / -400** (flips) |
| Text on brand | `--color-on-brand` (added) | `#F2F7F4` | forest-950 |
| Accent (copper) | `--color-accent*`, `--color-fg-accent` (added) | copper-500 / -700 | copper-400 |
| Page / card / sunken | `--color-neutral-primary` / `-soft` / `-secondary` | paper-50 / -0 / -100 | forest nocturne steps |
| Body / heading text | `--color-body`, `--color-heading` | slate-800 / forest-900 | `#C4D6CB` / `#EFF5F1` |
| Default border | `--color-default` | paper-300 | white @ 12% |
| Success/Warning/Danger | `--color-{success,warning,danger}-*` | muted green/amber/red | dark steps |
| Links / active nav | `--color-fg-brand` | forest-700 | copper-300 |

Fonts: `--font-sans`/`--font-body` → Hanken Grotesk, `--font-serif` (added) →
Source Serif 4 (applied to `h1–h4` in `@layer base`), `--font-mono` → Maple Mono.
Radii: `--radius-xs/-sm` tightened; shadows retinted forest (light) / near-black (dark).

## Components

All standard UI is **Flowbite** (buttons, dropdowns, modals, tooltips, toggles,
tabs). Brand-specific additions, kept minimal:
- **Agent-action buttons:** Flowbite primary button (`bg-brand text-on-brand`) +
  a **copper sparkle** icon (`text-accent dark:text-on-brand`). No purple.
- **Eyebrow / mono helpers:** `.wf-eyebrow`, `.wf-mono`, `.tnum` in `@layer base`.
- **Logo:** `assets/wilfred-mark.svg` in the nav + "Wilfred" in `font-serif`.

Icons remain the app's existing inline outline SVGs (already on the brand's
stroke style; Lucide was not adopted).
