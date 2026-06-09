# Wilfred — Cheat Sheet

Exact values. For the *why*, read `brand-guide.md`. For the live source, see `tokens/`.

## Color

### Brand — Forest green (primary)
| Token | Hex | Use |
|---|---|---|
| `--forest-950` | `#06150E` | Dark theme ground |
| `--forest-900` | `#0B2418` | Headlines on light, the mark tile, inverse surfaces |
| `--forest-800` | `#103120` | **Primary brand fill** (buttons) |
| `--forest-700` | `#16432C` | Brand hover, structure |
| `--forest-600` | `#1E5638` | |
| `--forest-500` | `#2C6E4A` | Accents, links on dark, data |
| `--forest-400` | `#589A77` | |
| `--forest-300` | `#8FBBA4` | |
| `--forest-200` | `#BFD8CB` | |
| `--forest-100` | `#DDEAE2` | |
| `--forest-50` | `#EEF4F0` | Soft brand wash |

### Accent — Copper / brass (use sparingly)
| Token | Hex | Use |
|---|---|---|
| `--copper-700` | `#8A5A2B` | Accent text on light |
| `--copper-500` | `#BE8242` | **Primary accent** |
| `--copper-400` | `#D29C63` | Accent on dark, the mark stroke |
| `--copper-300` | `#E2BC93` | |
| `--copper-100` | `#F4E6D5` | Selection bg |

### Neutrals — warm Paper + cool Slate text
| Token | Hex | Use |
|---|---|---|
| `--paper-0` | `#FFFFFF` | Cards only (never the page) |
| `--paper-50` | `#FAF8F4` | **Page background** |
| `--paper-100` | `#F4F1EA` | Sunken / muted surface |
| `--paper-200` | `#E4DFD3` | Subtle borders |
| `--paper-300` | `#D5CEBE` | Default borders |
| `--slate-800` | `#283044` | **Body text** |
| `--slate-600` | `#525C72` | Muted text |
| `--slate-400` | `#939BAE` | Faint text |

### Functional (muted, never neon)
| Role | Token | Hex |
|---|---|---|
| Success | `--green-600` | `#1A9255` |
| Warning | `--amber-600` | `#B07B23` |
| Danger | `--red-600` | `#B23B36` |

> Success green is deliberately brighter/cooler than brand forest so **state never reads as identity**.

### Key semantic aliases (prefer these over raw ramps)
`--surface-page` · `--surface-card` · `--surface-sunken` · `--surface-brand` · `--text-heading` · `--text-body` · `--text-muted` · `--text-on-brand` · `--text-accent` · `--text-link` · `--border-subtle` · `--border-default` · `--brand` / `--brand-hover` / `--brand-active` · `--accent` / `--accent-hover` · `--focus-ring`. A full dark set is defined under `[data-theme="dark"]` / `.dark`.

## Type

| | Family | Token | Use |
|---|---|---|---|
| Display / headings | **Source Serif 4** | `--font-serif` | Every headline, page title, the wordmark. Semibold, tracked tight. |
| UI / body | **Hanken Grotesk** | `--font-sans` | Labels, controls, tables, running copy. |
| Identifiers | **Maple Mono** | `--font-mono` | Patent #s, doc IDs, dates, counts, code. Tabular figures; **ligatures off** for IDs. |

**Scale (px):** `2xs 11` · `xs 12` · `sm 13` · `base 15` · `md 17` · `lg 20` · `xl 24` · `2xl 30` · `3xl 38` · `4xl 48` · `5xl 60` · `6xl 76` · `7xl 94`
**Weights:** 400 / 500 / 600 / 700 · **Leading:** tight 1.14, snug 1.28, normal 1.5, relaxed 1.65
**Tracking:** tight −0.014em (headlines) · caps 0.12em (eyebrows)
**Eyebrow recipe:** Hanken, uppercase, `--tracking-caps`, `--text-xs`, semibold, copper.

## Space · shape · depth

- **4px base grid.** Section rhythm `--space-12/16/20` (48/64/80px).
- **Radii:** `xs 3` · `sm 5` · **`md 8` (workhorse: cards, inputs, buttons)** · `lg 12` · `xl 16` · `pill 999`. Nothing bubbly.
- **Borders:** hairline `1px --border-default` does most structural work. Prefer a ruled surface over a filled box.
- **Shadows (cool, forest-tinted, low):** `--shadow-xs/sm` for cards, `--shadow-lg/xl` for modals. Base tint is `rgba(8,28,18,…)` — never pure black.
- **Focus:** `--focus-ring` = `0 0 0 3px var(--ring-brand)` (copper variant on accent elements). Always visible.

## Motion
Calm and brief. `--duration-base 200ms`, `--ease-standard`. Fades + 4–8px translates. **No bounce, no spring, no decorative loops.** Honor reduced-motion.

## Layout
App shell: fixed left rail `--rail-width 272px`, fluid content, optional right canvas. Content caps at `--container-lg 1080px` / `--container-xl 1280px`. Gutter 24px.

## Logo
- `assets/wilfred-mark.svg` — primary: forest tile + copper "Transfer" mark (an arc from a solid origin node to an open destination node — lab → market).
- `assets/wilfred-mark-mono.svg` — single color via `currentColor`.
- `assets/wilfred-seal.svg` — formal medallion for stamp moments.
- Wordmark: "Wilfred" in Source Serif 4 semibold. Eyebrow tag: "AI for technology transfer".

## Icons
**Lucide** (`https://unpkg.com/lucide@latest`). Outline only, `stroke-width 1.75–2`, `currentColor`. 16px inline · 18–20px in buttons/nav · 24px standalone. Never recolor copper unless deliberate. No emoji, no dingbats.
