# Wilfred — Brand Reference

This folder is the **brand reference** for **Wilfred** (an AI workflow platform for technology transfer). It is a *reference*, not a build system — read it to design and write on-brand, then apply the rules in whatever you're building.

> **Building inside the `tto-agent` app? Read `DECISIONS.md` first.** It records the
> concrete choices made implementing this brand in the app (green = forest-700,
> button system, toggles, lists, avatars, Flowbite porting, etc.) and **overrides
> this reference where they differ.** `flowbite-mapping.md` shows the wiring.

## Read in this order
0. **`DECISIONS.md`** — app-specific, authoritative decisions (+ `flowbite-mapping.md` for how the brand maps onto Flowbite).
1. **`cheatsheet.md`** — one page: exact colors, fonts, spacing, radii, shadows. Start here for fast lookups.
2. **`brand-guide.md`** — the full guide: brand voice & copywriting rules, color/type/layout philosophy, do's and don'ts.
3. **`tokens/`** — the source-of-truth CSS custom properties. If you're writing HTML/CSS, link `styles.css` and use the `var(--token)` names directly — never hardcode hexes you could reference.
4. **`assets/`** — the logo: `wilfred-mark.svg` (primary), `wilfred-mark-mono.svg` (single-color, inherits `currentColor`), `wilfred-seal.svg` (formal/stamp moments).

## The brand in one breath
Serious, trustworthy, business-professional — built for IP managers and commercialization officers, not a chirpy chatbot. **Forest green** for authority, a warm **copper/brass** accent used sparingly, on **warm paper** neutrals. **Source Serif 4** headlines, **Hanken Grotesk** UI/body, **Maple Mono** for reference IDs. Small radii, hairline rules, restrained cool shadows, calm motion, **no emoji**. Voice: a capable senior colleague who respects your time — sentence case, plain, precise, no hype.

## Hard rules (most-violated, so call them out)
- **No emoji, ever.** State = color + icon + a short word.
- **Sentence case** for all UI text. Title Case only for the wordmark and formal document titles.
- **Warm paper backgrounds** (`#FAF8F4`), never pure `#FFFFFF` pages; never pure-black shadows.
- **Copper is a seasoning, not a base** — accents only, never a large fill.
- Headlines in the **serif**, body/UI in the **sans**, IDs in the **mono**. Don't mix those up.

## Using the tokens in HTML
```html
<link rel="stylesheet" href="styles.css">
<!-- then use the variables -->
<h1 style="font-family: var(--font-serif); color: var(--text-heading)">Wilfred</h1>
<button style="background: var(--brand); color: var(--text-on-brand)">Ask Wilfred</button>
```
`styles.css` imports the five files in `tokens/` and loads the three webfonts (Source Serif 4 + Hanken Grotesk from Google Fonts; Maple Mono from the Fontsource CDN). A dark theme is available under `[data-theme="dark"]` or `.dark` on any ancestor.
