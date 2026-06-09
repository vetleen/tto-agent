# Handoff: Landing page redesign (`index.html`)

## Overview
A ground-up redesign of Wilfred's public marketing page — the view served by the
root URL to logged-out visitors. It replaces the current single-screen hero
(`templates/index.html`) with a full editorial landing page: hero, trust strip,
problem statement, three capability features, a security band, a testimonial, a
final CTA, and a footer. The voice and structure follow the Wilfred brand
(serious, editorial, forest-green + copper).

---

## ⚠️ How to implement this (read first)

The files in `reference/` are **design references built in HTML/React** — a
prototype showing the intended *look*. They are **not** code to drop in.

**Implement this manually, in the repo's existing conventions.** Specifically:

- This is a **Django template + Tailwind v4 + Flowbite** codebase. Rebuild the
  page as a normal Django template using **Tailwind utility classes** and the
  **design tokens already defined in `static/src/input.css`**.
- **Do NOT** import the prototype's React/Babel runtime, the `_ds_bundle.js`
  design-system bundle, or any of its CSS. **Do NOT** stand up a parallel styling
  system. The repo already has the entire Wilfred token set (forest/copper/paper
  ramps + semantic classes) wired into its Tailwind `@theme`. Use it.
- The prototype uses CSS custom properties like `var(--forest-950)` and a few
  bespoke `.ld-*` classes. **Translate every one of these into the repo's
  existing utilities** (see the mapping table below). The end result should be
  indistinguishable from the rest of the codebase — no new global CSS files
  unless a specific effect genuinely can't be expressed in utilities (and if so,
  add it to `input.css`/`output.css` the same way existing custom rules are).
- **The goal is purely visual: make the page look like the reference.** Keep
  Flowbite, keep the build pipeline, keep `_base.html`'s patterns. Nothing about
  the underlying system needs to change.

The prototype is **high-fidelity** — colors, type, spacing, and copy are final.
Match them.

---

## Where it goes in the repo

| File | Change |
|---|---|
| `templates/index.html` | Replace its contents. It already `{% extends "_base.html" %}` and renders inside the `landing_page` context (so `_base.html` sets `data-theme="dark"`). Keep extending base, but **override the `navbar` block** with the new top nav, and put all sections in the `content` block. |
| `static/img/brand/` | `wilfred-mark.svg`, `wilfred-mark-mono.svg`, `wilfred-seal.svg` **already exist here** — reference them with `{% static %}`. |
| `static/images/hero/` | Add the two photos from this bundle's `assets/` folder (see Assets). |
| `static/src/input.css` → rebuild `output.css` | Only if a section needs a utility/class that isn't generated yet. Re-run the Tailwind build (`package.json` has the scripts). |

Note: `_base.html`'s default navbar block links the brand to `/` and shows a
"Log in" link for anonymous users. The current `index.html` already overrides
`navbar`. Do the same — the new nav has brand + 3 anchor links + "Log in".

---

## Design tokens → repo utilities (the translation key)

Every value in the prototype maps to something the repo already defines in
`static/src/input.css`. Build with these, not raw hex.

### Color
The repo registers the full ramps as Tailwind colors **and** semantic aliases.
Both are available as utilities (`bg-forest-950`, `text-copper-300`,
`bg-neutral-primary`, `text-heading`, …).

| Prototype var | Use in repo | Notes |
|---|---|---|
| `--forest-950 #06150E` | `bg-forest-950` | Hero bg, security band, footer (darkest ink) |
| `--forest-900 #0B2418` | `bg-forest-900` | Final CTA bg, chat-mock surface |
| `--forest-800 #103120` | `bg-brand` / `bg-forest-800` | Brand fill |
| `--forest-700 #16432C` | `border-forest-700`, `bg-forest-700` | Mock borders, user avatar |
| `--forest-500 #2C6E4A` | `text-forest-500` | Feature checklist check icons |
| `--copper-500 #BE8242` | `text-accent` / `bg-accent` / `text-fg-accent` | The accent. Eyebrows, the hero CTA fill, quote marks. Use sparingly. |
| `--copper-400 #D29C63` | `text-copper-400` | Copper on dark (icons, caret, rules on film) |
| `--copper-300 #E2BC93` | `text-copper-300` | Eyebrow text & italic word **on dark** photo/ink |
| `--paper-50 #FAF8F4` | `bg-neutral-primary` | Page background |
| `--paper-0 #FFFFFF` | `bg-neutral-primary-soft` / `bg-paper-0` | Cards, trust strip, capabilities band |
| `--paper-400 #B6AD99` | `text-default-strong` | Muted serif fragment in statement headline |
| `--slate-500` body text | `text-body` (#283044) / `text-body-subtle` (#525C72) | Running copy |
| headings | `text-heading` (#0B2418) | Serif headlines on light |
| `--border-subtle` | `border-default-subtle` (#E4DFD3) | Hairlines on light |
| `--border-default` | `border-default` (#D5CEBE) | Stronger hairlines |
| `--border-inverse` (white 12%) | `border-default` **inside `.dark` scope** = `rgba(255,255,255,.12)` | Hairlines on dark. The page is dark-themed via `_base.html`, so dark-scope semantic borders resolve automatically. |

Text/scrims on photos use literal `rgba(6,21,14, …)` (forest-950) gradients —
keep those as inline styles or a small utility; they're brand scrims, not tokens.

### Typography
The repo's font utilities already map to the right families:

| Role | Repo utility | Family |
|---|---|---|
| Display & headings | `font-serif` | Source Serif 4 (600, tight tracking) |
| UI, labels, body | `font-sans` / default `font-body` | Hanken Grotesk |
| IDs, counts, metadata | `font-mono` | **Maple Mono** — the repo's mono. The prototype shows IBM Plex Mono; **use `font-mono` (Maple Mono)** to match the codebase. |

Headline sizes (clamped, fluid) from the prototype:
- Hero `h1`: `clamp(44px, 6.2vw, 82px)`, line-height 1.02, letter-spacing −0.028em, serif 600, color `#F4F7F4`. Italic word ("revenue") in `text-copper-300`.
- Section `h2`: `clamp(28–34px, ~3.5vw, 44–46px)`, line-height ~1.1, tracking −0.022em.
- Feature `h3`: `clamp(24px, 2.6vw, 33px)`, tracking −0.02em.
- Body/lede: 17–21px, line-height ~1.6, `text-body` (or `rgba(231,239,233,.82)` on dark).
- Eyebrow/overline: Hanken, 12px, weight 600, `text-transform: uppercase`,
  `letter-spacing: 0.12em`, copper. On light it's `text-fg-accent` with a 26px
  copper hairline before it; on dark it's `text-copper-300`.

### Spacing / shape
- Section vertical rhythm: `padding-block: clamp(72px, 9vw, 132px)`.
- Content container: `max-width: 1240px`, auto margins, side padding
  `clamp(20px, 6vw, 88px)`.
- Radii: cards/media use `--radius-lg` (12px); inputs/buttons 8px; pills for
  chips/tags. Repo's `rounded-lg` / `rounded-base` / `rounded-full` cover these.
- Borders do the structural work — hairline `1px` rules and whitespace, **not**
  heavy boxes or shadows. Shadows are low, cool, forest-tinted (repo's
  `shadow-*` tokens), reserved for the feature media and chat mock.

---

## Sections (top to bottom)

Exact copy lives in `reference/landing.jsx`. Screenshots in `reference/`
(`full-page.png`, `01–06-section.png`). Summary of each:

### 1. Top nav  (`navbar` block override)
Sticky, `bg-neutral-primary` at ~86% opacity + `backdrop-blur`, bottom hairline.
Height ~68px. Left: mark (30px) + "Wilfred" serif 22px. Center-left: three anchor
links — "Capabilities", "Security", "The workspace" — Hanken 14.5px,
`text-body-subtle` → `text-heading` on hover. Right (`margin-left:auto`): a text
"Log in" link + a primary **Log in** button. Links hide below 720px.
→ Point both "Log in" actions at `{% url 'accounts:login' %}`.

### 2. Hero
Full-bleed dark section on `bg-forest-950`. Background photo
(`landing_page_illustration.png`) cover-positioned ~60% x, filtered
`saturate(.82) brightness(.74)`. Over it, layered scrims (left→right forest
gradient + bottom-up forest gradient + a soft copper radial top-right) so text
stays legible — keep these as the prototype's CSS gradients.
Content max-width 760px, big vertical padding `clamp(96px,14vw,184px)`:
- Eyebrow "For technology transfer offices" (copper-300, with copper hairline).
- `h1`: "The office that turns research into *revenue*." ("revenue" italic, copper-300).
- Lede paragraph (see jsx), `rgba(231,239,233,.82)`, max 580px.
- Two buttons: **accent** "Log in" (copper fill) + **secondary** "See what
  Wilfred does" (→ `#capabilities`).
- Meta row, `font-mono` 12.5px, `rgba(231,239,233,.6)`, three items with copper
  Lucide icons (`shield-check`, `quote`, `file-search`) separated by tiny dots.

### 3. Trust strip
`bg-paper-0`, top+bottom hairline, ~34px vertical padding. Left: uppercase label
"Working inside research institutions" (`text-faint`, max 150px). Right: five
institution names in `font-serif` 16–20px, `text-default-strong`/slate, wrapping.
(Names are placeholder partners — see jsx.)

### 4. Problem statement
`bg-neutral-primary`. Two-column grid `0.8fr / 1.4fr`, gap ~72px, collapses to one
column < 860px. Left col: eyebrow "The problem" + serif `h2` "The work that
creates value is buried under the work that records it." Right col: two body
paragraphs, 16–18.5px, `text-body`.

### 5. Capabilities
`bg-paper-0`, hairline top+bottom. Header block (max 620px): eyebrow
"Capabilities" + `h2` "Everything the office does, with a colleague on it." +
intro paragraph (`text-body-subtle`).
Then **three alternating feature rows** (`.ld-feature`), each a two-column grid
(text / media), gap ~80px, every other one flipped (media on left). Rows stack to
one column < 860px. Each feature:
- Eyebrow (copper), serif `h3`, body paragraph (max 480px), and a checklist
  (`<ul>`, no bullets) — each item a `check` Lucide icon (`text-forest-500`) + text.
- **Media** is one of:
  - *Data rooms* — photo `landing_page_hero.png` in a rounded, hairline-bordered,
    shadowed frame (aspect 16/11) with a glass tag bottom-left:
    "📁 12 documents · 412 chunks indexed" (`folder-lock` icon, mono, dark glass).
  - *Chat that drafts* — a **chat mock** (not a photo): dark `bg-forest-900`
    rounded panel, title bar with 3 dots + "Thread · Patent Portfolio Q1", two
    data-room chips, a user message ("DO" avatar) and a Wilfred message ("W"
    copper avatar) ending in a mono citation chip + a blinking copper caret.
    Rebuild with divs; the blink is a CSS `@keyframes` (honor
    `prefers-reduced-motion`).
  - *Meetings* — photo `landing_page_illustration.png`, same frame, tag
    "🎙 Minutes with Wilfred · saved" (`mic` icon).

  (Tag icons are Lucide; the leading glyphs above are shorthand, **not emoji** —
  use the named Lucide icon. No emoji anywhere per brand.)

### 6. Security band
`bg-forest-950`, white text, a soft copper radial top-left. Top row: the
`wilfred-seal.svg` (88px) + eyebrow "Trust" + serif `h2` "Built for the most
confidential work in the building." Below: a 3-column grid of cells separated by
1px inverse hairlines (one column < 860px). Each cell: a copper Lucide icon
(`shield-check`, `quote`, `scan-eye`), serif `h4`, and body copy
(`rgba(231,239,233,.66)`). Copy in jsx.

### 7. Testimonial
`bg-paper-0`, bottom hairline, max 920px. Eyebrow "From the office", then a large
serif `blockquote` (24–38px) with copper quote marks, then attribution: a square
forest avatar "DO" + name "Dr. Daniel Okafor" / "Director, Technology Transfer
Office". (Placeholder quote — see jsx.)

### 8. Final CTA
`bg-forest-900`, copper radial glow from bottom. Centered: serif `h2` "Meet your
office's newest colleague." (34–60px, `#F4F7F4`), a one-line subhead, an **accent
Log in** button, and a mono note "An AI colleague for technology transfer".
(The only centered section — everything else is left-aligned editorial.)

### 9. Footer
`bg-forest-950`, ~40px padding. Left: mark (24px) + "Wilfred" serif. Right
(`margin-left:auto`): mono line "Wilfred · AI workflows for technology transfer".

---

## Buttons (map to existing components)
The prototype uses a design-system `<Button>` with `variant` + `size`. In the
repo, render these as normal anchors/buttons styled like the existing CTAs (the
current `index.html` and `_base.html` show the button patterns). Three roles:
- **accent** — copper fill (`bg-accent` / `bg-copper-500`), dark text
  (`#1A1206`), hover `bg-accent-hover`. One per view max (hero + final CTA reuse
  it; that's fine as they're different viewports).
- **primary** — forest fill (`bg-brand`, `text-on-brand`), hover `bg-brand`
  darker. Used by the nav "Log in".
- **secondary** — transparent/outline on dark: 1px inverse border, light text,
  subtle hover lift. Used by hero's "See what Wilfred does".
Sizes: `sm` (nav) and `lg` (hero/CTA). Match padding/radius to existing buttons.

## Interactions & behavior
- Anchor links scroll to `#capabilities`, `#security`, `#workspace` (the chat
  mock carries `id="workspace"`). Add `scroll-behavior: smooth` (honor reduced
  motion).
- Sticky nav with backdrop blur.
- Blinking caret in the chat mock: CSS `@keyframes` step blink, disabled under
  `prefers-reduced-motion: reduce`.
- The prototype has optional fade/rise entrance classes (`.ld-rise`) but ships
  them **inert** (no opacity hiding) per the brand's restraint — you can skip
  entrance animation entirely. No bounce/spring anywhere.
- Fully responsive: grids collapse to single column < 860px; nav links hide
  < 720px; all type uses `clamp()`.

## Icons
Lucide (the repo standardizes on it). Icons used: `shield-check`, `quote`,
`file-search`, `check`, `folder-lock`, `file-text`, `mic`, `scan-eye`. Outline,
`stroke-width` ~1.75–2, `currentColor`. Size with neighboring text (14–22px).
Use whatever Lucide-loading approach the repo already uses; do not introduce a
new icon set.

## Assets
| File | Where | Notes |
|---|---|---|
| `wilfred-mark.svg`, `wilfred-mark-mono.svg`, `wilfred-seal.svg` | already in `static/img/brand/` | reference via `{% static %}` |
| `landing_page_hero.png` (1408×768) | this bundle `assets/` → put in `static/images/hero/` | "Data rooms" feature media; researcher's desk |
| `landing_page_illustration.png` (1376×768) | this bundle `assets/` → put in `static/images/hero/` | Hero background **and** "Meetings" media; library corridor |

Both photos are AI-generated placeholders matching the brand's editorial/warm
direction — swap for licensed institutional photography before launch if desired.

## Reference files in this bundle
- `reference/Wilfred Landing.html` — the prototype shell (markup + all the
  `.ld-*` CSS you're translating to utilities).
- `reference/landing.jsx` — **the source of exact copy and structure** for every
  section. Read this for verbatim text.
- `reference/full-page.png`, `reference/0X-section.png` — rendered visual targets.
