# Wilfred — Design decisions log

Concrete decisions made while implementing the rebrand in the app (branch
`redesign`, 2026-06). These resolve ambiguities in the brand guide, record where
we **deviated** from the kit, and lock choices that future UI work must follow.
Where this log and the rest of the kit disagree, **this log wins** for the app.

Implementation lives in `static/src/input.css` (token layer) and the templates;
see `flowbite-mapping.md` for the Flowbite wiring.

---

## Color

- **Primary brand green = forest-700 `#16432C`** (token `--color-brand`),
  lightened one step from the kit's forest-800 `#103120` (too dark in use).
  Hover = forest-600 `#1E5638` (`--color-brand-strong`). Text on brand =
  `--color-on-brand` (`#F2F7F4` light).
- **Dark theme: brand flips to copper.** In `.dark`, `--color-brand` = copper-500
  `#BE8242` (forest-on-forest disappears). Every `--color-brand*` / `--color-fg-brand*`
  var flips forest→copper. Dark surfaces are the "forest nocturne" steps.
- **Copper is an accent only — never a button fill.** It seasons: sparkle icons,
  the rail active-edge, eyebrows, the canvas top-line accent, one avatar tone.
  (A copper hero button was considered and rejected; primaries stay green.)
- **State always uses functional colors** (success green / warning amber / danger
  red), distinct from brand forest. Applies to toggles, badges, banners, icons.

## Buttons

- **Primary** = `bg-brand text-on-brand hover:bg-brand-strong` (forest).
- **AI / agent-action** = the primary button **+ a copper sparkle icon**
  (`text-accent dark:text-on-brand`; sparkle path starts `M9.813 15.9…`). The old
  purple gradient is **retired** — the brand bans bluish-purple.
- **Secondary** = white outline: `bg-neutral-primary-soft border border-default
  text-heading hover:bg-neutral-secondary shadow-xs` (replaced the bland beige fill).
- **Ghost** = text only (`text-body hover:text-heading hover:bg-neutral-secondary`).
- **Danger** = `bg-fg-danger text-white hover:opacity-90`.
- **Disabled** = `bg-neutral-quaternary text-fg-disabled cursor-not-allowed`.

## Components

- **Inputs / textareas** = soft warm inset: `bg-neutral-secondary-soft` (paper-100)
  + `border-default` + `focus:ring-brand`. Not white, not the darker paper-150.
  Django form inputs come from `accounts/forms.py::_input_classes()`.
- **Toggles / switches**: ON track = success green (`bg-fg-success`), OFF track =
  `bg-neutral-quaternary-medium`; white knob + `shadow-sm`. Colour shows state.
- **Avatars**: initials on a **3-tone cycle** — forest-light `#DDEAE2`/forest-900,
  forest-700/near-white, copper-500/white — chosen by a hash of the seed (email),
  so a given identity is always the same colour. Filter:
  `core/templatetags/branding.py::avatar_style`.
- **Lists (documents, data rooms, meetings, inbox) share ONE row design:** white
  card (`bg-neutral-primary-soft border border-default rounded-base`), hairline
  `border-b border-default` rows, `px-4 py-2.5`, hover `bg-neutral-secondary-soft`,
  dates/counts in `wf-mono text-xs text-body-subtle`, trailing ⋮ menu. New lists
  must match this.
- **Row ⋮ menu**: `bg-neutral-primary-soft border border-default rounded-base
  shadow-lg`; items `hover:bg-neutral-secondary`; Delete = `text-fg-danger
  hover:bg-danger-soft`.
- **Pills / tags**: `pl-2 pr-1.5 py-1 rounded-full border border-default`, leading
  icon (folder/doc muted; skill = copper sparkle on `bg-brand-softer`), trailing ✕
  in a rounded hover target. Status badges = soft fill + matching dot.
- **Chat thread rail**: active item = copper left-edge bar (`before:bg-accent`) +
  `bg-neutral-secondary-soft` fill + heading text; hover = soft fill.

## Navigation & chrome

- **Breadcrumbs replace "← Back"** on detail headers (data-room, meeting):
  leading folder/calendar icon → chevron separators → leaf crumb in `text-heading`.
- **Active top-nav item** = copper underline (`md:border-b-2 md:border-accent`),
  desktop only (the mobile dropdown keeps plain text).
- **Canvas header divider** between title and toolbar = green (`border-t-brand`).
- **Wordmark** = `assets/wilfred-mark.svg` + "Wilfred" in `font-serif`.
  **`assistant_emoji` was removed entirely** (it was a logo stand-in; the mark
  replaces it). Per-skill emojis are unrelated and stay.

## Type & icons

- Headings → **Source Serif 4**, body/UI → **Hanken Grotesk**, IDs/code →
  **Maple Mono** (`.wf-mono` / `font-mono`, tabular figures).
- **Tailwind's type scale is kept** — only fonts/colors/radii/shadows were ported
  into Flowbite. (The kit's px scale was *not* adopted; it broke dense layouts.)
- **Icons: keep the app's existing inline outline SVGs.** Lucide was evaluated and
  **not adopted** — the current set already matches the brand stroke style, and a
  swap was high-risk for no visual gain.

## Implementation rules

- The brand is delivered by **re-skinning Flowbite**: brand *values* are ported
  into Flowbite's own CSS-variable namespace in `static/src/input.css` (an override
  `@theme {}` for light + a literal `.dark {}` block, both after the Flowbite
  import so they win by source order). **Do not link the kit's `styles.css`.**
- **Do not register `--color-slate-*`** as a Tailwind palette — it collides with
  Tailwind's built-in slate/zinc. Feed slate hexes into semantic vars directly.
  Forest / copper / paper ramps are safe to register.
- Rebuild after token/template class changes: `npm run build:css` (+ `build:js` if
  the CodeMirror editor changed) then `collectstatic`; restart runserver.

## Out of scope (deferred, owner-led)

- **Landing page** (`templates/index.html`) — being redesigned separately; it's
  fine if the reskin leaves it visually inconsistent for now.
- **Voice / copy** — the owner controls wording; no voice pass was done.
