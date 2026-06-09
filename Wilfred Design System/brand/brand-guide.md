# Wilfred — Brand Guide

**Wilfred** is an AI workflow platform for **Technology Transfer**. It gives technology-transfer offices (TTOs) — the teams inside universities and research institutions that turn research into patents, licenses, and spinouts — an AI colleague that handles the routine, document-heavy work of the office: intake and invention disclosures, data-room research, meeting minutes, and process guidance.

This document is the **brand guide** for Wilfred — the voice, color, typography, and layout rules a design or coding agent needs to produce on-brand interfaces, marketing, and decks. The exact token values live alongside it in `tokens/` and `cheatsheet.md`; the logo in `assets/`.

> **This is a ground-up rebrand.** The previous Wilfred identity was playful (a hand-drawn "Delicious Handrawn" wordmark on Flowbite defaults). This system replaces it with a **serious, trustworthy, business-professional** identity built for an audience of IP managers, commercialization officers, university administrators, and the legal and industry partners they work with.

---

## The product (what we're designing for)

Wilfred is a Django web application. Its core surfaces:

| Surface | What it is |
|---|---|
| **Chat** | The heart of the product. Thread-based assistant; each thread can attach one or more **data rooms** and **skills**. Streaming responses, tool use, a multi-canvas document editor, and sub-agent delegation. |
| **Data Rooms** | Secure document workspaces. Upload PDF/DOCX/TXT/MD/HTML; documents are chunked, embedded (pgvector), and made searchable via hybrid semantic + full-text retrieval. PII / guardrail status is surfaced per document. |
| **Meetings** | First-class meeting objects with live transcription, audio/transcript upload, attachments, artifacts, and "minutes with Wilfred" threads that search linked data rooms. |
| **Skills** | A three-tier (system / organization / user) agent-skills system. Skills customize the assistant's behavior and available tools per thread. |
| **Accounts** | Auth, per-user settings (including light/dark theme), organizations, usage & budget tracking. |
| **Landing / marketing** | Public site introducing Wilfred to TTO leaders. |

The mental model for the brand: **Wilfred is a trusted senior colleague** — composed, precise, discreet, quietly excellent. Not a chirpy chatbot. A dependable officer of the institution.

### Source material

This system was built from the product codebase:

- **GitHub:** [`vetleen/tto-agent`](https://github.com/vetleen/tto-agent) — the Wilfred Django application (Django 6, Tailwind CSS v4, Flowbite). Branch `main`. Explore this repo to understand real product structure, copy, and screen layouts when building new surfaces. Key reading: `README.md`, `CLAUDE.md`, `templates/_base.html`, and the per-app templates under `templates/{chat,documents,meetings,accounts,agent_skills}/`.

The **redesign direction** ("serious and trustworthy, modern, business professional — completely reimagine the brand") was specified by the design owner. Where this system and the live codebase disagree visually, **this system wins** — it is the target, not a mirror of today's UI. Product *structure and copy*, however, are taken faithfully from the codebase.

---

## CONTENT FUNDAMENTALS

How Wilfred writes. The voice is the product's most important brand asset — Wilfred earns trust through how it speaks.

### Voice in one line
**A capable colleague who respects your time and your judgment.** Calm, precise, plain-spoken, never performative.

### Person & address
- **Address the user as "you."** Speak about the assistant in the third person in the UI chrome ("Minutes with Wilfred", "Ask Wilfred"), and in the first person *only* inside conversation ("I searched three data rooms and found…"). The product is named Wilfred; the user is the officer in charge.
- **The user is the principal; Wilfred is staff.** Wilfred proposes, drafts, retrieves, and flags — the user decides. Never presumptuous, never sycophantic.

### Tone & register
- **Professional, not stiff.** Full sentences, real punctuation, no slang. But not legalese either — clear over clever.
- **Confident and exact.** Prefer specific nouns and numbers ("3 data rooms, 412 chunks") over vague reassurance ("lots of sources").
- **Discreet.** This is sensitive IP and pre-publication research. Copy never over-shares, never jokes about confidentiality, and treats data handling seriously.
- **No hype.** Avoid "revolutionary," "magical," "supercharge," "effortless," "unleash." Wilfred is competent, not breathless.

### Casing & mechanics
- **Sentence case everywhere** — buttons, headings, menu items, nav. ("Create data room", not "Create Data Room".)
- **Product nouns are capitalized as proper features:** Data Room, Meeting, Skill, Canvas, Thread. Lowercase when generic ("upload a document").
- **Title Case is reserved** for the wordmark and formal document titles only.
- **Oxford comma. One space after periods.** Numerals for counts and IDs (`12 documents`, `WO-2024-0421`). Spell out only at sentence starts.

### Emoji & ornament
- **No emoji. Ever.** They undercut the trustworthy register. State is communicated with color, icon, and a short word — never a 🎉.
- Em dashes for asides, set tight. Avoid exclamation marks (one is forgivable on a genuine success; never two).

### Microcopy patterns
| Situation | Wilfred says | Not |
|---|---|---|
| Empty state | "No data rooms yet. Create one to give Wilfred something to work with." | "Nothing here! 😢 Add your first room!" |
| Success | "Minutes saved to Patent Portfolio Q1." | "Boom! Saved! 🎉" |
| Working | "Searching 3 data rooms…" | "Hang tight, doing magic…" |
| Error | "That file couldn't be processed. PDFs, DOCX, TXT, MD, and HTML are supported." | "Oops! Something went wrong." |
| Destructive confirm | "Delete this data room and its 18 documents? This can't be undone." | "Are you sure???" |
| CTA | "Ask Wilfred", "Create data room", "Upload documents", "Start a meeting" | "Get started", "Let's go!" |

### Naming
- The assistant is **Wilfred** (never "the AI", "the bot", "the assistant" in user-facing copy — it's a named colleague).
- Features are nouns a TTO professional already knows or can learn in one read: Data Room, Disclosure, Minutes, Thread, Canvas, Skill.

---

## VISUAL FOUNDATIONS

The look: **institutional but modern** — the world where a research university meets a patent office and a boardroom. Premium, editorial, precise. Think the considered restraint of a good law firm's stationery and a research journal's typesetting, brought into a calm modern interface.

### Color
- **Forest green is the primary brand color** (`--forest-800 #103120`, deepening to `--forest-900 #0B2418`). It carries authority, growth and created value. It's the color of primary buttons, the brand mark, headlines on light, and the app's inverse surfaces.
- **Copper / brass is the single accent** (`--copper-500 #BE8242`). It evokes patent seals, brass nameplates, and craft — reading as patina against forest — and adds human warmth so the green never feels cold or corporate-generic. Used sparingly: the mark, key highlights, eyebrows, active accents, one hero CTA at most per view. **Copper is a seasoning, not a base.**
- **Neutrals are warm paper**, not cool gray — `--paper-50 #FAF8F4` page, `--paper-0 #FFFFFF` cards. This is the single biggest "premium editorial" signal; pure cold white is avoided. Body text sits in the slate ramp (`--slate-800`) so everything stays in one tonal family.
- **Functional colors are muted and grown-up:** a brighter emerald for value/success (`--green-600 #1A9255`, deliberately stepped away from brand forest so state never reads as identity), a tobacco amber for warnings, a brick red for danger. None are neon.
- **Dark (forest) theme** grounds onto a near-black forest (`--forest-950 #06150E`) with near-white text; copper brightens to `--copper-400` and becomes the primary fill, since forest-on-forest would disappear. Both themes ship as token scopes (`:root` and `[data-theme="dark"]` / `.dark`).

### Typography
- **Display & headings: Source Serif 4** — a modern transitional serif. Serifs do the heavy lifting of trust and editorial gravitas here; every headline, page title, and the wordmark are serif, set tight (`--tracking-tight`) and semibold.
- **UI & body: Hanken Grotesk** — a clean, slightly warm neo-grotesque. All labels, controls, table data, and running body copy. High legibility at small sizes for dense document UI.
- **Identifiers: Maple Mono** — patent numbers, document IDs, dates, token counts, code. TTO work is full of reference numbers; a warm, rounded monospace with tabular figures makes them scannable and honest. Code ligatures are disabled for identifier display. Open source (SIL OFL 1.1).
- **Eyebrows / overlines:** Hanken, uppercase, `--tracking-caps (0.12em)`, copper. The one recurring "label" motif that organizes pages.
- Type pairs serif headline → sans deck/body → mono metadata. Don't set body in the serif; don't set headlines in the sans.

### Space, grid & layout
- **4px base grid.** Generous — this system breathes. Section rhythm uses `--space-12/16/20`. Density is reserved for data tables and the chat rail.
- **Hairline rules (`1px`, `--border-default`) do most of the structural work** — Wilfred separates with lines and whitespace far more than with boxes or shadow. An editorial, ruled-ledger feeling.
- App shell: fixed left **rail** (`--rail-width 272px`) for threads/navigation, fluid content, optional right canvas/inspector panel. Content rails cap at `--container-lg/xl`.

### Shape, depth & borders
- **Small radii.** `--radius-md (8px)` is the workhorse for cards, inputs, buttons. Pills (`--radius-pill`) only for badges/tags and avatars. Nothing is bubbly; corners are crisp and institutional.
- **Cards = white (`--surface-card`) on paper, 1px `--border-subtle`, `--shadow-xs`/`sm`.** Restraint. Elevation is earned: shadows are cool and forest-tinted (`--shadow-*`), low and tight, never soft gray blooms. Modals/popovers use `--shadow-lg/xl`.
- **Borders define more than fills.** Default to a hairline-bordered surface over a filled one. Copper borders mark accented/active elements; forest borders mark the brand/primary.

### Motion
- **Calm and brief.** `--duration-base 200ms` with `--ease-standard`/`--ease-out`. Fades and short translates (4–8px). **No bounce, no spring, no decorative loops.** Streaming chat uses a quiet cursor/caret, not a dancing dot circus. Reduced-motion is fully honored.

### Interaction states
- **Hover:** surfaces lift one paper step (`--surface-card → --surface-sunken`) or borders darken `subtle → default`; primary buttons go `--brand → --brand-hover`. Subtle, never a color-pop.
- **Press:** a touch darker (`--brand-active`) and an optional 1px nudge; no scale-shrink toys.
- **Focus:** a 3px soft ring (`--focus-ring`, forest on most, copper on accent elements). Always visible, never removed.
- **Selected / active nav:** copper left-edge or copper text + `--surface-brand-soft` fill.
- **Disabled:** 45–50% opacity, no shadow, `not-allowed`.

### Imagery
- **Photography is editorial and institutional:** labs, libraries, architecture, hands and documents, partnership/handshake moments — shot warm, with natural light, never stocky-glossy. Apply a subtle forest duotone or a warm low-contrast grade so imagery sits in the palette. Generous negative space for type overlays.
- **Protection:** dark forest gradient scrims (`rgba(6,21,14, .7→0)`) under text on photos, bottom-up. Not flat black.
- **No illustration-as-decoration, no 3D blobs, no gradient mesh.** When a concept needs a visual, prefer a real photograph, a document/diagram, or restrained data viz over a cartoon.
- Data viz uses the forest ramp + copper for the "focus" series; functional emerald/red only for genuinely good/bad values.

### Anti-patterns (do not do)
- ❌ Bluish-purple SaaS gradients, gradient-mesh hero backgrounds.
- ❌ Emoji, hand-drawn fonts, playful bounces.
- ❌ Cards with a rounded body + single colored left border as the only structure.
- ❌ Pure `#FFFFFF` page backgrounds (use warm paper) or pure `#000` shadows.
- ❌ Copper as a large fill / background wash. It's an accent.
- ❌ Center-everything marketing layouts; prefer a structured, ruled, left-aligned editorial grid.

---

## ICONOGRAPHY

- **Library: [Lucide](https://lucide.dev)** (loaded from CDN: `https://unpkg.com/lucide@latest`). The live product hand-rolls inline outline SVGs in a Heroicons-ish style (stroke, 24×24, `stroke-width:2`, round caps/joins); Lucide is the same visual language — consistent outline geometry, 24px grid, 2px stroke — and is what this system standardizes on. *(Substitution flagged: the codebase has no packaged icon set of its own, so we adopt Lucide as the closest CDN match to its existing stroke icons.)*
- **Style rules:** outline only (no filled/duotone icon styles), `stroke-width: 1.75–2`, `currentColor` so icons inherit text color. Size with the type they sit beside — 16px inline, 18–20px in buttons/nav, 24px standalone. Never recolor an icon copper unless it is a deliberate accent moment.
- **Usage:** icons clarify, they don't decorate. One icon per action; no icon salad. Pair an icon with a label wherever space allows.
- **Brand marks** live in `assets/`: `wilfred-mark.svg` (forest tile + copper mark), `wilfred-mark-mono.svg` (single-color, `currentColor`), `wilfred-seal.svg` (the medallion/seal variant for premium and stamp-like moments). The mark is **The Transfer** — an arc carrying an invention from a solid origin node to an open destination node: technology transfer, lab to market.
- **No emoji as icons. No Unicode dingbats.** State chips use a Lucide glyph + word + color.

---

---

## What's in this folder

- **`CLAUDE.md`** — agent entry point: read order, hard rules, and how to use the tokens.
- **`cheatsheet.md`** — one page of exact values (color, type, space, shape).
- **`brand-guide.md`** — this file: product context, voice & copywriting, visual foundations, iconography.
- **`styles.css`** — single CSS entry point. Link it (or `@import` it) to get every token and all three webfonts. It imports the five files in `tokens/`.
- **`tokens/`** — design tokens (CSS custom properties):
  - `fonts.css` — webfont loading (Source Serif 4 + Hanken Grotesk via Google Fonts; Maple Mono via the Fontsource CDN).
  - `colors.css` — forest / copper / paper / slate ramps + semantic aliases; light default + dark scope.
  - `typography.css` — families, type scale, weights, leading, tracking, composite roles.
  - `spacing.css` — spacing grid, radii, borders, shadows, motion, layout rails.
  - `base.css` — element defaults + brand helper classes (`.wf-eyebrow`, `.wf-display`, `.wf-mono`, `.wf-rule`).
- **`assets/`** — `wilfred-mark.svg` (primary), `wilfred-mark-mono.svg` (single-color, `currentColor`), `wilfred-seal.svg` (formal/stamp).

> This is a portable brand reference. It intentionally omits the React component library and full-screen UI kits from the source design system — those live in the design-system project. If you need a component's exact markup, ask for it.

