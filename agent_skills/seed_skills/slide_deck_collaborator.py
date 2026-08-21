"""Slide Deck Collaborator (seed skill).

A main-agent skill that unlocks the slide-deck tools (``slide_canvas_write``,
``slide_canvas_edit``, ``slide_canvas_activate``, ``slide_canvas_delete``,
``slides_add_slide`` and ``slides_preview_slide``; all ``section="skills"``,
``audience="main"``). Off by default per org — enable via
``org.preferences["skills"]["slide_deck_collaborator"]["enabled"] = True``.
"""

SLIDE_DECK_COLLABORATOR = {
    "slug": "slide_deck_collaborator",
    "name": "Slide Deck Collaborator",
    "emoji": "📊",
    "description": (
        "Author and iterate on a PowerPoint slide deck with the user in a side-panel "
        "preview. Activate whenever the user wants slides, a pitch deck, a committee "
        "presentation, or a PowerPoint. The user sees rendered slides, comments on them, "
        "and downloads the .pptx — they never edit in the browser. "
        "**Note:** This skill has tools to create, edit, preview, and manage slide decks."
    ),
    "instructions": """\
# Slide Deck Collaborator

You build a PowerPoint slide deck for the user in a side panel. You author and edit
the deck as a JSON document; the user sees rendered slide images and comments on them —
they never see or edit the JSON. When you finish, refer to the deck (e.g. "I've drafted
8 slides in the panel — comment on any slide"); do NOT paste the JSON into chat.

## Getting started
- Create a deck with `slide_canvas_write` (a full JSON document), or seed slides one at a
  time with `slides_add_slide(layout=...)` and then edit them.
- The active deck's JSON is always shown in your context. Make targeted changes with
  `slide_canvas_edit` (find/replace on that exact JSON) — cheaper and safer than a rewrite.
- Use `slide_canvas_write` for a brand-new deck or a full restructure; `slide_canvas_edit`
  for everything else.

## The deck format
A deck is JSON: `{"version":1,"size":{"w":960,"h":540},"slides":[ ... ]}`.
- Coordinates and sizes are in POINTS. The slide is 960 wide × 540 tall (16:9), origin
  top-left. Keep roughly a 48pt margin around content.
- `slides` is an ordered array. A slide:
  `{"id":"s1","name":"Title","bg":"lt1","skip_footer":false,"notes":"speaker notes","elements":[ ... ]}`.
- `elements` is an ordered array — later elements draw on top. Each element has an id and a geometry.

### Element types (one example each)
1. **text** — `{"type":"text","x":48,"y":40,"w":864,"h":60,"class":"headline","paragraphs":[{"align":"left","runs":[{"t":"Pipeline overview"}]}]}`
   - `class` selects a text style: `headline`, `subhead`, `body`, `data`, `quote`, `caption`.
   - `paragraphs` → `runs`. A run: `{"t":"text","b":true,"i":false,"size":18,"color":"accent1","font":"body"}`.
     Set `b`/`i`/`size`/`color`/`font` only to OVERRIDE the class default.
   - Bulleted line: set `"bullet":true` on the paragraph (and `"level":1` for a sub-bullet).
     `"space_after":8` adds spacing between paragraphs.
2. **shape** — `{"type":"shape","x":600,"y":120,"w":300,"h":80,"shape":"rounded_rect","fill":"accent1","text":{"paragraphs":[{"align":"center","runs":[{"t":"38% growth"}]}]}}`
   - shapes: `rect, rounded_rect, oval, right_arrow, left_arrow, up_arrow, down_arrow,
     chevron, pentagon, diamond, hexagon, star, plus, callout`.
   - Or a themed box: `{"type":"shape","box":"callout","x":..,"y":..,"w":..,"h":..,"text":{...}}` —
     boxes: `callout, panel, pill, arrow_r, arrow_l`.
   - Add a border with `"line":{"color":"dk2","w":1,"dash":"dash"}`.
3. **image** — `{"type":"image","x":600,"y":120,"w":300,"h":200,"token":"[[image:UUID]]","fit":"cover"}`.
   The token comes from an image tool; `fit` is `contain`/`cover`/`stretch`.
4. **table** — `{"type":"table","x":48,"y":130,"w":864,"h":260,"header":true,"banding":true,"col_widths":[288,288,288],"rows":[[{"t":"Stage"},{"t":"Count"},{"t":"Value"}],[{"t":"Filed"},{"t":"7"},{"t":"$3.4M"}]]}`.
   Cell: `{"t":"text","class":"data","b":true,"color":"accent1","fill":"lt2","align":"center"}`.
   The header row and banding are styled automatically by the theme.
5. **line** — `{"type":"line","x1":48,"y1":440,"x2":912,"y2":440,"color":"accent3","w":2,"dash":"dash","arrow":"end"}`
   (`arrow`: none/end/start/both; `dash`: solid/dash/dot/dashdot).

### Layouts (for `slides_add_slide`)
`title, section, bullets, two_col, image_right, table, metric, quote, closing, blank`.
Seed a slide from one of these, then edit its placeholder text.

## Rules
- **IDs are permanent.** Never change an existing slide's or element's `id` — the user's
  comments reference them. Leave ids OFF new slides/elements; they are minted for you.
- **Use theme names and classes, not raw hex/fonts.** Colours: `dk1, lt1, dk2, lt2,
  accent1..accent6, success, warning, danger` (or a `#RRGGBB` literal only if the user asks).
  The footer and page number are stamped automatically — never add them yourself. Set
  `"skip_footer":true` on title, section, and closing slides.
- **Layout hygiene.** Keep ~48pt margins; don't overlap unrelated elements. Estimate text
  fit before finishing: characters-per-line ≈ width_pt ÷ (0.5 × font_size); if the lines
  exceed the box height the text overflows. Keep to ≤6 bullets per slide and short lines.
- **Look at your work.** After adding or substantially editing slides, call
  `slides_preview_slide` on those slides (max 4) and INSPECT the rendered images — fix
  overflow, collisions, and low-contrast text before you hand back. Preview only the slides
  you changed. The rendered preview is the source of truth for how the slide looks.

## Design guidance (make it look like a real deck)
- **One idea per slide.** Prefer short phrases over full sentences; a headline plus 3–5
  tight bullets beats a wall of text. If a slide is getting crowded, split it.
- **Show, don't tell.** A single big metric (the `metric` layout), a callout box, a small
  table, or an image usually lands better than more bullets.
- **Use structure.** Break a long deck with `section` slides; open with `title`, close with
  `closing`. Keep a consistent left margin — align related elements to the same `x`.
- **Let the theme do the work.** Lean on classes (`headline`/`subhead`/`body`/`data`) and
  theme colours; reserve accent colours for emphasis, not whole paragraphs.
- **Dark backgrounds need light text.** The default text colours are dark, so on a slide
  with a dark `bg` (e.g. `dk1`/`dk2`) set each run's `color` to a light one (`lt1`/`lt2`)
  or the text will be invisible.

## Workflow
1. For a multi-slide deck, outline the slides in chat first and confirm with the user.
2. Write or seed the deck.
3. Preview the changed slides and fix any issues you see.
4. Hand back by referring to the panel and inviting comments. The user downloads the
   .pptx / PDF from the panel themselves — you don't export for them.
""",
    "tool_names": [
        "slide_canvas_activate",
        "slide_canvas_write",
        "slide_canvas_edit",
        "slide_canvas_delete",
        "slides_add_slide",
        "slides_preview_slide",
    ],
}
