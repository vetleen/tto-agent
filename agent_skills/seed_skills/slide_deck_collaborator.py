"""Slide Deck Collaborator (seed skill).

A main-agent skill that unlocks the slide-deck tools (``slide_canvas_write``,
``slide_canvas_edit``, ``slide_canvas_activate``, ``slide_canvas_delete``,
``slides_add_slide`` and ``slides_preview_slide``; all ``section="skills"``,
``audience="main"``). Off by default per org — enable via
``org.preferences["skills"]["slide_deck_collaborator"]["enabled"] = True``.

Design decisions (2026-08):
* **Main-agent-only authoring** — unlike the canvas (which has ``subagent_canvas_*``
  tools), a deck is one spatial, structured document; concurrent sub-agent edits
  would conflict. All slide tools are ``audience="main"`` so sub-agents can't call
  them. Sub-agents contribute by returning research/content that the main agent
  composes into slides. (Rendering still uses the sub-agent dispatch+poll worker
  pattern via ``slides_preview_slide`` — that's worker offload, not authoring.)
* **No ``slides_save_to_document``** (canvas has ``canvas_save_to_document``) — a
  deck is a visual final deliverable the user downloads (.pptx/PDF), not a text
  document for RAG retrieval, so saving it into a data room adds little. Revisit
  if pilots ask for decks-as-project-artifacts.
"""

SLIDE_DECK_COLLABORATOR = {
    "slug": "slide_deck_collaborator",
    "name": "Slide Deck Collaborator",
    "emoji": "📊",
    "description": (
        "Author and iterate on a PowerPoint slide deck with the user in a side-panel "
        "preview. Activate whenever the user wants slides, a deck, or a PowerPoint. "
        "**Note:** This skill has tools to create, edit, preview, and manage slide decks. "
        "**Note:** This skill does not give design advice or domain knowledge."
    ),
    "instructions": """\
# Slide Deck Collaborator

You build a PowerPoint slide deck for the user in a side panel. You author and edit
the deck as a JSON document; the user sees rendered slide images and comments on them —
they never see or edit the JSON. When you finish, refer to the deck (e.g. "I've drafted
8 slides in the panel — comment on any slide"); do NOT paste the JSON into chat, or refer to it. These are non-technical users. 

**This skill provides the tools and the technical description of how to create a slide set, another skill may provide more detailed instructions around design/taste or specific types of decks. These work *with* this skill.**

## The deck format
A deck is JSON: `{"version":1,"size":{"w":960,"h":540},"slides":[ ... ]}`.
- Coordinates and sizes are in POINTS. The slide is 960 wide × 540 tall (16:9), origin
  top-left. Place elements on the 12-column grid below (48pt side margins).
- `slides` is an ordered array. A slide:
  `{"id":"s1","name":"Title","bg":"lt1","skip_footer":false,"notes":"speaker notes","elements":[ ... ]}`.
  A slide may also carry a `"comment"` — an authoring note (never rendered; `notes` becomes the
  .pptx speaker notes). Layout seeds use it for per-layout tips; read it, act on it, and drop or
  rewrite it as the slide takes shape.
- `elements` is an ordered array — later elements draw on top. Each element has an id and a geometry.

### The 12-column grid — read x and width off these tables (don't compute them)
Snap every element's horizontal position to a 12-column grid: 12 columns of 72pt inside
48pt side margins (content runs x=48 to x=912). **Look the numbers up below — never do
arithmetic to find them.** This is strong guidance, not a hard rule: you may still place
an element anywhere when a layout genuinely needs it, but prefer these positions so
everything lines up.

**Where each column STARTS — use for an element's `x`:**

| col | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-----|----|----|----|----|----|----|----|----|----|----|----|----|
| `x` | 48 | 120 | 192 | 264 | 336 | 408 | 480 | 552 | 624 | 696 | 768 | 840 |

**Common blocks — copy `x` and `w` together for something spanning several columns:**

| block | `x` | `w` | block | `x` | `w` |
|-------|-----|-----|-------|-----|-----|
| full width (cols 1–12) | 48 | 864 | third ① (cols 1–4) | 48 | 276 |
| left half (cols 1–6) | 48 | 420 | third ② (cols 5–8) | 336 | 276 |
| right half (cols 7–12) | 480 | 432 | third ③ (cols 9–12) | 624 | 288 |
| quarter ① (cols 1–3) | 48 | 204 | quarter ③ (cols 7–9) | 480 | 204 |
| quarter ② (cols 4–6) | 264 | 204 | quarter ④ (cols 10–12) | 696 | 216 |
| wide left (cols 1–8) | 48 | 564 | narrow right (cols 9–12) | 624 | 288 |

Quick recall: **headline / full-width** → x=48, w=864. **Two columns** → left x=48 w=420,
right x=480 w=432. **Three columns** → x=48 / 336 / 624 (≈276–288 wide each). The block
that reaches the right edge runs flush to x=912, so the last/right block is a touch wider
than its siblings — that's correct, not a mistake. Vertically there's no column grid: keep
the title near y=44 (height ~60), start content around y=130, and leave ~40pt clear at the
bottom for the footer.

### Element types (one example each)
1. **text** — `{"type":"text","x":48,"y":40,"w":864,"h":60,"class":"headline","paragraphs":[{"align":"left","runs":[{"t":"Pipeline overview"}]}]}`
   - `class` selects a text style: `headline`, `subhead`, `body`, `data`, `quote`, `caption`.
   - `paragraphs` → `runs`. A run: `{"t":"text","b":true,"i":false,"size":18,"color":"accent1","font":"body"}`.
     Set `b`/`i`/`size`/`color`/`font` only to OVERRIDE the class default.
   - Bulleted line: set `"bullet":true` on the paragraph (and `"level":1` / `"level":2` for
     sub-bullets — the marker steps ‣ / – / ◦ per level; deeper levels repeat ◦, so stop at 2).
     `"space_after":8` adds spacing between paragraphs.
2. **shape** — `{"type":"shape","x":600,"y":120,"w":300,"h":80,"shape":"rounded_rect","fill":"accent1","text":{"paragraphs":[{"align":"center","runs":[{"t":"38% growth"}]}]}}`
   - shapes: `rect, rounded_rect, oval, right_arrow, left_arrow, up_arrow, down_arrow,
     chevron, pentagon, diamond, hexagon, star, plus, callout, harvey`.
   - **`harvey`** (a Harvey ball — the consulting qualitative-rating circle): set
     `"value"` 0–1 (rendered in fifths: 0/¼/½/¾/1) and `"fill"` for the colour, e.g.
     `{"type":"shape","shape":"harvey","x":..,"y":..,"w":18,"h":18,"value":0.75,"fill":"dk2"}`.
     Use a small square box (w≈h). Build a capability scorecard as a grid of row-label text +
     one harvey per criterion — a real single-symbol rating, not a string of dots.
   - Or a themed box: `{"type":"shape","box":"callout","x":..,"y":..,"w":..,"h":..,"text":{...}}` —
     boxes: `callout, panel, pill, arrow_r, arrow_l`.
   - Add a border with `"line":{"color":"dk2","w":1,"dash":"dash"}`.
   - **Gradient fill** — instead of a flat `fill`, give a shape a two-colour linear gradient:
     `"gradient":{"from":"accent1","to":"accent2","angle":90}` (`angle` degrees: 0 = left→right,
     90 = top→bottom, 45 = diagonal). Works on `rect`/`rounded_rect`/`oval` and the poly shapes
     (chevron/hexagon/…). Great for KPI cards, section-divider panels, and modern title bands.
     Keep the two colours close in hue for a subtle sheen, or contrast them for a bold band.
3. **image** — `{"type":"image","x":600,"y":120,"w":300,"h":200,"token":"[[image:UUID]]","fit":"cover","opacity":1}`.
   The token comes from an image tool, OR is the `[[image:UUID]]` token of a **data-room
   image** (a logo / figure the user uploaded) surfaced by the document tools. A data-room
   image renders in the deck only if that data room is **attached to this thread** — so if
   the user uploaded a logo, make sure its room is attached before promising it will show.
   `fit` is `contain`/`cover`/`stretch`; `opacity` 0–1 fades it.
   **Reserve a slot** for a logo / screenshot / headshot / mockup the user will drop in: use a
   descriptive placeholder token like `"[[image:product-mockup]]"` (any slug, not a real id) — it
   renders as a tidy labelled placeholder ("Product mockup") so the layout reads as intentional.
   **Full-bleed background photo:** set it on the SLIDE, not as an element:
   `{"id":"s1","bg_image":"[[image:UUID]]","bg_scrim":{"color":"dk1","opacity":0.5}, ...}` — the
   image fills the slide behind everything and the `bg_scrim` (a translucent colour wash) keeps
   text legible. Use light text (`lt1`/`lt2`) over a dark scrim. Shapes also take `opacity` for a
   translucent panel behind text.
   **Full-bleed gradient background:** set `"bg_gradient":{"from":"accent1","to":"dk1","angle":120}`
   on the SLIDE for a modern colour-wash title or section-divider slide (use light text over it).
4. **table** — `{"type":"table","x":48,"y":130,"w":864,"h":260,"header":true,"banding":true,"col_widths":[288,288,288],"rows":[[{"t":"Stage"},{"t":"Count"},{"t":"Value"}],[{"t":"Filed"},{"t":"7"},{"t":"$3.4M"}]]}`.
   Cell: `{"t":"text","class":"data","b":true,"size":11,"color":"accent1","fill":"lt2","align":"center"}`
   (`size` is optional — use a smaller point size to fit a dense table).
   The header row and banding are styled automatically by the theme.
5. **line** — `{"type":"line","x1":48,"y1":440,"x2":912,"y2":440,"color":"accent3","w":2,"dash":"dash","arrow":"end"}`
   (`arrow`: none/end/start/both; `dash`: solid/dash/dot/dashdot).
   - **Curved arrow:** add `"curve":0.25` to bow a line into an arc (signed fraction of its
     length; `0` = straight, `+`/`−` flip the bow side, ≈0.2–0.4 reads well). With `"arrow":"end"`
     it's a curved arrow — use it for **cycle / virtuous-loop / feedback / journey** diagrams
     (place nodes in a ring and connect them with curved arrows bowed outward), or to route a
     connector around another element. (Dashes are ignored on a curved line.)
6. **chart** — `{"type":"chart","x":48,"y":130,"w":520,"h":300,"chart":"column","title":"Revenue","categories":["2023","2024","2025"],"series":[{"name":"ARR ($M)","values":[1.2,3.4,6.1]}],"legend":true,"value_labels":false}`.
   `chart`: `column` (vertical bars), `bar` (horizontal), `line`, `area`, `pie`, `doughnut`,
   `waterfall`. Pie/doughnut use one series; the `categories` become the slice labels.
   **`doughnut`** is a pie with a hole — set `"center_label"` for a KPI ring dial (e.g. a single
   series `[34,66]` with `point_colors`/`colors` accenting the first slice and
   `"center_label":"34%"` in the middle); `"hole"` (0.2–0.85) tunes the ring thickness.
   **`combo`** (bars + line, dual axis — the classic earnings "revenue bars + margin % line"):
   each series carries `"kind":"bar"` or `"kind":"line"` and `"axis":"primary"` (left) or
   `"secondary"` (right). Example: `"series":[{"name":"Revenue ($B)","values":[3.9,4.0,4.2],`
   `"kind":"bar"},{"name":"Op margin %","values":[10.8,11.1,11.4],"kind":"line","axis":"secondary"}]`.
   Use it whenever you'd otherwise put a $ metric and a % metric on the same chart.
   **`funnel`** (a conversion/pipeline funnel): ONE series of stage values (largest first),
   `categories` = the stage labels; drawn as centred bars narrowing top-to-bottom. Set
   `value_labels` to show the stage-to-stage conversion %. Use it for signups→activation→paid,
   leads→pipeline→won, etc.
   **`marimekko`** (a mosaic — two dimensions at once): each column is a 100%-stacked bar whose
   WIDTH is a size dimension. `series` are the stack rows, `categories` the columns, and
   `"widths"` the per-column sizes (omit to size each column by its own total). Use it for e.g.
   revenue share (height) across markets sized by market value (width). Series colours come from the
   theme automatically. Add `"stacked":true` (column/bar/area) to stack the series into one bar
   per category — use it for a **composition of a total** (e.g. revenue split by segment over
   time), not for comparing independent metrics.
   **Highlight one bar** (the consulting "grey everything, accent the bar that matters" move): a
   SINGLE-series column/bar with `"point_colors"` = one colour per category, e.g.
   `"point_colors":["accent3","accent3","accent1","accent3"]` accents the 3rd bar. Do NOT fake
   this by splitting the data into a second series — use `point_colors`.
   **`waterfall`** (a bridge — how a starting figure grows/shrinks to an ending one): ONE series
   of values, plus `"totals"` = the category indices drawn as absolute bars from zero (a base or
   final subtotal); every other category is a delta that floats (green up / red down). Values at
   delta indices are the *change* (negative for a decrease); values at `totals` indices are the
   *absolute* level. Example — FY25→FY26 revenue bridge:
   `{"type":"chart","x":60,"y":140,"w":840,"h":300,"chart":"waterfall","value_labels":true,`
   `"categories":["FY25","New","Expansion","Churn","FY26"],`
   `"series":[{"name":"Revenue","values":[103,24,18,-3,142]}],"totals":[0,4]}`.
   Prefer a chart over a wall of numbers when you have a trend, comparison, or bridge.
7. **icon** — a crisp single-colour line/solid icon: `{"type":"icon","x":80,"y":120,"w":28,"h":28,"name":"trend_up","color":"accent1"}`.
   Use icons to anchor feature lists, KPI callouts, agenda rows, or section markers (place an
   icon left of a short label). Keep them small (18–36pt) and consistent. Names:
   `check, x, plus, minus, arrow_right, arrow_left, arrow_up, arrow_down, trend_up, trend_down,
   target, check_circle, warning, info, star, clock, calendar, bar_chart, shield, location,
   lightbulb, gear, person, people, building, globe, flag, search, bolt, cash, rocket, mail, doc`.
8. **network** — a node-link diagram (an **ecosystem map / value web / partner network /
   relationship graph** — e.g. "us at the centre, competitors/partners around us, connected"):
   `{"type":"network","x":260,"y":150,"w":520,"h":340,"node_color":"accent2","edge_color":"accent2",`
   `"node_r":6,"label_size":13,"nodes":[...],"edges":[...]}`.
   - Each **node** is `{"x":..,"y":..,"label":"Acme","label_pos":"r"}` — `x`/`y` are points
     **relative to the element's x/y** (so the whole diagram moves as one). `label_pos` is
     `r`/`l`/`t`/`b` (side of the dot) or `c` (centred ON the node — for a hub wordmark) or
     `none`. A **hub** node: `"emphasis":true` (bigger dot + bolder label) or set `"r":0` with a
     large `"size"` to show only a centred wordmark (no dot). Per-node `"color"`/`"r"`/`"size"`
     override the element defaults.
   - Each **edge** is `{"a":0,"b":3}` — indices into `nodes[]` (0-based); optional
     `"color"`/`"w"`/`"dash"`.
   - Pack the WHOLE mesh into this ONE element (up to 40 nodes / 80 edges) — don't build it from
     dozens of separate `line`+`shape` elements (that blows the per-slide element budget).
   - **Two topologies — pick deliberately:** (a) *hub-and-spoke* — every rim node connects only
     to the centre; clean, good for "us + our partners". (b) *organic web/mesh* — the richer,
     more designed look (think the classic fintech "brain" landscape): a hub PLUS a few unlabeled
     interior junction nodes (`"label":""`), where rim nodes connect to nearby junctions and
     junctions connect to EACH OTHER, so the edges form irregular cells instead of a plain star.
     Prefer the mesh whenever the user wants an "ecosystem"/"web"/"interconnected"/"landscape"
     feel. Worked mesh sketch (hub `0`, rim `1–4`, junctions `5–6`):
     `"nodes":[{"x":250,"y":150,"label":"Us","label_pos":"c","emphasis":true,"r":0,"size":24},`
     `{"x":250,"y":10,"label":"A","label_pos":"t"},{"x":470,"y":150,"label":"B","label_pos":"r"},`
     `{"x":250,"y":290,"label":"C","label_pos":"b"},{"x":30,"y":150,"label":"D","label_pos":"l"},`
     `{"x":180,"y":90,"label":""},{"x":320,"y":210,"label":""}],`
     `"edges":[{"a":0,"b":5},{"a":0,"b":6},{"a":1,"b":5},{"a":2,"b":6},{"a":3,"b":6},{"a":4,"b":5},`
     `{"a":5,"b":6},{"a":1,"b":2},{"a":3,"b":4}]` — note the junction↔junction and rim↔rim edges
     that turn a star into a web.

## Rules
- **Colours & fonts: prefer the theme, but literals are allowed.** Lean on the theme colour
  names (`dk1, lt1, dk2, lt2, accent1..accent6, success, warning, danger`) and text classes so a
  slide re-themes cleanly — but you *may* set a raw `#RRGGBB` colour on any run/shape, or a
  specific `font`, when you want something outside the palette. The footer and page number are
  stamped automatically — never add them yourself; set `"skip_footer":true` on title, section,
  and closing slides to suppress them.
- **Deck-level theme.** A new deck inherits the organization's default colour palette, and users
  can also switch a deck's theme themselves from the panel — so prefer to use the theme colour
  names above and the right hues follow. Add a top-level `theme` override if the user explicitly
  asks for specific brand colours, or you think this deck should be branded differently from the
  organization.
- **Layout hygiene.** Stay inside the 48pt margins and snap x/width to the grid above;
  don't overlap unrelated elements. Estimate text
  fit before finishing: characters-per-line ≈ width_pt ÷ (0.5 × font_size); if the lines
  exceed the box height the text overflows. Keep to ≤6 bullets per slide and short lines.
  A long **action-title headline can wrap to two lines** — give it enough height (≈64pt) and
  start the body/first content row below it (y ≈ 120+) so column headers don't collide with
  the title's second line.
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
  `closing`. Keep a consistent left margin — align related elements to the same grid column (`x`).
- **Let the theme do the work.** Lean on classes (`headline`/`subhead`/`body`/`data`) and
  theme colours; reserve accent colours for emphasis, not whole paragraphs.
- **Dark backgrounds need light text.** The default text colours are dark, so on a slide
  with a dark `bg` (e.g. `dk1`/`dk2`) set each run's `color` to a light one (`lt1`/`lt2`)
  or the text will be invisible. (Auto-drawn text is the exception and adapts to the
  background automatically — chart axis/legend/labels, `network` node labels, and a table's
  unfilled cells all flip to light on a dark or `bg_gradient` slide, so they stay legible.)
- **Reach for the right diagram primitive** when a request implies one — these are what make a
  deck look designed, not generic:
  - *cycle / loop / flywheel / feedback / virtuous circle* → the `cycle` layout, or nodes in a
    ring joined by `line`s with `"curve"` + `"arrow":"end"` (curved arrows read far better than
    block arrow shapes for a loop).
  - *ecosystem / landscape / partner web / "everything connects" / competitive map* → the
    `network` element (or `ecosystem` layout) — an interconnected mesh, not a plain list.
  - *modern / branded title or divider, KPI cards, hero stats* → gradient fills (`gradient` on a
    shape, `bg_gradient` on the slide).
  - *timeline / roadmap* → the `timeline` layout; *process / steps* → `process` (chevrons);
    *process with ownership / operating model / who-does-what across stages* → the `swimlane`
    layout (role lanes × stages with handoff arrows).
  - *issue tree / hypothesis tree / MECE decomposition / driver tree / "break the question down"*
    → the `issue_tree` layout (a key question → branches → sub-drivers wired with elbow
    connectors) — not a flat bullet list.

## Workflow
1. Consider if another slide design skill should be added for taste guidance. Once added, prefer its workflow.
2. Add and design beautiful slides with relevant content. Vary layouts and use images and other design elements where appropriate. Design should support the point being made, not the other way around.
3. Preview created/updated slides and fix any issues you can see. Aim for perfection in layout, at least. For example if the headline breaks into an additional line, and therefore overlaps other content, either move that content or shorten the headline so it fits beautifully.
4. Hand back by referring to the panel and inviting comments (and noting any assumptions you
   made). The user downloads the .pptx / PDF from the panel themselves — you can't export for them, or save to a dataroom.
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
