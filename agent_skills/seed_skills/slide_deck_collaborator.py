"""Slide Deck Collaborator (seed skill).

A main-agent skill that unlocks the slide-deck tools (``slides_create_deck``,
``slide_canvas_write``, ``slide_canvas_edit``, ``slide_canvas_activate``,
``slide_canvas_delete``, ``slides_add_slide``, ``slides_preview_slide``,
``slides_list_themes`` and ``slides_set_theme``; all ``section="skills"``,
``audience="main"``). Off by default per org — enable via
``org.preferences["skills"]["slide_deck_collaborator"]["enabled"] = True``.

The element vocabularies the instructions advertise (shapes, themed boxes, chart
kinds, icon names) are GENERATED from the renderer/schema modules, so what the
model is told exists is exactly what validates and renders. The grid tables and
the diagram recipes are hand-written mirrors of ``chat.slides.layouts`` — keep
them in sync when the grid changes.

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
* **Layouts vs. recipes (2026-09)** — the layout catalogue is kept short (one
  cover, one divider, the content workhorses, and the connector-heavy diagrams
  whose geometry is hard to re-derive: swimlane, roadmap_gantt, issue_tree).
  Simpler diagrams (process chevrons, cycle, timeline, 2×2 matrix, ecosystem) and
  the table live here as *recipes* with exact coordinates, built on a `bullets`
  or `blank` slide, so the model isn't choosing between 28 near-duplicates.
"""

from chat.slides.icons import ICON_NAMES
from chat.slides.schema import CHART_KINDS, SHAPE_NAMES
from chat.slides.theme import WILFRED_BASE_THEME

_INSTRUCTIONS = """\
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
  .pptx speaker notes). Layout seeds use it for per-layout tips and pre-designed OPTIONAL
  elements (a logo slot, a section number, a comparison variant…); read it, act on it, and
  drop or rewrite it as the slide takes shape.
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
the title near y=44 (height ~60), start content around y=130, and end it by y≈480. The
strip y=486–508 is reserved for the source line (see *Sources & the takeaway bar* below);
the footer band is stamped under that.

### Element types (one example each)
1. **text** — `{"type":"text","x":48,"y":40,"w":864,"h":60,"class":"headline","paragraphs":[{"align":"left","runs":[{"t":"Pipeline overview"}]}]}`
   - `class` selects a text style: `headline`, `subhead`, `body`, `data`, `quote`, `caption`.
   - `paragraphs` → `runs`. A run: `{"t":"text","b":true,"i":false,"size":18,"color":"accent1","font":"body"}`.
     Set `b`/`i`/`size`/`color`/`font` only to OVERRIDE the class default.
   - Bulleted line: set `"bullet":true` on the paragraph (and `"level":1` / `"level":2` for
     sub-bullets — the marker steps ‣ / – / ◦ per level; deeper levels repeat ◦, so stop at 2).
     `"space_after":8` adds spacing between paragraphs.
2. **shape** — `{"type":"shape","x":600,"y":120,"w":300,"h":80,"shape":"rounded_rect","fill":"accent1","text":{"paragraphs":[{"align":"center","runs":[{"t":"38% growth"}]}]}}`
   - `shape` must be one of: `__SHAPE_NAMES__`. These are ALL the shapes there are — each
     renders identically in the preview and the downloaded .pptx; anything else is rejected.
     (`plus` is a thin maths plus, `cross` a fat Greek cross.)
   - **Text inside a non-rectangular shape** is laid out in the shape's inner text box, not
     its full outline — a diamond's is half its width and height, an arrow's is its shaft, a
     chevron's excludes the notches, an oval's is the inscribed box. The preview wraps exactly
     where PowerPoint does, so size the shape for its label (keep labels short) and check it.
   - **Shadows:** shapes are flat by default. For a soft drop shadow on EVERY shape in the deck
     (all-or-nothing, so the deck stays consistent) set `"theme":{"effects":{"shape_shadow":true}}`
     at deck level with `slide_canvas_edit`; it survives a theme switch. Never fake one per shape.
   - **`chevron`** — the notch/tip depth is half the shape's SHORT side (48pt at the usual
     h=96). To interlock a chain, start each chevron `notch − 8` before the previous one
     ends: e.g. w=246 at x=48/254/460/666 fills the full content width (see the *Process*
     recipe below). Steps are peers — give them all the same fill and recolour a single
     chevron only to spotlight it.
   - **`harvey`** (a Harvey ball — the consulting qualitative-rating circle): set
     `"value"` 0–1 (rendered in fifths: 0/¼/½/¾/1) and `"fill"` for the colour, e.g.
     `{"type":"shape","shape":"harvey","x":..,"y":..,"w":18,"h":18,"value":0.75,"fill":"dk2"}`.
     Use a small square box (w≈h). Build a capability scorecard as a grid of row-label text +
     one harvey per criterion — a real single-symbol rating, not a string of dots.
   - Or a themed box: `{"type":"shape","box":"callout","x":..,"y":..,"w":..,"h":..,"text":{...}}` —
     boxes: `__BOX_NAMES__` (a box picks its shape, fill and text colour from the theme;
     `callout` is an accent rounded panel, `panel` a pale one, `takeaway` the conclusion bar,
     `pill` a small dark tag, `arrow_r`/`arrow_l` labelled block arrows).
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
4. **table** — a title-over-table slide is a `bullets` slide with the bullets replaced by this
   (plus the source line):
   `{"type":"table","x":48,"y":130,"w":864,"h":300,"header":true,"banding":true,"col_widths":[312,276,276],"rows":[[{"t":"Column A"},{"t":"Column B"},{"t":"Column C"}],[{"t":"Row 1"},{"t":"—"},{"t":"—"}],[{"t":"Row 2"},{"t":"—"},{"t":"—"}]]}`.
   Cell: `{"t":"text","class":"data","b":true,"size":11,"color":"accent1","fill":"lt2","align":"center"}`.
   The header row and banding are styled automatically by the theme; colour cells (`fill`) to
   make a heat-map.
   **Sizing a table:** `col_widths` are RELATIVE — they are scaled to fit `w` — so give the
   label column the most room (e.g. `[312,276,276]` for three columns, `[240,208,208,208]` for
   four) and narrow it (`w`, still on the grid) for a small table. Every row is a fixed ≈26pt
   tall whatever `h` says (`h` is only the box hint), so budget 26pt per row: y=130 + 26 × rows
   ≤ 480 ⇒ at most ~13 rows including the header — split a longer table across slides rather
   than shrinking it. Set `h` to rows × 26 so what you place below it doesn't collide. Use a
   cell `size` of 10–11 only to keep long text on one line.
5. **line** — `{"type":"line","x1":48,"y1":440,"x2":912,"y2":440,"color":"accent3","w":2,"dash":"dash","arrow":"end"}`
   (`arrow`: none/end/start/both; `dash`: solid/dash/dot/dashdot).
   - **Curved arrow:** add `"curve":0.25` to bow a line into an arc (signed fraction of its
     length; `0` = straight, `+`/`−` flip the bow side, ≈0.2–0.4 reads well). With `"arrow":"end"`
     it's a curved arrow — use it for **cycle / virtuous-loop / feedback / journey** diagrams
     (place nodes in a ring and connect them with curved arrows bowed outward), or to route a
     connector around another element. (Dashes are ignored on a curved line.)
6. **chart** — `{"type":"chart","x":48,"y":130,"w":520,"h":300,"chart":"column","title":"Revenue","categories":["2023","2024","2025"],"series":[{"name":"ARR ($M)","values":[1.2,3.4,6.1]}],"legend":true,"value_labels":false}`.
   `chart` must be one of: `__CHART_KINDS__` — i.e. `column` (vertical bars), `bar` (horizontal),
   `line`, `area`, `pie`, `doughnut`, `waterfall`, `scatter`, `histogram`, `dot`, `bullet`,
   `marimekko`, `funnel`, `combo`.
   Pie/doughnut use one series; the `categories` become the slice labels.
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
   **`scatter`** (relationship between two measures): each series carries `"points"` — a list of
   `[x, y]` (or `[x, y, size]` for a **bubble**, radius ∝ size). Multiple series read as coloured
   groups. Example: `"series":[{"name":"Deals","points":[[3,120],[5,90],[8,200]]}]`. Use it for
   "does X drive Y"; reach for bubble only when a 3rd variable matters.
   **`histogram`** (a distribution): ONE series of RAW values; set `"bins"` (2–50, or omit for an
   auto count) and it buckets them into contiguous columns. Example:
   `{"chart":"histogram","bins":8,"series":[{"values":[12,15,18,22,25,...]}]}`. Use it for "what's
   the spread" — not a mean-only bar.
   **`dot`** (a dot plot / lollipop — the clean alternative to a sorted bar): `categories` +
   `series` values like a bar, drawn as a dot at each value. **Sort the categories yourself** for a
   ranked dot plot. Multiple series ⇒ several dots per row (a good before/after or A-vs-B).
   **`bullet`** (distance from target): `categories` = KPI rows, `series[0].values` = the actual,
   `"targets"` = the target per row (each in its OWN units — $, count, months). Each row is scaled
   to its own target, so different-unit KPIs sit together cleanly; `"bands"` are ascending grey
   qualitative zones expressed as FRACTIONS of the target (e.g. `[0.6,0.85,1.0]` = 60/85/100%, omit
   for a 50/75/100% default). The row label sits above its bar. Example:
   `{"chart":"bullet","categories":["Revenue ($M)","NPS"],"series":[{"values":[8.2,58]}],`
   `"targets":[10,60],"bands":[0.6,0.85,1.0]}`. Use it instead of a gauge/speedometer.
   Prefer a chart over a wall of numbers when you have a trend, comparison, or bridge.

   **Which chart for which question** (starting point → *usually avoid*):
   - Change over time → `line` (slope = a 2-point line). *Avoid dozens of clustered columns.*
   - Which is larger / ranking → sorted `bar` or `dot` (sort the data). *Avoid an unsorted or
     many-slice pie.*
   - Distance from target → `bullet`. *Avoid a gauge/speedometer.*
   - Distribution → `histogram`. *Avoid a mean-only bar.*
   - Relationship of two measures → `scatter` (`bubble` only if a 3rd variable matters). *Avoid a
     dual-axis chart as the default.*
   - Composition / share → `stacked` bar or `marimekko`. *Avoid multiple pies.*
   - Contribution to a change → `waterfall`. *Avoid decorative arrows.*
   - $ metric + % metric together → `combo`. Pipeline stages → `funnel`.
   - Exact values → a `table` (colour cells for a heat-map). Schedule → the `roadmap_gantt` layout.
7. **icon** — a crisp single-colour line/solid icon: `{"type":"icon","x":80,"y":120,"w":28,"h":28,"name":"trend_up","color":"accent1"}`.
   Use icons to anchor feature lists, KPI callouts, agenda rows, or section markers (place an
   icon left of a short label). Keep them small (18–36pt) and consistent. `name` must be one of
   these — the complete set (an unknown name draws nothing):
   `__ICON_NAMES__`.
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
     to the centre; clean, good for "us + our partners" (the *Ecosystem* recipe below). (b)
     *organic web/mesh* — the richer, more designed look (think the classic fintech "brain"
     landscape): a hub PLUS a few unlabeled interior junction nodes (`"label":""`), where rim
     nodes connect to nearby junctions and junctions connect to EACH OTHER, so the edges form
     irregular cells instead of a plain star. Prefer the mesh whenever the user wants an
     "ecosystem"/"web"/"interconnected"/"landscape" feel. Worked mesh sketch (hub `0`, rim
     `1–4`, junctions `5–6`):
     `"nodes":[{"x":250,"y":150,"label":"Us","label_pos":"c","emphasis":true,"r":0,"size":24},`
     `{"x":250,"y":10,"label":"A","label_pos":"t"},{"x":470,"y":150,"label":"B","label_pos":"r"},`
     `{"x":250,"y":290,"label":"C","label_pos":"b"},{"x":30,"y":150,"label":"D","label_pos":"l"},`
     `{"x":180,"y":90,"label":""},{"x":320,"y":210,"label":""}],`
     `"edges":[{"a":0,"b":5},{"a":0,"b":6},{"a":1,"b":5},{"a":2,"b":6},{"a":3,"b":6},{"a":4,"b":5},`
     `{"a":5,"b":6},{"a":1,"b":2},{"a":3,"b":4}]` — note the junction↔junction and rim↔rim edges
     that turn a star into a web.

### Diagram recipes — build these from elements on a `bullets` (or `blank`) slide
Verified geometry that renders cleanly on the grid; copy the numbers rather than inventing
your own. Each sits under the usual headline (x=48 y=44 w=864 h=60).
- **Process / steps (chevrons).** Four interlocking chevrons across the content width: shape
  `chevron` w=246 h=96 y=176 at x=48/254/460/666, fill accent1, a centred bold lt1 label in each;
  a `body` size-13 caption under each step (w=190 h=90 y=292 at x=56/262/468/674). Three steps:
  w=314 at x=48/322/596 (captions w=250); five: w=204 at x=48/212/376/540/704 (captions w=150).
  Steps are peers — same fill for all; recolour one (e.g. dk2) only to spotlight it.
- **Cycle / flywheel.** A ring: shape `oval` x=352 y=179 w=256 h=256, no fill, `"line":{"color":
  "accent3","w":2}`; the loop's name centred inside it (`subhead` size 20, x=380 y=294 w=200 h=30,
  align center). Four station dots (shape `oval` 20×20, fill dk2) ON the ring at (470,169) N /
  (598,297) E / (470,425) S / (342,297) W, each with a `subhead` size-18 dk2 label reading
  outward: N x=380 y=112 w=200 h=30 align center valign bottom; E x=632 y=292 w=190 h=30 valign
  middle; S x=380 y=456 w=200 h=30 align center; W x=138 y=292 w=190 h=30 align right valign
  middle. To show direction, join neighbouring dots with `line`s using `"curve":0.25` and
  `"arrow":"end"` (curved arrows read far better than block arrows for a loop).
- **Timeline / milestones on a line.** One `line` x1=48 y1=256 x2=912 y2=256 color accent3 w=2;
  at the start of each column x=48/264/480/696 a dot (shape `oval` 20×20 y=246 fill accent1),
  the milestone label (`subhead` size 20 color dk2, y=292 w=180 h=30) and one line of detail
  (`caption` size 12, y=328 w=180 h=70) hanging left-aligned below it. Five milestones:
  x=48/221/394/567/740 with w=160. (Several parallel workstreams → the `roadmap_gantt` layout.)
- **2×2 matrix.** Four 276×148 tiles (shape `rect`) at (264,140) (552,140) (264,296) (552,296):
  three fill lt2 with dk2 text, the quadrant that carries the recommendation fill accent1 with lt1
  text. Per tile a `body` bold size-15 label at (tile.x+20, tile.y+16) w=236 h=28 and a size-12
  one-liner at (tile.x+20, tile.y+44) w=236 h=60. Axis labels in `body` bold 12, align center:
  x-axis "Effort →" at x=264 y=456 w=564 h=24; y-axis "Impact →" at x=150 y=284 w=140 h=24 with
  `"rotation":-90`.
- **Ecosystem / partner map (hub-and-spoke).** One `network` element x=210 y=150 w=560 h=330,
  node_color/edge_color accent2, node_r 6, label_size 13; hub
  `{"x":270,"y":150,"label":"Us","label_pos":"c","emphasis":true,"r":0,"size":26,"color":"dk1"}`;
  rim nodes (270,18) label_pos t / (500,70) r / (520,240) r / (270,300) b / (40,240) l / (20,70) l;
  edges from the hub to every rim node plus the rim ring 1-2, 2-3, 3-4, 4-5, 5-6, 6-1. For the
  richer web look use the mesh sketch under element 8.
- **Table** — see element 4. **A-vs-B comparison** — the `two_col` layout's comment carries the
  variant. **Issue / driver tree**, **swimlane** and **Gantt roadmap** stay layouts (their
  connectors are fiddly): `issue_tree`, `swimlane`, `roadmap_gantt`.

## Rules
- **Prefer the premade layouts.** Create slides from the pre-designed layouts (the `layouts`
  list of `slides_create_deck`, or `slides_add_slide`) whenever you add a slide, and build
  diagrams from the recipes above; reach for the `blank` layout or a bespoke
  `slide_canvas_write` rewrite only when no premade layout or recipe fits.
- **Colours & fonts: prefer the theme, but literals are allowed.** Lean on the theme colour
  names (`dk1, lt1, dk2, lt2, accent1..accent6, success, warning, danger`) and text classes so a
  slide re-themes cleanly — but you *may* set a raw `#RRGGBB` colour on any run/shape, or a
  specific `font`, when you want something outside the palette. The footer and page number are
  stamped automatically — never add them yourself; set `"skip_footer":true` on title, section,
  and closing slides to suppress them.
- **Deck-level theme.** A deck has a named theme — palette, fonts, tables, and a footer band. A
  new deck already carries the default theme (the user's default, else the org's, else the
  built-in Forest) — keep it unless the user wants a different look. Five built-in themes exist
  (Forest, Slate, Warm sand, Monochrome, Ocean) and the organization and the user may have saved
  more: `slides_list_themes` lists them all and `slides_set_theme` applies one to the deck. To
  re-palette just this deck without a named theme, edit the deck's top-level `theme.colors`
  object — a sparse override merged onto the theme, e.g. `"theme":{"colors":{"accent1":"#B87333"}}`
  (colours only; fonts are fixed by the theme). You cannot create or save named themes — the
  user does that from the panel. Prefer theme colour NAMES in slides so a re-theme stays coherent.
  If the current theme carries a logo, place it where it fits (e.g. the cover) with the token
  `[[image:company-logo]]`.
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
  theme colours; reserve accent colours for emphasis, not whole paragraphs. Colour carries
  meaning, never decoration: peer objects (process steps, roadmap bars, tiles, KPI numbers)
  share ONE colour, and you vary it only to encode something or to highlight the one that
  matters (like the accented quadrant in the 2×2 matrix recipe).
- **Sources & the takeaway bar.** The strip y=486–508, above the stamped footer, is
  reserved for a one-line source note — every chart/table/data slide should carry one
  (the `chart` layout seeds it; add the same element on table and bespoke data slides):
  `{"type":"text","x":48,"y":488,"w":864,"h":20,"class":"caption","paragraphs":[{"runs":[{"t":"Source: Eurostat energy balance, 2024; team analysis.","size":9}]}]}`.
  When a slide needs an explicit one-line conclusion (the "so what"), add the themed
  takeaway bar just above the source line and end the body content by y≈438:
  `{"type":"shape","box":"takeaway","x":48,"y":446,"w":864,"h":34,"text":{"paragraphs":[{"align":"center","runs":[{"t":"The one-line takeaway of the slide."}]}]}}`.
  Keep it to one line; on slides without a source line it may sit lower (y=454).
- **Dark backgrounds need light text.** The default text colours are dark, so on a slide
  with a dark `bg` (e.g. `dk1`/`dk2`) set each run's `color` to a light one (`lt1`/`lt2`)
  or the text will be invisible. (Auto-drawn text is the exception and adapts to the
  background automatically — chart axis/legend/labels, `network` node labels, and a table's
  unfilled cells all flip to light on a dark or `bg_gradient` slide, so they stay legible.)
- **Reach for the right diagram primitive** when a request implies one — these are what make a
  deck look designed, not generic:
  - *cycle / loop / flywheel / feedback / virtuous circle* → the *Cycle* recipe (a ring of
    stations joined by curved arrows — far better than block arrow shapes for a loop).
  - *ecosystem / landscape / partner web / "everything connects" / competitive map* → the
    `network` element (the *Ecosystem* recipe, or the mesh sketch) — an interconnected mesh,
    not a plain list.
  - *modern / branded title or divider, KPI cards, hero stats* → gradient fills (`gradient` on a
    shape, `bg_gradient` on the slide).
  - *single-track timeline / milestones on a line* → the *Timeline* recipe; *multi-workstream
    schedule / roadmap / Gantt (tracks × time with duration bars)* → the `roadmap_gantt` layout;
    *process / steps* → the *Process* recipe (chevrons); *process with ownership / operating
    model / who-does-what across stages* → the `swimlane` layout (role lanes × stages with
    handoff arrows); *prioritisation / two-criteria positioning* → the *2×2 matrix* recipe.
  - *issue tree / hypothesis tree / MECE decomposition / driver tree / "break the question down"*
    → the `issue_tree` layout (a key question → branches → sub-drivers wired with elbow
    connectors) — not a flat bullet list.

## Workflow
1. Consider if another slide design skill should be added for taste guidance. Once added, prefer its workflow.
2. Create the deck with `slides_create_deck`, passing the ordered list of layout ids as your
   storyboard — plan the narrative first and pick one layout per point. The result returns the
   full seeded deck JSON (minted ids, placeholder text); fill the placeholders in with
   `slide_canvas_edit`. (`slide_canvas_write` is only for rebuilding an existing deck around a
   new narrative, never for creating one.)
3. Add and design beautiful slides with relevant content. Vary layouts and use images and other design elements where appropriate. Design should support the point being made, not the other way around.
4. Preview created/updated slides and fix any issues you can see. Aim for perfection in layout, at least. For example if the headline breaks into an additional line, and therefore overlaps other content, either move that content or shorten the headline so it fits beautifully.
5. Hand back by referring to the panel and inviting comments (and noting any assumptions you
   made). The user downloads the .pptx / PDF from the panel themselves — you can't export for them, or save to a dataroom.
"""


def _render_instructions() -> str:
    """Fill the generated vocabularies into the hand-written instructions."""
    return (
        _INSTRUCTIONS
        .replace("__SHAPE_NAMES__", ", ".join(SHAPE_NAMES))
        .replace("__BOX_NAMES__", ", ".join(WILFRED_BASE_THEME["boxes"]))
        .replace("__CHART_KINDS__", ", ".join(CHART_KINDS))
        .replace("__ICON_NAMES__", ", ".join(ICON_NAMES))
    )


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
    "instructions": _render_instructions(),
    "tool_names": [
        "slide_canvas_activate",
        "slides_create_deck",
        "slide_canvas_write",
        "slide_canvas_edit",
        "slide_canvas_delete",
        "slides_add_slide",
        "slides_preview_slide",
        "slides_list_themes",
        "slides_set_theme",
    ],
}
