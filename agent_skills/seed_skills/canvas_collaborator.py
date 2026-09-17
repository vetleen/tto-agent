"""Canvas Collaborator (seed skill).

A main-agent skill that unlocks the canvas workspace tools (``canvas_activate``,
``canvas_write``, ``canvas_edit``, ``canvas_delete``, ``canvas_save_to_document``,
``canvas_paste_user_text`` and ``chat_attachment_open_to_canvas``; all
``section="skills"``, ``audience="main"``). Activating it lets the assistant draft
and edit documents in the side-panel canvas. Sub-agents have their own canvas
tools (``subagent_canvas_*``) and do not use these.
"""

CANVAS_COLLABORATOR = {
    "slug": "canvas_collaborator",
    "name": "Canvas Collaborator",
    "emoji": "📝",
    "description": (
        "Draft and edit text with the user in a side-panel canvas. Activate whenever "
        "writing, drafting, or revising substantial text, like a document, letter, report, grant application or memo "
        "— the canvas is the user-friendly way to deliver it. Also attach this skill if there is already text in a canvas you have access to. "
        "Avoid using the canvas if the text is a single paragraph that can just as well be delivered in the chat conversation with the user. "
        "**Note:** This skill has tools that enable creating, editing, deleting and otherwise managing the canvas."
    ),
    "instructions": """\
# Canvas Collaborator

After working in the canvas, don't reproduce the content in
chat — just refer to it (e.g. "I've drafted it in the canvas").

# Getting existing content into the canvas
Prefer copying text into the canvas over retyping it — retyping risks paraphrasing
or dropping detail. You have three verbatim-copy paths, and none of them require
you to reproduce the text yourself:

- **The user's own words** → `canvas_paste_user_text`. Your context lists the
  user's messages under `# Your messages`, numbered; pass that number to paste one
  in verbatim. It appends to the canvas (after an `anchor` phrase already in the
  canvas if you give one, otherwise at the end) and creates the canvas if needed.
- **A file the user attached to the chat** → `chat_attachment_open_to_canvas`.
  Your context lists them under `# Attachments`, numbered. Word/PDF/text files load
  as editable text; images can't (embed the `[[image:uuid]]` token instead).
- **A data-room document** → `document_open_to_canvas` (from the Data Room Tools
  skill), which opens the document's editable text into a canvas.

# Markdown
Use markdown in the canvas.

## Highlighting
You may wrap text in `==double equals==` to highlight it (e.g. `the ==key word== is`). It renders as a yellow \
highlight in the canvas and exports as a yellow highlight in .docx. Use primarily when the user asks you to \
highlight a section, or specific words. or when you want to draw attention to a specific section, for instance if \
the user asks where in a document the prior art search is mentioned, you might highlight that particular paragraph \
or sentence or whatever. For emphasis in text you produce, you would normally use italics or bold, not a highlight. \

## Diagrams
You can include Mermaid diagrams in the canvas using fenced code blocks with the `mermaid` language tag. These \
render as visual diagrams in preview mode and export as images in .docx. Example:

```mermaid
graph TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action]
    B -->|No| D[End]
```

Supported diagram types: `graph`/`flowchart`, `sequenceDiagram`, `classDiagram`, `stateDiagram-v2`, `erDiagram`, `gantt`, `pie`, `quadrantChart`, `gitgraph`, `timeline`, `mindmap`, `sankey-beta`, `xychart-beta`, `block-beta`. Do NOT use unsupported types such as `radarChart`, `radar`, or `spider` — Mermaid does not support radar/spider charts. If you need to visualise scores across dimensions (e.g., readiness levels), use a table or a `xychart-beta` bar chart instead.


""",
    "tool_names": [
        "canvas_activate",
        "canvas_write",
        "canvas_edit",
        "canvas_delete",
        "canvas_save_to_document",
        "canvas_paste_user_text",
        "chat_attachment_open_to_canvas",
    ],
}
