"""Meeting Summarizer seed skill."""

MEETING_SUMMARIZER = {
    "slug": "meeting-summarizer",
    "name": "Meeting Summarizer",
    "description": (
        "Turns a meeting transcript into faithful, well-structured minutes or "
        "a summary. Use when the user opens a meeting in chat to draft minutes, "
        "asks to summarize a transcript, extract decisions or action items, or "
        "produce a recap of a recorded conversation. Also use when the user "
        "mentions 'meeting notes', 'meeting minutes', 'recap', 'takeaways', or "
        "'what was decided'. Works for any meeting type — board meetings, "
        "standups, interviews, brainstorms, workshops, sales calls."
    ),
    "instructions": """\
# Meeting Summarizer

Produce a faithful, readable record of a meeting from its transcript. A \
transcript is required; if you don't have it ask for it. Meeting metadata (name, \
agenda, participants, duration, meeting_id) should also be provided by the user, but is not critical.

## Minutes or summary?

- **Minutes** — a structured record of what happened, organized by topic. \
Default for formal or decision-making meetings.
- **Summary** — a short narrative recap of the key points. Better for informal \
meetings, brainstorms, interviews, or quick recaps.

Infer which one fits from the meeting type. Ask only if genuinely ambiguous. \
The `save_meeting_minutes` tool accepts `kind="minutes"`, `"summary"`, or \
`"notes"` — pick the one that matches what you produced.

## Workflow

1. **Read the transcript end-to-end before drafting.** Note the meeting's \
actual shape — decision-making, status update, brainstorm, interview, pitch. \
That shape determines the right structure, not a fixed template.

2. **Draft to a new canvas** using `write_canvas` with a title like \
`Meeting minutes — <meeting name>` (or `Meeting summary — <meeting name>`). \
A new title creates a new tab — do not overwrite the transcript canvas.

3. **Build the structure from what's actually in the transcript.** Include a \
section only if there is real content for it — never pad with empty headings. \
A sensible default to adapt:
   - **Header**: meeting name, date, duration, known participants
   - **Key discussion points**, grouped by topic — usually the bulk of the document
   - **Decisions** — only if any were made, one line each with enough context \
to stand alone later
   - **Action items** — only if any were assigned; format as *owner — task — \
due date*, omitting fields that were not stated
   - **Open questions** — only if any were flagged as unresolved

   For a summary, collapse the middle into a tight narrative or a handful of \
bullets and skip the header table unless it adds value.

4. **Stay faithful to the transcript.**
   - Never invent participants, decisions, action items, quotes, or numbers.
   - If something is ambiguous, omit it or mark it explicitly \
(e.g. *unclear from transcript*).
   - Quote verbatim only when wording matters — exact numbers, commitments, \
product names, legal language. Otherwise paraphrase tightly.
   - Report, don't editorialize. "The team discussed pricing" — not \
"The team had a productive discussion about pricing".

5. **Iterate, then save.** Offer to refine the draft. When the user is \
satisfied, call `save_meeting_minutes` with the final canvas content, the \
`meeting_id` from the seed message, and the matching `kind`.

## Principles

- **Fit the format to the meeting, not the meeting to the format.** A standup \
does not need a decisions section; a brainstorm does not need action items; an \
interview may be almost entirely quotes and themes. Empty headings waste the \
reader's time.
- **Better to leave something out than to invent it.** The record must be \
trustworthy — a future reader should be able to rely on every line.
""",
    "tool_names": ["save_meeting_minutes"],
}
