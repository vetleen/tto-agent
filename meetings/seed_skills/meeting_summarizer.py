"""Meeting Summarizer seed skill.

Intentionally short for v1 — the user (Vetle) plans to refine the body
later. The point of seeding it now is to have something the
"Create meeting minutes with Wilfred" flow can attach to a chat thread.
"""

MEETING_SUMMARIZER = {
    "slug": "meeting-summarizer",
    "name": "Meeting Summarizer",
    "description": (
        "Turns a meeting transcript into well-structured minutes. Use when "
        "the user opens a meeting in chat to draft minutes, asks to summarize "
        "a meeting transcript, produce action items, or extract decisions "
        "from a recorded conversation."
    ),
    "instructions": (
        "# Meeting Summarizer\n"
        "Produce concise, faithful meeting minutes from a transcript.\n"
        "\n"
        "1. Read the attached transcript end-to-end before drafting.\n"
        "2. Draft in a canvas titled `Meeting minutes — <meeting name>`.\n"
        "3. Use this structure: **Date/duration**, **Participants**, "
        "**Agenda**, **Discussion** (bulleted, grouped by topic), "
        "**Decisions**, **Action items** (owner — task — due date if stated).\n"
        "4. Quote verbatim only when wording matters (numbers, commitments, "
        "names of deliverables). Otherwise paraphrase tightly.\n"
        "5. Never invent attendees, decisions, or action items not present "
        "in the transcript.\n"
        "6. When the user is satisfied, call `save_meeting_minutes` with the "
        "canvas content (kind defaults to `minutes`)."
    ),
    "tool_names": ["save_meeting_minutes"],
}
