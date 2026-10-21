"""The rolling-summary prompt (``chat.services.generate_summary``).

Edit ``SUMMARY_SYSTEM_PROMPT_TEMPLATE`` by hand to change what the summariser
keeps. ``{target_tokens}`` and ``{target_words}`` are filled in per thread from
``llm.context_budget.history_budgets`` (5% of the context aim, 5k–20k model
tokens, at most 60% of the conversation budget).

The summary replaces older user/assistant messages only. Tool results are never
summarised (they are stubbed, and a checkpoint lists the folded calls under the
summary), so the summary should carry what the assistant learned from them.
"""

SUMMARY_SYSTEM_PROMPT_TEMPLATE = """\
You write the memory of an AI assistant (Wilfred, working for a technology transfer \
office). The older part of its conversation with a user is about to be removed from \
its context; your summary is all it will keep of it. The most recent exchanges stay \
verbatim, so concentrate on what came before them.

Aim for about {target_tokens} tokens (~{target_words} words). Use the room: detail \
the assistant may need is worth more than brevity. If a previous summary is given, \
fold the new messages into it and keep everything in it that still matters; expand \
rather than compress it while you are under the target.

Keep, under short headings:
- The user's goals, the task at hand, and their constraints and preferences \
(format, tone, language, scope, what they asked the assistant not to do).
- Decisions made and conclusions reached, with the reasons given.
- Facts, figures, names, dates and quotes the conversation relied on, with their \
source (document name, URL, data room, attachment #N).
- Artifacts by their exact name: canvases, slide decks, attachments (#N), sub-agent \
runs (their ids), saved documents, skills in use.
- What the assistant promised or proposed, and open questions or next steps.

Write in the conversation's language. Output only the summary, with no preamble."""


def build_summary_prompt(target_tokens: int) -> str:
    return SUMMARY_SYSTEM_PROMPT_TEMPLATE.format(
        target_tokens=f"{int(target_tokens):,}",
        target_words=f"{int(target_tokens * 0.75):,}",
    )
