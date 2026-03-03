"""System prompt builders for the chat app."""

from __future__ import annotations

from typing import Any


def build_system_prompt(
    project,
    *,
    history_meta: dict[str, Any] | None = None,
    doc_context: dict[str, Any] | None = None,
) -> str:
    """Build the system prompt for a project chat session.

    *history_meta*, when provided, should contain keys like
    ``total_messages``, ``included_messages``, and ``has_summary``.

    *doc_context*, when provided, should contain:
    - ``total_doc_count``: int
    - ``documents``: list of dicts with ``doc_index``, ``filename``,
      ``description``, ``token_count``
    """
    prompt = f'''\
# Identity
You are Wilfred, a helpful assistant for the project "{project.name}".

# Instructions
- Answer the user's queries concisely and accurately.
- Plan out your responses for max clarity, using the MECE framework (each part of the answer should be Mutually Exclusive from other parts, but Collectively Exhaustive of the issue).
'''

    # -- Document context section --
    if doc_context and doc_context.get("total_doc_count", 0) > 0:
        total = doc_context["total_doc_count"]
        docs = doc_context.get("documents", [])

        prompt += f"\n# Project Documents\nThis project has {total} document{'s' if total != 1 else ''}."

        if docs:
            prompt += " Based on the user's latest message, these documents may be relevant:\n\n"
            for doc in docs:
                idx = doc["doc_index"]
                fname = doc["filename"]
                tokens = doc.get("token_count") or 0
                desc = doc.get("description", "")
                token_note = f" (~{tokens:,} tokens)" if tokens else ""
                line = f'{idx}. [{idx}] "{fname}"{token_note}'
                if desc:
                    line += f" — {desc}"
                prompt += line + "\n"

            shown = len(docs)
            remaining = total - shown
            if remaining > 0:
                prompt += f"\nThere {'is' if remaining == 1 else 'are'} {remaining} other document{'s' if remaining != 1 else ''} not shown here.\n"
        prompt += "\n"
    else:
        prompt += "\nThis project has no documents uploaded yet.\n\n"

    # -- Tools section --
    prompt += """\
# Tools
- **search_documents(query, k)**: Search the project's documents for relevant passages. You can use the user's exact wording or sharpen the search query for better results.
- **read_document(doc_indices)**: Read the full content of specific documents by their index number (e.g. [1, 3]).

When the user asks about project-specific topics, use search_documents to find relevant passages. Use read_document when you need the full context of a specific document.
"""

    if history_meta:
        total = history_meta.get("total_messages", 0)
        included = history_meta.get("included_messages", 0)
        has_summary = history_meta.get("has_summary", False)

        if total > included:
            prompt += (
                f"\nThis conversation has {total} messages total. "
                f"The {included} most recent are shown."
            )
            if has_summary:
                prompt += (
                    " A summary of earlier messages is provided."
                )

    return prompt
