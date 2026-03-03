"""System prompt builders for the chat app."""

from __future__ import annotations

from typing import Any


def build_system_prompt(project, *, history_meta: dict[str, Any] | None = None) -> str:
    """Build the system prompt for a project chat session.

    *history_meta*, when provided, should contain keys like
    ``total_messages``, ``included_messages``, and ``has_summary``.
    """
    prompt = (
        "You are a helpful assistant for the project "
        f'"{project.name}". '
        "You have access to a search_documents tool that can search "
        "the project's uploaded documents. Use it when the user asks "
        "about document contents or needs specific information from their files. "
        "Answer concisely and accurately."
    )

    if history_meta:
        total = history_meta.get("total_messages", 0)
        included = history_meta.get("included_messages", 0)
        has_summary = history_meta.get("has_summary", False)

        if total > included:
            prompt += (
                f"\n\nThis conversation has {total} messages total. "
                f"The {included} most recent are shown."
            )
            if has_summary:
                prompt += (
                    " A summary of earlier messages is provided."
                )

    return prompt
