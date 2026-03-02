"""System prompt builders for the chat app."""

from __future__ import annotations


def build_system_prompt(project) -> str:
    """Build the system prompt for a project chat session."""
    return (
        "You are a helpful assistant for the project "
        f'"{project.name}". '
        "You have access to a search_documents tool that can search "
        "the project's uploaded documents. Use it when the user asks "
        "about document contents or needs specific information from their files. "
        "Answer concisely and accurately."
    )
