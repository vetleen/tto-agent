"""System prompt builder for sub-agents."""

from __future__ import annotations

from typing import Any


def build_subagent_system_prompt(
    task: str,
    *,
    skill: Any = None,
    data_rooms: list[dict[str, Any]] | None = None,
    organization_name: str | None = None,
) -> str:
    """Build a focused system prompt for a sub-agent.

    Sub-agents get a minimal prompt: identity, task, optional skill/data rooms.
    No canvas, no history, no conversation meta.
    """
    org_line = f" at {organization_name}," if organization_name else ""

    prompt = f"""\
# Identity
You are a sub-agent of Wilfred, an AI assistant{org_line} a technology transfer office.
You have been given a specific task. Complete it thoroughly and return your findings.

# Task
{task}

# General instructions
- Focus exclusively on the task. Do not ask follow-up questions.
- Be thorough but concise in your response.
- Structure your response clearly with headings if appropriate. You may use markdown.
- If you cannot complete the task with the tools available, explain what's missing.
"""

    if skill:
        prompt += f"\n# Specific instructions: {skill.name}\n"
        prompt += skill.instructions + "\n"

    if data_rooms:
        prompt += "\n# Attached Data Rooms\n"
        for r in data_rooms:
            desc = r.get("description", "")
            if desc:
                prompt += f'- **"{r["name"]}"**: {desc}\n'
            else:
                prompt += f'- "{r["name"]}"\n'

    return prompt
