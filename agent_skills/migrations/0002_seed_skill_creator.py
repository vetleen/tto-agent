"""Data migration to seed the Skill Creator system skill."""

from django.db import migrations

SKILL_CREATOR_SLUG = "skill-creator"
SKILL_CREATOR_NAME = "Skill Creator"
SKILL_CREATOR_DESCRIPTION = (
    "Guide me through designing a new agent skill for Wilfred."
)
SKILL_CREATOR_INSTRUCTIONS = """\
You are helping the user design a new **Agent Skill** for Wilfred.

An Agent Skill is a predefined set of instructions that gets injected into \
Wilfred's system prompt when active. It shapes how Wilfred responds in a \
specific domain or workflow.

## Your process

1. **Understand the goal** — Ask what the skill should do. Probe for:
   - Target audience (who uses it?)
   - Key tasks the skill should handle
   - Tone and style expectations
   - Any domain knowledge or terminology to include
   - Whether it needs access to specific tools (e.g. document search, canvas)

2. **Draft the skill definition** — Produce a structured draft with:
   - **slug**: A short, URL-safe identifier (e.g. `patent-drafter`)
   - **name**: Human-readable display name (e.g. "Patent Drafter")
   - **description**: One sentence shown in the selection menu
   - **instructions**: The full system prompt instructions (this is the core)
   - **tool_names**: List of tool names the skill needs (or empty list)

3. **Review and iterate** — Present the draft to the user. Refine based on \
feedback. Pay special attention to the instructions — they should be clear, \
specific, and actionable.

4. **Present the final definition** — Once the user approves, present the \
final skill definition clearly formatted in chat so an admin can create it \
in the system.

## Guidelines for writing good instructions

- Be specific about the persona and expertise level
- Include step-by-step workflows where appropriate
- Define output format expectations (e.g. "use bullet points", "write in formal tone")
- Mention which tools to use and when
- Include guardrails (what the skill should NOT do)
- Keep instructions focused — one skill per domain/workflow
"""


def create_skill_creator(apps, schema_editor):
    AgentSkill = apps.get_model("agent_skills", "AgentSkill")
    AgentSkill.objects.create(
        slug=SKILL_CREATOR_SLUG,
        name=SKILL_CREATOR_NAME,
        description=SKILL_CREATOR_DESCRIPTION,
        instructions=SKILL_CREATOR_INSTRUCTIONS,
        tool_names=[],
        level="system",
    )


def remove_skill_creator(apps, schema_editor):
    AgentSkill = apps.get_model("agent_skills", "AgentSkill")
    AgentSkill.objects.filter(slug=SKILL_CREATOR_SLUG, level="system").delete()


class Migration(migrations.Migration):
    dependencies = [
        ("agent_skills", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(create_skill_creator, remove_skill_creator),
    ]
