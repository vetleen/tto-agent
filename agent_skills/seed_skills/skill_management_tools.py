"""Skill Management Tools (seed toolkit).

A main-agent toolkit that unlocks the skill-authoring tools (``skill_create``,
``skill_edit``, the ``skill_resource_*`` tools, …). The instructions are only
the briefest work process: how to use each tool lives in the tool descriptions,
and how to write a good skill lives in the Skill Creator playbook, which
requires this toolkit.
"""

SKILL_MANAGEMENT_TOOLS = {
    "slug": "skill_management_tools",
    "name": "Skill Management Tools",
    "emoji": "🧰",
    "description": (
        "Create, edit and delete skills and their resources. Activate when the user "
        "wants to make a new skill or change an existing one."
    ),
    "instructions": """\
# Skill Management Tools

Work process:
1. Create the skill (`skill_create`), or find the one to change.
2. Draft the description and instructions in canvases, then save them to the skill.
3. Save or attach any resources the skill needs.
4. Give it its tools: on a main skill, require whole toolkits (`requires`); list \
individual tools (`tool_names`) only on a sub-agent skill or for a tool no toolkit covers.
5. Tell the user what you saved and where to find it.
""",
    "tool_names": [
        "skill_create",
        "skill_edit",
        "skill_delete",
        "skill_field_save",
        "skill_field_load",
        "skill_resource_list",
        "skill_resource_save",
        "skill_resource_attach",
        "skill_resource_update",
        "skill_resource_delete",
        "skill_tool_list",
        "skill_tool_inspect",
    ],
    "kind": "toolkit",
}
