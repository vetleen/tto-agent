"""Assistant Loops (seed skill).

A main-agent skill that unlocks the loop-management tools (``chat_loop_*``,
``section="skills"``, ``audience="main"``). Activating it lets the assistant set
up, list, edit, and stop recurring/scheduled loops. Sub-agents do not get these
tools.
"""

ASSISTANT_LOOP_TOOLS = {
    "slug": "assistant_loop_tools",
    "name": "Assistant Loops",
    "emoji": "🔁",
    "description": (
        "List, set up, edit or stop *Loops* — recurring, scheduled tasks that re-run a prompt "
        "on a cadence in their own thread. Activate when the user wants something done "
        "repeatedly or on a schedule, or if they ask you to create or manage a loop."
    ),
    "instructions": """\
# Assistant Loops
A Loop runs the same prompt on a cadence in its OWN new thread — not this \
conversation — and does NOT inherit this chat's data rooms, skills, or model, so \
pass those explicitly as the user intends.
""",
    "tool_names": [
        "chat_loop_create",
        "chat_loop_list",
        "chat_loop_edit",
        "chat_loop_stop",
    ],
}
