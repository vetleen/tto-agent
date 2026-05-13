"""Fix sub-agent result messages: change from role="tool" to role="user".

The original format (role="tool" with tool_call_id=SubAgentRun UUID) created
orphan tool results that OpenAI rejects — no assistant message has the
SubAgentRun UUID in its tool_calls array.  Changing to role="user" eliminates
the orphan and lets the message participate in summarization normally.
"""

from django.db import migrations


def fix_subagent_tool_messages(apps, schema_editor):
    ChatMessage = apps.get_model("chat", "ChatMessage")

    orphans = ChatMessage.objects.filter(
        role="tool",
        metadata__source="subagent",
    )
    for msg in orphans.iterator(chunk_size=200):
        subagent_run_id = (msg.metadata or {}).get("subagent_run_id", "")
        short_id = subagent_run_id[:8] if subagent_run_id else "unknown"
        msg.role = "user"
        msg.content = f"[Sub-agent result: {short_id}]\n{msg.content}"
        msg.tool_call_id = None
        msg.save(update_fields=["role", "content", "tool_call_id"])


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0015_persist_subagent_results_as_messages"),
    ]

    operations = [
        migrations.RunPython(
            fix_subagent_tool_messages,
            migrations.RunPython.noop,
        ),
    ]
