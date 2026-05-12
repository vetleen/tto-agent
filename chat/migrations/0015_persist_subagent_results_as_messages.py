"""Persist sub-agent results as hidden ChatMessages and remove result_delivered.

Sub-agent results were previously delivered via ephemeral context injection,
tracked by a result_delivered flag that was set before the LLM stream started.
If the stream failed, the result was permanently lost.

This migration:
1. Backfills hidden ChatMessage records for existing completed sub-agent runs
   that have results but no corresponding message yet.
2. Removes the now-unused result_delivered field.
"""

from django.db import migrations


def backfill_subagent_result_messages(apps, schema_editor):
    ChatMessage = apps.get_model("chat", "ChatMessage")
    SubAgentRun = apps.get_model("chat", "SubAgentRun")

    completed_runs = SubAgentRun.objects.filter(
        status="completed",
    ).exclude(result="")

    for run in completed_runs.iterator(chunk_size=200):
        already_exists = ChatMessage.objects.filter(
            tool_call_id=str(run.id),
        ).exists()
        if already_exists:
            continue
        ChatMessage.objects.create(
            thread_id=run.thread_id,
            role="tool",
            content=run.result,
            tool_call_id=str(run.id),
            metadata={"source": "subagent", "subagent_run_id": str(run.id)},
            token_count=0,
            is_hidden_from_user=True,
        )


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0014_hide_existing_tool_loop_messages"),
    ]

    operations = [
        migrations.RunPython(
            backfill_subagent_result_messages,
            migrations.RunPython.noop,
        ),
        migrations.RemoveField(
            model_name="subagentrun",
            name="result_delivered",
        ),
    ]
