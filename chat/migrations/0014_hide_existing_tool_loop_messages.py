"""Backfill is_hidden_from_user=True on intermediate tool-loop messages.

Before this fix, _persist_tool_loop_messages persisted assistant messages
with empty content + tool_calls in metadata, and tool-result messages,
without marking them hidden. The chat_home view loaded them, and the
template rendered each assistant message as a bubble with just a header
("Wilfred HH:MM") and an empty content div.

This migration hides existing rows that match that pattern so prior
threads stop showing empty bubbles after deploy. The LLM context query
in consumers.py doesn't filter on is_hidden_from_user, so history sent
to the model is unchanged.
"""

from django.db import migrations


def hide_intermediate_tool_messages(apps, schema_editor):
    ChatMessage = apps.get_model("chat", "ChatMessage")

    # Assistant messages with empty content and tool_calls in metadata
    ChatMessage.objects.filter(
        role="assistant",
        content="",
        is_hidden_from_user=False,
        metadata__tool_calls__isnull=False,
    ).update(is_hidden_from_user=True)

    # Tool result messages — never user-visible (template has no tool branch)
    ChatMessage.objects.filter(
        role="tool",
        is_hidden_from_user=False,
    ).update(is_hidden_from_user=True)


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0013_multi_active_canvas"),
    ]

    operations = [
        migrations.RunPython(
            hide_intermediate_tool_messages,
            migrations.RunPython.noop,
        ),
    ]
