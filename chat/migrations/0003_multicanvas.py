"""Convert OneToOneField → ForeignKey for multi-canvas support.

Adds active_canvas FK on ChatThread, unique constraint on (thread, title),
and a data migration to set active_canvas for existing canvases.
"""

import django.db.models.deletion
from django.db import migrations, models


def set_active_canvas_forward(apps, schema_editor):
    """For every existing ChatCanvas, set its thread's active_canvas."""
    ChatCanvas = apps.get_model("chat", "ChatCanvas")
    ChatThread = apps.get_model("chat", "ChatThread")
    for canvas in ChatCanvas.objects.all():
        ChatThread.objects.filter(pk=canvas.thread_id).update(active_canvas=canvas)


def set_active_canvas_reverse(apps, schema_editor):
    """Reverse: clear active_canvas pointers."""
    ChatThread = apps.get_model("chat", "ChatThread")
    ChatThread.objects.all().update(active_canvas=None)


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0002_subagentrun"),
    ]

    operations = [
        # 1. Change OneToOneField → ForeignKey (removes unique constraint)
        migrations.AlterField(
            model_name="chatcanvas",
            name="thread",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="canvases",
                to="chat.chatthread",
            ),
        ),
        # 2. Add unique constraint on (thread, title)
        migrations.AddConstraint(
            model_name="chatcanvas",
            constraint=models.UniqueConstraint(
                fields=["thread", "title"],
                name="unique_canvas_title_per_thread",
            ),
        ),
        # 3. Add index on (thread, created_at)
        migrations.AddIndex(
            model_name="chatcanvas",
            index=models.Index(
                fields=["thread", "created_at"],
                name="chat_chatcanvas_thread_created_idx",
            ),
        ),
        # 4. Add ordering Meta
        migrations.AlterModelOptions(
            name="chatcanvas",
            options={"ordering": ["created_at"]},
        ),
        # 5. Add active_canvas FK on ChatThread
        migrations.AddField(
            model_name="chatthread",
            name="active_canvas",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="+",
                to="chat.chatcanvas",
            ),
        ),
        # 6. Data migration: set active_canvas for existing canvases
        migrations.RunPython(
            set_active_canvas_forward,
            set_active_canvas_reverse,
        ),
    ]
