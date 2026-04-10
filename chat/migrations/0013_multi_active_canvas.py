"""Add is_active + last_activated_at to ChatCanvas for multi-active canvas support."""

from django.db import migrations, models


def populate_active_flags(apps, schema_editor):
    """Set is_active=True on each thread's current active_canvas."""
    ChatThread = apps.get_model("chat", "ChatThread")
    ChatCanvas = apps.get_model("chat", "ChatCanvas")
    for thread in ChatThread.objects.filter(active_canvas__isnull=False).select_related("active_canvas"):
        ChatCanvas.objects.filter(pk=thread.active_canvas_id).update(
            is_active=True,
            last_activated_at=thread.active_canvas.updated_at,
        )


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0012_chatmessage_is_hidden_from_user_chatthread_metadata"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatcanvas",
            name="is_active",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="chatcanvas",
            name="last_activated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddIndex(
            model_name="chatcanvas",
            index=models.Index(
                fields=["thread", "is_active", "-last_activated_at"],
                name="chat_chatca_thread__24ff93_idx",
            ),
        ),
        migrations.RunPython(populate_active_flags, migrations.RunPython.noop),
    ]
