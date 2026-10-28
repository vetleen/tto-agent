"""Rename the Automatic category "mid" to "medium" in ChatThread.auto_route."""

from django.db import migrations


def _rename(apps, old, new):
    ChatThread = apps.get_model("chat", "ChatThread")
    for thread in ChatThread.objects.filter(auto_route__category=old).iterator():
        thread.auto_route = {**thread.auto_route, "category": new}
        ChatThread.objects.filter(pk=thread.pk).update(auto_route=thread.auto_route)


def forward(apps, schema_editor):
    _rename(apps, "mid", "medium")


def backward(apps, schema_editor):
    _rename(apps, "medium", "mid")


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0061_chatthread_auto_route"),
    ]

    operations = [
        migrations.RunPython(forward, backward),
    ]
