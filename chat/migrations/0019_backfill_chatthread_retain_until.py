from __future__ import annotations

from datetime import timedelta

from django.db import migrations
from django.db.models import F


def backfill(apps, schema_editor):
    ChatThread = apps.get_model("chat", "ChatThread")
    ChatThread.objects.filter(retain_until__isnull=True).update(
        retain_until=F("updated_at") + timedelta(days=365),
    )


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0018_chatthread_retain_until"),
    ]

    operations = [
        migrations.RunPython(backfill, reverse_code=migrations.RunPython.noop),
    ]
