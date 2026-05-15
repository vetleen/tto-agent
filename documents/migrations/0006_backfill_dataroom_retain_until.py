from __future__ import annotations

from datetime import timedelta

from django.db import migrations
from django.db.models import F


def backfill(apps, schema_editor):
    DataRoom = apps.get_model("documents", "DataRoom")
    DataRoom.objects.filter(retain_until__isnull=True).update(
        retain_until=F("updated_at") + timedelta(days=365),
    )


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0005_dataroom_retain_until"),
    ]

    operations = [
        migrations.RunPython(backfill, reverse_code=migrations.RunPython.noop),
    ]
