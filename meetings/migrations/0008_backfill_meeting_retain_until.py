from __future__ import annotations

from datetime import timedelta

from django.db import migrations
from django.db.models import F


def backfill(apps, schema_editor):
    Meeting = apps.get_model("meetings", "Meeting")
    Meeting.objects.filter(retain_until__isnull=True).update(
        retain_until=F("updated_at") + timedelta(days=90),
    )


class Migration(migrations.Migration):

    dependencies = [
        ("meetings", "0007_meeting_retain_until"),
    ]

    operations = [
        migrations.RunPython(backfill, reverse_code=migrations.RunPython.noop),
    ]
