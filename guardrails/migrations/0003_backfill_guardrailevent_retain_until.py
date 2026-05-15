from __future__ import annotations

from datetime import timedelta

from django.db import migrations
from django.db.models import F


def backfill(apps, schema_editor):
    GuardrailEvent = apps.get_model("guardrails", "GuardrailEvent")
    GuardrailEvent.objects.filter(retain_until__isnull=True).update(
        retain_until=F("created_at") + timedelta(days=180),
    )


class Migration(migrations.Migration):

    dependencies = [
        ("guardrails", "0002_guardrailevent_retain_until"),
    ]

    operations = [
        migrations.RunPython(backfill, reverse_code=migrations.RunPython.noop),
    ]
