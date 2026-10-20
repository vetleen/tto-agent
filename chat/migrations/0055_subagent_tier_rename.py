"""Rename sub-agent model tiers: "mid" -> "medium", "top" -> "high" (new "low" needs no data)."""

from django.db import migrations, models

_FORWARD = {"mid": "medium", "top": "high"}
_BACKWARD = {"medium": "mid", "high": "top", "low": "mid"}


def _remap(apps, mapping):
    SubAgentRun = apps.get_model("chat", "SubAgentRun")
    for old, new in mapping.items():
        SubAgentRun.objects.filter(model_tier=old).update(model_tier=new)


def forward(apps, schema_editor):
    _remap(apps, _FORWARD)


def backward(apps, schema_editor):
    _remap(apps, _BACKWARD)


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0054_canvasimportjob"),
    ]

    operations = [
        migrations.AlterField(
            model_name="subagentrun",
            name="model_tier",
            field=models.CharField(default="medium", max_length=10),
        ),
        migrations.RunPython(forward, backward),
    ]
