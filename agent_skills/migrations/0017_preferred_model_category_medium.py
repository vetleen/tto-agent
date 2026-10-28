"""Rename the Automatic category "mid" to "medium" in AgentSkill.preferred_model_category."""

from django.db import migrations


def forward(apps, schema_editor):
    AgentSkill = apps.get_model("agent_skills", "AgentSkill")
    AgentSkill.objects.filter(preferred_model_category="mid").update(preferred_model_category="medium")


def backward(apps, schema_editor):
    AgentSkill = apps.get_model("agent_skills", "AgentSkill")
    AgentSkill.objects.filter(preferred_model_category="medium").update(preferred_model_category="mid")


class Migration(migrations.Migration):

    dependencies = [
        ("agent_skills", "0016_agentskill_preferred_model"),
    ]

    operations = [
        migrations.RunPython(forward, backward),
    ]
