from agent_skills.services import slug_is_auto
from django.db import migrations, models


def backfill_slug_customized(apps, schema_editor):
    """Freeze the slug of user skills whose slug was clearly hand-picked.

    New behavior auto-follows ``name`` on rename while ``slug_customized`` is
    False. Existing user skills default to False, so a genuinely custom slug
    would start drifting on the next rename — undesirable. Mark those custom
    rows as customized. Rows whose slug is still an auto-derived / placeholder
    form are left auto so an existing ``untitled-skill`` self-corrects on its
    first real rename. Org/system skills are out of scope (their slug UI is
    read-only and their prefs are keyed by slug without a rename migration).
    """
    AgentSkill = apps.get_model("agent_skills", "AgentSkill")
    for row in AgentSkill.objects.filter(level="user").iterator():
        if not slug_is_auto(row.name, row.slug):
            row.slug_customized = True
            row.save(update_fields=["slug_customized"])


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("agent_skills", "0009_agentskill_soft_delete"),
    ]

    operations = [
        migrations.AddField(
            model_name="agentskill",
            name="slug_customized",
            field=models.BooleanField(default=False),
        ),
        migrations.RunPython(backfill_slug_customized, noop_reverse),
    ]
