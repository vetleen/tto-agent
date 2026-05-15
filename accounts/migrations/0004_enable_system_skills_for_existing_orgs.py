"""Data migration: explicitly enable existing system skills for all orgs.

System skills now default to *disabled* in org settings.  To avoid breaking
existing organisations that have been using system skills without an explicit
preference, this migration writes ``{"enabled": True}`` for every active
system skill in every org that has not already set an explicit value.
"""

from django.db import migrations


def forwards(apps, schema_editor):
    Organization = apps.get_model("accounts", "Organization")
    AgentSkill = apps.get_model("agent_skills", "AgentSkill")

    system_slugs = list(
        AgentSkill.objects.filter(level="system", is_active=True)
        .values_list("slug", flat=True)
    )
    if not system_slugs:
        return

    for org in Organization.objects.all().iterator():
        prefs = org.preferences or {}
        skills = prefs.get("skills", {})
        changed = False
        for slug in system_slugs:
            sp = skills.get(slug, {})
            if not isinstance(sp, dict):
                sp = {}
            if "enabled" not in sp:
                sp["enabled"] = True
                skills[slug] = sp
                changed = True
        if changed:
            prefs["skills"] = skills
            org.preferences = prefs
            org.save(update_fields=["preferences"])


def backwards(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0003_user_org_descriptions"),
        ("agent_skills", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(forwards, backwards, elidable=True),
    ]
