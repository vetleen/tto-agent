"""Carry org per-tool toggles from Skill Creator to the Skill Management toolkit.

The ``skill_*`` tools moved off the ``skill-creator`` seed into the new
``skill_management_tools`` toolkit (which Skill Creator now requires). An org
admin's per-tool switches live under the carrying skill's slug
(``Organization.preferences["skills"][slug]["tools"][tool]``), so without this
they would silently stop applying. The toggles for the moved tools are COPIED
to the toolkit's entry (the old entry is left as is); an existing toggle under
the toolkit always wins. ``enabled`` is not touched: the toolkit is on
everywhere, like every toolkit. Backward is a no-op.
"""

from django.db import migrations

_SOURCE = "skill-creator"
_TARGET = "skill_management_tools"
_MOVED_TOOLS = (
    "skill_create", "skill_edit", "skill_delete", "skill_field_save",
    "skill_field_load", "skill_resource_list", "skill_resource_save",
    "skill_resource_attach", "skill_resource_update", "skill_resource_delete",
    "skill_tool_list", "skill_tool_inspect",
)


def forward(apps, schema_editor):
    Organization = apps.get_model("accounts", "Organization")
    for org in Organization.objects.exclude(preferences={}).iterator():
        prefs = org.preferences or {}
        skills = prefs.get("skills")
        if not isinstance(skills, dict):
            continue
        source = skills.get(_SOURCE)
        toggles = source.get("tools") if isinstance(source, dict) else None
        if not isinstance(toggles, dict):
            continue
        moved = {t: v for t, v in toggles.items() if t in _MOVED_TOOLS}
        if not moved:
            continue
        target = skills.get(_TARGET)
        if not isinstance(target, dict):
            target = {}
        target_tools = target.get("tools")
        if not isinstance(target_tools, dict):
            target_tools = {}
        merged = {**moved, **target_tools}
        if merged == target_tools:
            continue
        target["tools"] = merged
        skills[_TARGET] = target
        prefs["skills"] = skills
        org.preferences = prefs
        org.save(update_fields=["preferences"])


class Migration(migrations.Migration):

    dependencies = [
        ("agent_skills", "0013_rename_toolkit_slugs"),
        ("accounts", "0014_subagent_tier_rename_prefs"),
    ]

    operations = [
        migrations.RunPython(forward, migrations.RunPython.noop),
    ]
