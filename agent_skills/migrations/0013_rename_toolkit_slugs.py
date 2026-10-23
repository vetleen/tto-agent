"""Rename the toolkit seed slugs so every toolkit ends in ``_tools``.

``canvas_collaborator -> canvas_tools``, ``image_generator -> image_generation_tools``,
``slide_deck_collaborator -> slide_deck_tools``. The system rows are renamed IN
PLACE (ids kept, so ``ChatThreadSkill`` rows and user forks' ``parent`` links
survive) and marked ``kind="toolkit"``; the slug-keyed preference stores are
re-keyed:

- ``Organization.preferences["skills"][slug]`` — moved to the new key. When the
  org also owns an org-level skill under the old slug the entry is COPIED
  instead, so that skill keeps its toggles.
- ``UserSettings.preferences["skills"][slug]`` — an entry selecting the renamed
  system row moves; a slug-wide disable (``selected_skill_id: None``) is copied
  (the old key still disables a personal fork of the old slug); an entry
  selecting any other skill (a fork) is left alone.

An existing entry under the new key always wins. If a live system row already
holds the new slug (partial deploy) the old row is left untouched.
"""

import copy

from django.db import migrations
from django.db.models import Q

_FORWARD = {
    "canvas_collaborator": "canvas_tools",
    "image_generator": "image_generation_tools",
    "slide_deck_collaborator": "slide_deck_tools",
}
_BACKWARD = {new: old for old, new in _FORWARD.items()}

_TOOLKIT_SLUGS = (
    "assistant_loop_tools", "canvas_tools", "data_room_tools",
    "image_generation_tools", "web_research_tools", "slide_deck_tools",
)


def _rename_system_rows(AgentSkill, mapping) -> dict[str, str]:
    """Rename live system rows; return ``{old_slug: row id}`` for those renamed."""
    live = Q(level="system", deleted_at__isnull=True)
    renamed: dict[str, str] = {}
    for old, new in mapping.items():
        row = AgentSkill.objects.filter(live, slug=old).first()
        if row is None:
            continue
        if AgentSkill.objects.filter(live, slug=new).exists():
            continue  # target already seeded under the new slug; leave both
        row.slug = new
        row.save(update_fields=["slug"])
        renamed[old] = str(row.id)
    return renamed


def _rekey_org_prefs(AgentSkill, Organization, mapping) -> None:
    for org in Organization.objects.exclude(preferences={}).iterator():
        prefs = org.preferences or {}
        skills = prefs.get("skills")
        if not isinstance(skills, dict):
            continue
        changed = False
        for old, new in mapping.items():
            if old not in skills:
                continue
            owns_old = AgentSkill.objects.filter(
                level="org", organization_id=org.id, slug=old, deleted_at__isnull=True,
            ).exists()
            if owns_old:
                if new not in skills:
                    skills[new] = copy.deepcopy(skills[old])
                    changed = True
                continue
            entry = skills.pop(old)
            skills.setdefault(new, entry)
            changed = True
        if changed:
            prefs["skills"] = skills
            org.preferences = prefs
            org.save(update_fields=["preferences"])


def _rekey_user_prefs(UserSettings, mapping, renamed: dict[str, str]) -> None:
    for old, new in mapping.items():
        system_id = renamed.get(old)
        qs = UserSettings.objects.filter(preferences__skills__has_key=old)
        for us in qs.iterator():
            prefs = us.preferences or {}
            skills = prefs.get("skills")
            if not isinstance(skills, dict):
                continue
            entry = skills.get(old)
            if not isinstance(entry, dict):
                continue
            selected = entry.get("selected_skill_id")
            if selected is None:
                if new in skills:
                    continue
                skills[new] = dict(entry)
            elif system_id is not None and str(selected) == system_id:
                skills.pop(old)
                skills.setdefault(new, entry)
            else:
                continue  # selects a fork under the old slug — leave it
            prefs["skills"] = skills
            us.preferences = prefs
            us.save(update_fields=["preferences"])


def _migrate(apps, mapping, *, mark_toolkits: bool) -> None:
    AgentSkill = apps.get_model("agent_skills", "AgentSkill")
    Organization = apps.get_model("accounts", "Organization")
    UserSettings = apps.get_model("accounts", "UserSettings")

    renamed = _rename_system_rows(AgentSkill, mapping)
    if mark_toolkits:
        AgentSkill.objects.filter(
            level="system", deleted_at__isnull=True, slug__in=_TOOLKIT_SLUGS,
        ).update(kind="toolkit")
    _rekey_org_prefs(AgentSkill, Organization, mapping)
    _rekey_user_prefs(UserSettings, mapping, renamed)


def forward(apps, schema_editor):
    _migrate(apps, _FORWARD, mark_toolkits=True)


def backward(apps, schema_editor):
    _migrate(apps, _BACKWARD, mark_toolkits=False)


class Migration(migrations.Migration):

    dependencies = [
        ("agent_skills", "0012_agentskill_kind_requires"),
        ("accounts", "0014_subagent_tier_rename_prefs"),
    ]

    operations = [
        migrations.RunPython(forward, backward),
    ]
