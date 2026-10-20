"""Rename sub-agent tier keys in Organization.preferences: "mid" -> "medium", "top" -> "high".

Touches three places: ``feature_models.subagent_<tier>`` (per-tier model
override), ``subagents.context_budget.<tier>`` and ``subagents.reasoning.<tier>``.
A key already present under the new name wins over the old one.
"""

from django.db import migrations

_FORWARD = {"mid": "medium", "top": "high"}
_BACKWARD = {"medium": "mid", "high": "top"}


def _rename_keys(d: dict, mapping: dict, prefix: str = "") -> bool:
    changed = False
    for old, new in mapping.items():
        old_key, new_key = prefix + old, prefix + new
        if old_key in d:
            value = d.pop(old_key)
            d.setdefault(new_key, value)
            changed = True
    return changed


def remap_preferences(prefs: dict, mapping: dict) -> bool:
    """Rename the tier keys in *prefs* in place; return whether anything changed."""
    changed = False
    feature_models = prefs.get("feature_models")
    if isinstance(feature_models, dict):
        changed |= _rename_keys(feature_models, mapping, prefix="subagent_")
    subagents = prefs.get("subagents")
    if isinstance(subagents, dict):
        for section in ("context_budget", "reasoning"):
            block = subagents.get(section)
            if isinstance(block, dict):
                changed |= _rename_keys(block, mapping)
    return changed


def _migrate(apps, mapping):
    Organization = apps.get_model("accounts", "Organization")
    for org in Organization.objects.exclude(preferences={}).iterator():
        prefs = org.preferences or {}
        if remap_preferences(prefs, mapping):
            org.preferences = prefs
            org.save(update_fields=["preferences"])


def forward(apps, schema_editor):
    _migrate(apps, _FORWARD)


def backward(apps, schema_editor):
    _migrate(apps, _BACKWARD)


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0013_alter_usersettings_theme"),
    ]

    operations = [
        migrations.RunPython(forward, backward),
    ]
