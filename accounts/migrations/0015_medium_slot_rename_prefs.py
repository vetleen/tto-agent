"""Rename "mid" to "medium" in stored preferences.

Organization.preferences: the slot default ``models.mid`` and the Automatic
category's feature keys ``feature_models.auto_mid`` / ``feature_reasoning.auto_mid``.
UserSettings.preferences: ``models.mid`` (ignored since model choice moved to
the org, renamed for consistency). A key already present under the new name
wins over the old one.
"""

from django.db import migrations

_RENAMES = (("models", "mid", "medium"), ("feature_models", "auto_mid", "auto_medium"),
            ("feature_reasoning", "auto_mid", "auto_medium"))


def remap_preferences(prefs: dict, *, backward: bool = False) -> bool:
    """Rename the keys in *prefs* in place; return whether anything changed."""
    changed = False
    for section, old, new in _RENAMES:
        if backward:
            old, new = new, old
        block = prefs.get(section)
        if isinstance(block, dict) and old in block:
            value = block.pop(old)
            block.setdefault(new, value)
            changed = True
    return changed


def _migrate(apps, backward):
    for model_name in ("Organization", "UserSettings"):
        Model = apps.get_model("accounts", model_name)
        for row in Model.objects.exclude(preferences={}).iterator():
            prefs = row.preferences or {}
            if remap_preferences(prefs, backward=backward):
                row.preferences = prefs
                row.save(update_fields=["preferences"])


def forward(apps, schema_editor):
    _migrate(apps, backward=False)


def backward(apps, schema_editor):
    _migrate(apps, backward=True)


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0014_subagent_tier_rename_prefs"),
    ]

    operations = [
        migrations.RunPython(forward, backward),
    ]
