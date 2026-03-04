"""Data migration: copy UserSettings.theme into preferences["theme"]."""

from django.db import migrations


def copy_theme_to_preferences(apps, schema_editor):
    UserSettings = apps.get_model("accounts", "UserSettings")
    for settings in UserSettings.objects.all():
        prefs = settings.preferences or {}
        if "theme" not in prefs:
            prefs["theme"] = settings.theme
            settings.preferences = prefs
            settings.save(update_fields=["preferences"])


def reverse_copy(apps, schema_editor):
    # No-op: the theme CharField still exists as the canonical source.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_add_preferences_jsonfield"),
    ]

    operations = [
        migrations.RunPython(copy_theme_to_preferences, reverse_copy),
    ]
