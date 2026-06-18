# Step 2 of 3 for multi-skill threads: copy each thread's single attached
# skill (the soon-to-be-removed `skill` FK) into the new ChatThreadSkill
# table. Must run AFTER 0024 (table exists) and BEFORE 0026 (FK removed).
#
# We deliberately do NOT filter on is_active: this preserves the pre-existing
# stored relationship verbatim. Runtime access gates re-check is_active and
# per-user/org access when loading skills, so an inactive carry-over row is
# inert rather than leaked.
from django.db import migrations


def copy_skill_to_m2m(apps, schema_editor):
    ChatThread = apps.get_model('chat', 'ChatThread')
    ChatThreadSkill = apps.get_model('chat', 'ChatThreadSkill')
    rows = [
        ChatThreadSkill(thread_id=thread_id, skill_id=skill_id)
        for thread_id, skill_id in ChatThread.objects.filter(
            skill_id__isnull=False
        ).values_list('id', 'skill_id')
    ]
    if rows:
        ChatThreadSkill.objects.bulk_create(rows, ignore_conflicts=True)


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0024_chatthreadskill'),
    ]

    # Reverse is a no-op: collapsing N attachments back to one FK can't be
    # done honestly, and forward-only is fine for this one-way model change.
    operations = [
        migrations.RunPython(copy_skill_to_m2m, migrations.RunPython.noop),
    ]
