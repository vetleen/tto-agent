"""Data migration to backfill token_count on existing ChatMessage rows."""

from django.db import migrations


def backfill_token_count(apps, schema_editor):
    ChatMessage = apps.get_model("chat", "ChatMessage")
    from core.tokens import count_tokens

    for msg in ChatMessage.objects.filter(token_count=0).iterator(chunk_size=500):
        msg.token_count = count_tokens(msg.content)
        msg.save(update_fields=["token_count"])


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0002_add_token_count_and_summary_fields"),
    ]

    operations = [
        migrations.RunPython(backfill_token_count, migrations.RunPython.noop),
    ]
