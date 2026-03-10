"""Flatten parent/child chunks: step 1 — delete child chunk rows."""

from django.db import migrations


def delete_child_chunks(apps, schema_editor):
    """Delete all child chunks before removing the fields."""
    DataRoomDocumentChunk = apps.get_model("documents", "DataRoomDocumentChunk")
    DataRoomDocumentChunk.objects.filter(is_child=True).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0002_parent_child_chunking"),
    ]

    operations = [
        migrations.RunPython(delete_child_chunks, migrations.RunPython.noop),
    ]
