"""Flatten parent/child chunks to a single flat chunk model."""

from django.db import migrations, models


def delete_child_chunks(apps, schema_editor):
    """Delete all child chunks before removing the fields."""
    DataRoomDocumentChunk = apps.get_model("documents", "DataRoomDocumentChunk")
    DataRoomDocumentChunk.objects.filter(is_child=True).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0002_parent_child_chunking"),
    ]

    operations = [
        # 1. Delete child chunk rows
        migrations.RunPython(delete_child_chunks, migrations.RunPython.noop),
        # 2. Drop the constraint that includes is_child
        migrations.RemoveConstraint(
            model_name="dataroomdocumentchunk",
            name="documents_chunk_unique_per_document",
        ),
        # 3. Remove parent FK
        migrations.RemoveField(
            model_name="dataroomdocumentchunk",
            name="parent",
        ),
        # 4. Remove is_child field
        migrations.RemoveField(
            model_name="dataroomdocumentchunk",
            name="is_child",
        ),
        # 5. Add new constraint on (document, chunk_index)
        migrations.AddConstraint(
            model_name="dataroomdocumentchunk",
            constraint=models.UniqueConstraint(
                fields=("document", "chunk_index"),
                name="documents_chunk_unique_per_document",
            ),
        ),
    ]
