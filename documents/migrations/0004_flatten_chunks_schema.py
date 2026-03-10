"""Flatten parent/child chunks: step 2 — remove parent FK, is_child field, update constraint."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0003_flatten_chunks"),
    ]

    operations = [
        # 1. Drop the constraint that includes is_child
        migrations.RemoveConstraint(
            model_name="dataroomdocumentchunk",
            name="documents_chunk_unique_per_document",
        ),
        # 2. Remove parent FK
        migrations.RemoveField(
            model_name="dataroomdocumentchunk",
            name="parent",
        ),
        # 3. Remove is_child field
        migrations.RemoveField(
            model_name="dataroomdocumentchunk",
            name="is_child",
        ),
        # 4. Add new constraint on (document, chunk_index)
        migrations.AddConstraint(
            model_name="dataroomdocumentchunk",
            constraint=models.UniqueConstraint(
                fields=("document", "chunk_index"),
                name="documents_chunk_unique_per_document",
            ),
        ),
    ]
