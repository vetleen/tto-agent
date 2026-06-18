"""Contract step: make chunk/tag version FK non-null and drop the document FK.

Runs after the 0013 backfill, so every chunk and tag already has a version.
Order matters: drop the document-based constraint/index before removing the
document column, then add the version-based constraint/index.
"""
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0013_backfill_document_versions"),
    ]

    operations = [
        # --- chunk: drop document-based constraint + index ---
        migrations.RemoveConstraint(
            model_name="dataroomdocumentchunk",
            name="documents_chunk_unique_per_document",
        ),
        migrations.RemoveIndex(
            model_name="dataroomdocumentchunk",
            name="documents_d_documen_a04548_idx",
        ),
        # --- chunk: version becomes the owner ---
        migrations.AlterField(
            model_name="dataroomdocumentchunk",
            name="version",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="chunks",
                to="documents.dataroomdocumentversion",
            ),
        ),
        migrations.RemoveField(
            model_name="dataroomdocumentchunk",
            name="document",
        ),
        migrations.AddConstraint(
            model_name="dataroomdocumentchunk",
            constraint=models.UniqueConstraint(
                fields=("version", "chunk_index"),
                name="documents_chunk_unique_per_version",
            ),
        ),
        migrations.AddIndex(
            model_name="dataroomdocumentchunk",
            index=models.Index(
                fields=["version", "chunk_index"],
                name="documents_chunk_version_idx",
            ),
        ),
        migrations.AlterModelOptions(
            name="dataroomdocumentchunk",
            options={"ordering": ["version", "chunk_index"]},
        ),
        # --- tag: drop document-based constraint ---
        migrations.RemoveConstraint(
            model_name="dataroomdocumenttag",
            name="documents_tag_unique_per_document",
        ),
        migrations.AlterField(
            model_name="dataroomdocumenttag",
            name="version",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="tags",
                to="documents.dataroomdocumentversion",
            ),
        ),
        migrations.RemoveField(
            model_name="dataroomdocumenttag",
            name="document",
        ),
        migrations.AddConstraint(
            model_name="dataroomdocumenttag",
            constraint=models.UniqueConstraint(
                fields=("version", "key"),
                name="documents_tag_unique_per_version",
            ),
        ),
    ]
