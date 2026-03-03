"""Add unique constraint on (project, doc_index).

Split from 0005 because Postgres cannot create an index on a table with
pending trigger events from the data migration in the same transaction.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0005_add_document_description_and_index"),
    ]

    operations = [
        migrations.AddConstraint(
            model_name="projectdocument",
            constraint=models.UniqueConstraint(
                condition=models.Q(("doc_index__gt", 0)),
                fields=("project", "doc_index"),
                name="documents_unique_doc_index_per_project",
            ),
        ),
    ]
