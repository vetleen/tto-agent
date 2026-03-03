"""Add description and doc_index fields to ProjectDocument."""

from django.db import migrations, models


def backfill_doc_index(apps, schema_editor):
    """Assign sequential doc_index per project, ordered by uploaded_at."""
    ProjectDocument = apps.get_model("documents", "ProjectDocument")
    # Group by project
    project_ids = (
        ProjectDocument.objects.values_list("project_id", flat=True).distinct()
    )
    for project_id in project_ids:
        docs = ProjectDocument.objects.filter(project_id=project_id).order_by(
            "uploaded_at"
        )
        for idx, doc in enumerate(docs, start=1):
            ProjectDocument.objects.filter(pk=doc.pk).update(doc_index=idx)


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0004_add_chunk_search_vector"),
    ]

    operations = [
        migrations.AddField(
            model_name="projectdocument",
            name="description",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="projectdocument",
            name="doc_index",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.RunPython(backfill_doc_index, migrations.RunPython.noop),
        migrations.AddConstraint(
            model_name="projectdocument",
            constraint=models.UniqueConstraint(
                condition=models.Q(("doc_index__gt", 0)),
                fields=("project", "doc_index"),
                name="documents_unique_doc_index_per_project",
            ),
        ),
    ]
