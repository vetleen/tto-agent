# Generated manually for hybrid search support

import django.contrib.postgres.indexes
import django.contrib.postgres.search
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0003_remove_projectdocument_created_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="projectdocumentchunk",
            name="search_vector",
            field=django.contrib.postgres.search.SearchVectorField(null=True),
        ),
        migrations.AddIndex(
            model_name="projectdocumentchunk",
            index=django.contrib.postgres.indexes.GinIndex(
                fields=["search_vector"], name="chunk_search_vector_gin"
            ),
        ),
    ]
