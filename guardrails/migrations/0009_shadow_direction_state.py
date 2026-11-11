from django.db import migrations, models


def _direction_fields(model_name):
    return [
        migrations.AddField(
            model_name=model_name, name="direction",
            field=models.CharField(default="decisions_shadow", max_length=16),
        ),
        migrations.AddField(
            model_name=model_name, name="state",
            field=models.CharField(default="done", max_length=12),
        ),
        migrations.AddField(
            model_name=model_name, name="shadow_note",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
    ]


class Migration(migrations.Migration):

    dependencies = [
        ("guardrails", "0008_classifiershadowresult"),
    ]

    operations = [
        # Columns by system, not role: classifier_* is the legacy classifier.
        migrations.RenameField("classifiershadowresult", "prod_flagged", "classifier_flagged"),
        migrations.RenameField("classifiershadowresult", "prod_confidence", "classifier_confidence"),
        migrations.RenameField("classifiershadowresult", "prod_tags", "classifier_tags"),
        migrations.AlterField(
            model_name="classifiershadowresult", name="classifier_flagged",
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name="chunkscanshadowresult", name="classifier_flagged",
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name="chunkscanshadowresult", name="quick_pathway",
            field=models.CharField(blank=True, default="", max_length=12),
        ),
        *_direction_fields("chunkscanshadowresult"),
        *_direction_fields("classifiershadowresult"),
        migrations.AddIndex(
            model_name="chunkscanshadowresult",
            index=models.Index(fields=["state", "run_date"], name="guardrails_chunkshadow_state"),
        ),
        migrations.AddIndex(
            model_name="classifiershadowresult",
            index=models.Index(fields=["state", "created_at"], name="guardrails_clfshadow_state"),
        ),
    ]
