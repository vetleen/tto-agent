from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0025_piiscanshadowrun_piiscanshadowresult"),
    ]

    operations = [
        migrations.AlterField(
            model_name="piiscanshadowresult", name="quick_pathway",
            field=models.CharField(blank=True, default="", max_length=12),
        ),
        migrations.AddField(
            model_name="piiscanshadowresult", name="classifier_categories",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="piiscanshadowresult", name="direction",
            field=models.CharField(default="decisions_shadow", max_length=16),
        ),
        migrations.AddField(
            model_name="piiscanshadowresult", name="state",
            field=models.CharField(default="done", max_length=12),
        ),
        migrations.AddField(
            model_name="piiscanshadowresult", name="shadow_note",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddIndex(
            model_name="piiscanshadowresult",
            index=models.Index(fields=["state", "run_date"], name="documents_piishadow_state"),
        ),
    ]
