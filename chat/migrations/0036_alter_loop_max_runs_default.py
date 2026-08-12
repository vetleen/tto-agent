from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0035_subagentrun_canvas_subagentrun_canvas_title"),
    ]

    operations = [
        migrations.AlterField(
            model_name="loop",
            name="max_runs",
            field=models.PositiveIntegerField(blank=True, default=50, null=True),
        ),
    ]
