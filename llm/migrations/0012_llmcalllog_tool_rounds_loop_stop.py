from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("llm", "0011_llmcalllog_output_truncated_rounds"),
    ]

    operations = [
        migrations.AddField(
            model_name="llmcalllog",
            name="tool_rounds",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="llmcalllog",
            name="loop_stop",
            field=models.CharField(blank=True, default="", max_length=16),
        ),
    ]
