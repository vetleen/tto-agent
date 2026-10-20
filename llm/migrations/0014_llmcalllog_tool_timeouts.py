from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("llm", "0013_opsusagelog_throttle_wait_ms"),
    ]

    operations = [
        migrations.AddField(
            model_name="llmcalllog",
            name="tool_timeout_count",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="llmcalllog",
            name="tool_timeouts",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
