from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("llm", "0012_llmcalllog_tool_rounds_loop_stop"),
    ]

    operations = [
        migrations.AddField(
            model_name="opsusagelog",
            name="throttle_wait_ms",
            field=models.PositiveIntegerField(default=0),
        ),
    ]
