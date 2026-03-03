from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("llm", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="llmcalllog",
            name="raw_prompt",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
