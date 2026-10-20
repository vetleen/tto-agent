from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0055_subagent_tier_rename"),
    ]

    operations = [
        migrations.AddField(
            model_name="subagentrun",
            name="round_budget",
            field=models.CharField(default="standard", max_length=10),
        ),
        migrations.AddField(
            model_name="subagentrun",
            name="rounds_used",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="subagentrun",
            name="budget_exhausted",
            field=models.BooleanField(default=False),
        ),
    ]
