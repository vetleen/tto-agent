from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("agent_skills", "0011_skill_sharing"),
    ]

    operations = [
        migrations.AddField(
            model_name="agentskill",
            name="kind",
            field=models.CharField(
                choices=[("playbook", "Playbook"), ("toolkit", "Toolkit")],
                default="playbook",
                max_length=10,
            ),
        ),
        migrations.AddField(
            model_name="agentskill",
            name="requires",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
