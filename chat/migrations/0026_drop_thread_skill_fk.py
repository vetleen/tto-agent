# Step 3 of 3 for multi-skill threads: drop the old single-skill `skill` FK
# (which frees the `chat_threads` reverse accessor on AgentSkill) and rename
# the M2M's temporary reverse accessor from `thread_skills_m2m` to the final
# `chat_threads`. After this the model state matches chat/models.py.
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0025_copy_thread_skill_to_m2m'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='chatthread',
            name='skill',
        ),
        migrations.AlterField(
            model_name='chatthread',
            name='skills',
            field=models.ManyToManyField(blank=True, related_name='chat_threads', through='chat.ChatThreadSkill', to='agent_skills.agentskill'),
        ),
    ]
