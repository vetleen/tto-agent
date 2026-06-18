# Step 1 of 3 for multi-skill threads: create the ChatThreadSkill through
# table and the ChatThread.skills M2M. The M2M reverse accessor is given a
# TEMPORARY name (`thread_skills_m2m`) because the old `skill` FK still holds
# `chat_threads` on AgentSkill at this point — they cannot coexist. Migration
# 0026 removes the FK and renames this reverse accessor to `chat_threads`.
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('agent_skills', '0001_initial'),
        ('chat', '0023_alter_loop_max_runs'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChatThreadSkill',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('attached_at', models.DateTimeField(auto_now_add=True)),
                ('skill', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='thread_skill_links', to='agent_skills.agentskill')),
                ('thread', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='thread_skills', to='chat.chatthread')),
            ],
            options={
                'ordering': ['attached_at', 'id'],
                'unique_together': {('thread', 'skill')},
            },
        ),
        migrations.AddField(
            model_name='chatthread',
            name='skills',
            field=models.ManyToManyField(blank=True, related_name='thread_skills_m2m', through='chat.ChatThreadSkill', to='agent_skills.agentskill'),
        ),
    ]
