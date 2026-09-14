from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('agent_skills', '0008_skillresource_optimized_file'),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name='agentskill',
            name='unique_system_skill_slug',
        ),
        migrations.RemoveConstraint(
            model_name='agentskill',
            name='unique_org_skill_slug',
        ),
        migrations.RemoveConstraint(
            model_name='agentskill',
            name='unique_user_skill_slug',
        ),
        migrations.AddField(
            model_name='agentskill',
            name='deleted_at',
            field=models.DateTimeField(blank=True, db_index=True, null=True),
        ),
        migrations.AddConstraint(
            model_name='agentskill',
            constraint=models.UniqueConstraint(
                condition=models.Q(('level', 'system'), ('deleted_at__isnull', True)),
                fields=('slug',),
                name='unique_system_skill_slug',
            ),
        ),
        migrations.AddConstraint(
            model_name='agentskill',
            constraint=models.UniqueConstraint(
                condition=models.Q(('level', 'org'), ('deleted_at__isnull', True)),
                fields=('slug', 'organization'),
                name='unique_org_skill_slug',
            ),
        ),
        migrations.AddConstraint(
            model_name='agentskill',
            constraint=models.UniqueConstraint(
                condition=models.Q(('level', 'user'), ('deleted_at__isnull', True)),
                fields=('slug', 'created_by'),
                name='unique_user_skill_slug',
            ),
        ),
    ]
