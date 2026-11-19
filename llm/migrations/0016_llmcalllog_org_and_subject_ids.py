from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('llm', '0015_db_defaults_rolling_deploy'),
    ]

    operations = [
        migrations.AddField(
            model_name='llmcalllog',
            name='org_id',
            field=models.PositiveBigIntegerField(blank=True, db_index=True, null=True),
        ),
        migrations.AddField(
            model_name='llmcalllog',
            name='document_id',
            field=models.PositiveBigIntegerField(blank=True, db_index=True, null=True),
        ),
        migrations.AddField(
            model_name='llmcalllog',
            name='data_room_id',
            field=models.PositiveBigIntegerField(blank=True, db_index=True, null=True),
        ),
    ]
