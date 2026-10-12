from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0050_chat_attachment_file_max_length'),
    ]

    operations = [
        migrations.AddField(
            model_name='subagentrun',
            name='shared_attachments',
            field=models.JSONField(blank=True, default=list),
        ),
    ]
