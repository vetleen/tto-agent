# Make Loop.max_runs optional: NULL means "unlimited" (the new default for new
# loops). Existing loops keep whatever finite cap they were created with.
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0028_alter_imageasset_blob'),
    ]

    operations = [
        migrations.AlterField(
            model_name='loop',
            name='max_runs',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
    ]
