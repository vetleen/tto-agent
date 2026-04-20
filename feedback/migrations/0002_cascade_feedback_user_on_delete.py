import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def delete_orphaned_feedback(apps, schema_editor):
    """GDPR: rows where user is NULL are already unattributed personal data
    (user_agent, screenshot, free-text) with no owner. Deleting them matches
    the new CASCADE policy and is required before we can add NOT NULL.
    """
    Feedback = apps.get_model("feedback", "Feedback")
    Feedback.objects.filter(user__isnull=True).delete()


def noop(apps, schema_editor):
    # Irreversible: once rows are gone we cannot resurrect them.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("feedback", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RunPython(delete_orphaned_feedback, reverse_code=noop),
        migrations.AlterField(
            model_name="feedback",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="feedback_submissions",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
