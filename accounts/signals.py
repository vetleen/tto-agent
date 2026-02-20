from django.db.models.signals import post_save
from django.dispatch import receiver

from django.contrib.auth import get_user_model


@receiver(post_save, sender=get_user_model())
def create_user_settings(sender, instance, created, **kwargs):
    if created:
        from accounts.models import UserSettings

        UserSettings.objects.create(user=instance)
