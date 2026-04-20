from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver

from django.contrib.auth import get_user_model


@receiver(post_save, sender=get_user_model())
def create_user_settings(sender, instance, created, **kwargs):
    if created:
        from accounts.models import UserSettings

        UserSettings.objects.get_or_create(user=instance)


@receiver(pre_delete, sender=get_user_model())
def redact_llm_call_logs_on_user_delete(sender, instance, **kwargs):
    """GDPR Art. 17: when a user is deleted, redact the content of every
    LLMCallLog row attributed to them. The row itself stays for cost/usage
    analytics, but prompt, raw_output, and tool schemas (which can contain
    user messages and document excerpts) are scrubbed. The existing
    on_delete=SET_NULL on LLMCallLog.user fires after this signal and nulls
    the FK.
    """
    from llm.models import LLMCallLog

    LLMCallLog.objects.filter(user=instance).update(
        prompt={"redacted": True},
        raw_output="",
        tools=None,
    )
