import logging

from django.db.models.signals import post_delete, post_save, pre_delete
from django.dispatch import receiver

from django.contrib.auth import get_user_model

logger = logging.getLogger(__name__)


def _delete_file_field(file_field, *, model, pk):
    """Delete a FileField's underlying blob from storage, tolerating a missing
    file or a storage error (which must not abort the delete). Mirrors the
    post_delete cleanup pattern in documents/meetings/chat/feedback signals."""
    if not file_field:
        return
    name = file_field.name
    if not name:
        return
    try:
        file_field.storage.delete(name)
    except Exception:
        logger.exception(
            "Failed to delete %s blob (pk=%s path=%s)", model, pk, name
        )


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


@receiver(post_delete, sender=get_user_model())
def delete_profile_pictures_on_user_delete(sender, instance, **kwargs):
    """Remove the user's avatar blobs from storage when the row is deleted
    (Django does not clean FileFields on delete). Without this, GDPR erasure
    leaves the person's photos in the media bucket indefinitely."""
    _delete_file_field(instance.profile_picture, model="User.profile_picture", pk=instance.pk)
    _delete_file_field(
        instance.profile_picture_original,
        model="User.profile_picture_original",
        pk=instance.pk,
    )


@receiver(post_delete, sender="accounts.Organization")
def delete_logo_on_org_delete(sender, instance, **kwargs):
    """Remove the org logo blob from storage on delete."""
    _delete_file_field(instance.logo, model="Organization.logo", pk=instance.pk)


@receiver(post_delete, sender="accounts.FontAsset")
def delete_blob_on_font_asset_delete(sender, instance, **kwargs):
    """Remove the font blob from storage on delete. Connecting this receiver
    also stops Django from fast-deleting FontAsset, so per-object cleanup now
    fires for both queryset .delete() (org_fonts_delete) and Organization
    cascade deletes."""
    _delete_file_field(instance.blob, model="FontAsset.blob", pk=instance.pk)
