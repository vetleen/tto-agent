"""Tests for accounts signals (create_user_settings + blob cleanup on delete)."""
import tempfile

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings

from accounts.models import FontAsset, Organization, UserSettings

User = get_user_model()


class CreateUserSettingsSignalTests(TestCase):
    def test_user_creation_creates_settings(self) -> None:
        user = User.objects.create_user(email="sig@example.com", password="pass")
        self.assertTrue(UserSettings.objects.filter(user=user).exists())

    def test_created_settings_have_default_theme(self) -> None:
        user = User.objects.create_user(email="sig@example.com", password="pass")
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.theme, UserSettings.Theme.SYSTEM)

    def test_superuser_creation_creates_settings(self) -> None:
        user = User.objects.create_superuser(email="admin@example.com", password="pass")
        self.assertTrue(UserSettings.objects.filter(user=user).exists())

    def test_signal_does_not_duplicate_on_save(self) -> None:
        """Saving an existing user should not create a second UserSettings."""
        user = User.objects.create_user(email="sig@example.com", password="pass")
        self.assertEqual(UserSettings.objects.filter(user=user).count(), 1)

        user.email = "updated@example.com"
        user.save()
        self.assertEqual(UserSettings.objects.filter(user=user).count(), 1)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class DeleteBlobOnDeleteTests(TestCase):
    """post_delete receivers must remove FileField blobs from storage so
    deleting a user/org/font asset doesn't orphan files (GDPR + storage hygiene)."""

    def test_user_delete_removes_avatar_blobs(self) -> None:
        user = User.objects.create_user(email="pic@example.com", password="pass")
        user.profile_picture.save("pic.jpg", ContentFile(b"imgdata"), save=True)
        user.profile_picture_original.save("orig.jpg", ContentFile(b"origdata"), save=True)
        storage = user.profile_picture.storage
        pic_name = user.profile_picture.name
        orig_name = user.profile_picture_original.name
        self.assertTrue(storage.exists(pic_name))
        self.assertTrue(storage.exists(orig_name))

        user.delete()

        self.assertFalse(storage.exists(pic_name))
        self.assertFalse(storage.exists(orig_name))

    def test_org_delete_removes_logo_blob(self) -> None:
        org = Organization.objects.create(name="LogoOrg", slug="logoorg")
        org.logo.save("logo.png", ContentFile(b"logodata"), save=True)
        storage = org.logo.storage
        name = org.logo.name
        self.assertTrue(storage.exists(name))

        org.delete()

        self.assertFalse(storage.exists(name))

    def test_fontasset_queryset_delete_removes_blob(self) -> None:
        # Connecting the post_delete receiver stops Django from fast-deleting
        # FontAsset, so even a queryset .delete() cleans the blob.
        org = Organization.objects.create(name="FontOrg", slug="fontorg")
        asset = FontAsset.objects.create(
            organization=org, family="X", family_norm="x", weight=400,
        )
        asset.blob.save("x.ttf", ContentFile(b"fontdata"), save=True)
        storage = asset.blob.storage
        name = asset.blob.name
        self.assertTrue(storage.exists(name))

        FontAsset.objects.filter(pk=asset.pk).delete()

        self.assertFalse(storage.exists(name))
