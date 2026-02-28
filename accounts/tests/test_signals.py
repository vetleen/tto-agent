"""Tests for accounts signals (create_user_settings on user creation)."""
from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import UserSettings

User = get_user_model()


class CreateUserSettingsSignalTests(TestCase):
    def test_user_creation_creates_settings(self) -> None:
        user = User.objects.create_user(email="sig@example.com", password="pass")
        self.assertTrue(UserSettings.objects.filter(user=user).exists())

    def test_created_settings_have_default_theme(self) -> None:
        user = User.objects.create_user(email="sig@example.com", password="pass")
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.theme, UserSettings.Theme.LIGHT)

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
