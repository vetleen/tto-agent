"""Tests for accounts context processors."""
from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase

from accounts.context_processors import theme
from accounts.models import UserSettings

User = get_user_model()


class ThemeContextProcessorTests(TestCase):
    def setUp(self) -> None:
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            email="ctx@example.com",
            password="pass",
        )

    def test_returns_theme_for_authenticated_user(self) -> None:
        request = self.factory.get("/")
        request.user = self.user
        context = theme(request)
        self.assertIn("theme", context)
        self.assertEqual(context["theme"], UserSettings.Theme.LIGHT)

    def test_returns_dark_theme_when_set(self) -> None:
        settings = UserSettings.objects.get(user=self.user)
        settings.theme = UserSettings.Theme.DARK
        settings.save()

        request = self.factory.get("/")
        request.user = self.user
        context = theme(request)
        self.assertEqual(context["theme"], "dark")

    def test_returns_empty_context_for_anonymous_user(self) -> None:
        from django.contrib.auth.models import AnonymousUser

        request = self.factory.get("/")
        request.user = AnonymousUser()
        context = theme(request)
        self.assertEqual(context, {})
        self.assertNotIn("theme", context)

    def test_creates_settings_if_missing(self) -> None:
        """Context processor uses get_or_create, so it should handle missing settings."""
        UserSettings.objects.filter(user=self.user).delete()

        request = self.factory.get("/")
        request.user = self.user
        context = theme(request)
        self.assertIn("theme", context)
        self.assertEqual(context["theme"], UserSettings.Theme.LIGHT)
        self.assertTrue(UserSettings.objects.filter(user=self.user).exists())
