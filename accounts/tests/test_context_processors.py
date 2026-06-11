"""Tests for accounts context processors."""
from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase

from accounts.context_processors import nav_context
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
        context = nav_context(request)
        self.assertIn("theme", context)
        self.assertEqual(context["theme"], UserSettings.Theme.LIGHT)

    def test_returns_dark_theme_when_set(self) -> None:
        settings = UserSettings.objects.get(user=self.user)
        settings.theme = UserSettings.Theme.DARK
        settings.save()

        request = self.factory.get("/")
        request.user = self.user
        context = nav_context(request)
        self.assertEqual(context["theme"], "dark")

    def test_returns_no_theme_for_anonymous_user(self) -> None:
        # Anonymous users still get assistant_name (the landing page is branded),
        # but no theme/admin/budget context, which is user-specific.
        from django.contrib.auth.models import AnonymousUser

        request = self.factory.get("/")
        request.user = AnonymousUser()
        context = nav_context(request)
        self.assertNotIn("theme", context)
        self.assertNotIn("user_is_org_admin", context)

    def test_missing_settings_defaults_to_light_without_creating(self) -> None:
        """The context processor is read-only: a missing UserSettings row
        (only possible for pre-signal legacy users) renders the default theme
        and is NOT auto-created — row creation belongs to the post_save signal."""
        UserSettings.objects.filter(user=self.user).delete()

        request = self.factory.get("/")
        request.user = self.user
        context = nav_context(request)
        self.assertIn("theme", context)
        self.assertEqual(context["theme"], UserSettings.Theme.LIGHT)
        self.assertFalse(UserSettings.objects.filter(user=self.user).exists())
