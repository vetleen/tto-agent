"""Tests for accounts settings views (theme_update)."""
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import UserSettings

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class ThemeUpdateViewTests(TestCase):
    def setUp(self) -> None:
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="tester@example.com",
            password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("accounts:theme_update")

    def test_requires_login(self) -> None:
        response = self.client.post(self.url, {"theme": "dark"})
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_rejects_get_request(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 405)

    def test_set_theme_to_dark(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"theme": "dark"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["theme"], "dark")
        settings = UserSettings.objects.get(user=self.user)
        self.assertEqual(settings.theme, "dark")

    def test_set_theme_to_light(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        # First set to dark
        self.client.post(self.url, {"theme": "dark"})
        # Then switch back to light
        response = self.client.post(self.url, {"theme": "light"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["theme"], "light")
        settings = UserSettings.objects.get(user=self.user)
        self.assertEqual(settings.theme, "light")

    def test_invalid_theme_returns_400(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"theme": "invalid"})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_empty_theme_returns_400(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"theme": ""})
        self.assertEqual(response.status_code, 400)

    def test_missing_theme_returns_400(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, 400)

    def test_theme_value_is_stripped_and_lowercased(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"theme": "  DARK  "})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["theme"], "dark")

    def test_creates_settings_if_not_exists(self) -> None:
        """theme_update should create UserSettings via get_or_create if missing."""
        self.client.login(email=self.user.email, password=self.password)
        # Delete any auto-created settings from signal
        UserSettings.objects.filter(user=self.user).delete()
        self.assertFalse(UserSettings.objects.filter(user=self.user).exists())

        response = self.client.post(self.url, {"theme": "dark"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(UserSettings.objects.filter(user=self.user).exists())
        self.assertEqual(UserSettings.objects.get(user=self.user).theme, "dark")
