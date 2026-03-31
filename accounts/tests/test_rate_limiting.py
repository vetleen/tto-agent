"""Tests for rate limiting on auth endpoints."""
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

User = get_user_model()

_RATE_LIMIT_SETTINGS = {
    "ALLOWED_HOSTS": ["testserver"],
    "RATELIMIT_ENABLE": True,
    "CACHES": {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
}


@override_settings(**_RATE_LIMIT_SETTINGS)
class LoginRateLimitTests(TestCase):
    def setUp(self) -> None:
        self.url = reverse("accounts:login")
        self.user = User.objects.create_user(email="u@example.com", password="pass")

    def test_allows_up_to_5_posts_per_minute(self) -> None:
        for _ in range(5):
            response = self.client.post(self.url, {
                "username": "u@example.com",
                "password": "wrong",
            })
            self.assertNotEqual(response.status_code, 429)

    def test_blocks_6th_post_in_one_minute(self) -> None:
        for _ in range(5):
            self.client.post(self.url, {
                "username": "u@example.com",
                "password": "wrong",
            })
        response = self.client.post(self.url, {
            "username": "u@example.com",
            "password": "wrong",
        })
        self.assertEqual(response.status_code, 429)

    def test_rate_limited_page_renders(self) -> None:
        for _ in range(6):
            response = self.client.post(self.url, {
                "username": "u@example.com",
                "password": "wrong",
            })
        self.assertContains(response, "Too many attempts", status_code=429)


@override_settings(**_RATE_LIMIT_SETTINGS)
class PasswordResetRateLimitTests(TestCase):
    def setUp(self) -> None:
        self.url = reverse("accounts:password_reset")

    def test_allows_up_to_3_posts_per_hour(self) -> None:
        for _ in range(3):
            response = self.client.post(self.url, {"email": "any@example.com"})
            self.assertNotEqual(response.status_code, 429)

    def test_blocks_4th_post_in_one_hour(self) -> None:
        for _ in range(3):
            self.client.post(self.url, {"email": "any@example.com"})
        response = self.client.post(self.url, {"email": "any@example.com"})
        self.assertEqual(response.status_code, 429)
