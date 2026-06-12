"""Tests for rate limiting on auth and LLM-classifier endpoints.

Rate limiting is globally disabled under ``manage.py test`` (see config/settings.py)
because the default cache is a real Redis whose counters survive across runs. The
classes below opt back in with RATELIMIT_ENABLE=True plus an in-memory cache, and
clear that cache in setUp so methods stay order-independent (locmem persists for
the whole process).
"""
import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization
from guardrails.schemas import ClassifierResult

User = get_user_model()

_RATE_LIMIT_SETTINGS = {
    "ALLOWED_HOSTS": ["testserver"],
    "RATELIMIT_ENABLE": True,
    "CACHES": {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
}

# Every login attempt hashes the submitted password; MD5 keeps the 30+ post
# username-throttle tests fast.
_FAST_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

_CLEAN = ClassifierResult(
    is_suspicious=False, concern_tags=[], confidence=0.1,
    reasoning="Clean content.",
)


def _pin_ratelimit_window(testcase: TestCase) -> None:
    """Pin django-ratelimit's time window for the duration of one test.

    The limiter buckets counts into fixed windows (the window id is part of
    the cache key), so a test that posts N times in a loop can straddle a
    window boundary mid-loop and see the count reset — e.g. the 6th post of a
    5/m test lands in a fresh bucket and passes instead of returning 429.
    Pinning the window makes the counting deterministic; per-key isolation is
    unaffected (the key value is hashed into the cache key separately).
    """
    patcher = patch("django_ratelimit.core._get_window", return_value=2_000_000_000)
    patcher.start()
    testcase.addCleanup(patcher.stop)


@override_settings(**_RATE_LIMIT_SETTINGS)
class LoginRateLimitTests(TestCase):
    def setUp(self) -> None:
        cache.clear()
        _pin_ratelimit_window(self)
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
            response = self.client.post(
                self.url,
                {"username": "u@example.com", "password": "wrong"},
                HTTP_ACCEPT="text/html",
            )
        self.assertContains(response, "too many attempts", status_code=429)


@override_settings(**_RATE_LIMIT_SETTINGS)
class PasswordResetRateLimitTests(TestCase):
    def setUp(self) -> None:
        cache.clear()
        _pin_ratelimit_window(self)
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


@override_settings(**_RATE_LIMIT_SETTINGS)
class XffSpoofingRegressionTests(TestCase):
    """The IP rate-limit key must use the LAST X-Forwarded-For entry.

    Heroku appends the true client IP as the last entry; earlier entries are
    client-controlled. The old configuration fed the whole header to
    ``ipaddress.ip_network`` and 500'd on any multi-entry value.
    """

    def setUp(self) -> None:
        cache.clear()
        _pin_ratelimit_window(self)
        self.url = reverse("accounts:login")

    def _post(self, xff: str):
        return self.client.post(
            self.url,
            {"username": "u@example.com", "password": "wrong"},
            HTTP_X_FORWARDED_FOR=xff,
        )

    def test_multi_entry_xff_does_not_500(self) -> None:
        response = self._post("spoofed-junk, 10.0.0.1, 203.0.113.7")
        self.assertEqual(response.status_code, 200)

    def test_spoofed_prefixes_cannot_evade_the_limit(self) -> None:
        for i in range(5):
            self._post(f"10.{i}.0.1, 203.0.113.7")
        response = self._post("10.99.0.1, 203.0.113.7")
        self.assertEqual(response.status_code, 429)

    def test_distinct_client_ips_are_isolated(self) -> None:
        for i in range(5):
            self._post(f"10.{i}.0.1, 203.0.113.7")
        response = self._post("spoofed-junk, 198.51.100.9")
        self.assertNotEqual(response.status_code, 429)


@override_settings(**_RATE_LIMIT_SETTINGS, PASSWORD_HASHERS=_FAST_HASHERS)
class UsernameLoginRateLimitTests(TestCase):
    """Distributed (multi-IP) brute force against one account is bounded at 30/h."""

    def setUp(self) -> None:
        cache.clear()
        _pin_ratelimit_window(self)
        self.url = reverse("accounts:login")
        User.objects.create_user(email="victim@example.com", password="right-horse")

    def _post(self, username: str, ip: str):
        return self.client.post(
            self.url,
            {"username": username, "password": "wrong"},
            REMOTE_ADDR=ip,
        )

    def test_distributed_attempts_on_one_account_blocked_at_31(self) -> None:
        for i in range(30):
            response = self._post("victim@example.com", f"10.0.{i}.1")
            self.assertNotEqual(response.status_code, 429)
        response = self._post("victim@example.com", "10.0.99.1")
        self.assertEqual(response.status_code, 429)

    def test_case_and_whitespace_variants_share_one_bucket(self) -> None:
        variants = ["victim@example.com", "Victim@Example.COM", " victim@example.com "]
        for i in range(30):
            self._post(variants[i % len(variants)], f"10.1.{i}.1")
        response = self._post("VICTIM@example.com", "10.1.99.1")
        self.assertEqual(response.status_code, 429)

    def test_empty_username_posts_do_not_share_a_bucket(self) -> None:
        # The key falls back to the client IP when the field is empty, so
        # malformed posts from distinct sources must never pool into one bucket.
        for i in range(31):
            response = self._post("", f"10.2.{i}.1")
            self.assertNotEqual(response.status_code, 429)


@override_settings(**_RATE_LIMIT_SETTINGS)
class AdminLoginRateLimitTests(TestCase):
    """/admin/login/ carries the same per-IP throttle as the main login form."""

    def setUp(self) -> None:
        cache.clear()
        _pin_ratelimit_window(self)
        self.url = reverse("admin:login")

    def test_blocks_6th_post_in_one_minute(self) -> None:
        for _ in range(5):
            response = self.client.post(self.url, {
                "username": "staff@example.com",
                "password": "wrong",
            })
            self.assertNotEqual(response.status_code, 429)
        response = self.client.post(self.url, {
            "username": "staff@example.com",
            "password": "wrong",
        })
        self.assertEqual(response.status_code, 429)

    def test_get_is_not_throttled(self) -> None:
        for _ in range(6):
            self.client.post(self.url, {
                "username": "staff@example.com",
                "password": "wrong",
            })
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)


@override_settings(**_RATE_LIMIT_SETTINGS)
class RateLimited429ContentNegotiationTests(TestCase):
    """Browser form posts get the branded page; fetch() callers get JSON."""

    def setUp(self) -> None:
        cache.clear()
        _pin_ratelimit_window(self)
        self.url = reverse("accounts:login")
        self.payload = {"username": "u@example.com", "password": "wrong"}
        for _ in range(6):
            self.client.post(self.url, self.payload)

    def test_html_when_client_accepts_html(self) -> None:
        response = self.client.post(
            self.url, self.payload, HTTP_ACCEPT="text/html,application/xhtml+xml",
        )
        self.assertEqual(response.status_code, 429)
        self.assertIn("text/html", response["Content-Type"])
        self.assertContains(response, "too many attempts", status_code=429)

    def test_json_when_no_accept_header(self) -> None:
        response = self.client.post(self.url, self.payload)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertIn("error", response.json())

    def test_json_when_client_accepts_json(self) -> None:
        response = self.client.post(
            self.url, self.payload, HTTP_ACCEPT="application/json",
        )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertIn("error", response.json())


@override_settings(**_RATE_LIMIT_SETTINGS)
class LlmEndpointRateLimitTests(TestCase):
    """The endpoints that fire a synchronous LLM classifier call are throttled
    per user at 10/m (cost-amplification guard)."""

    def setUp(self) -> None:
        cache.clear()
        _pin_ratelimit_window(self)
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="admin@test.com", password=self.password
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Test Org", slug="test-org")
        Membership.objects.create(
            user=self.admin_user, org=self.org, role=Membership.Role.ADMIN
        )

        for target in ("classify_description_sync", "classify_soul_sync"):
            patcher = patch(f"guardrails.classifier.{target}", return_value=_CLEAN)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.client.login(email=self.admin_user.email, password=self.password)

    def _post(self, url_name: str, payload: dict):
        return self.client.post(
            reverse(url_name), json.dumps(payload), content_type="application/json",
        )

    def test_eleventh_post_within_a_minute_is_throttled(self) -> None:
        endpoints = [
            ("accounts:profile_update", {"description": "Test"}),
            ("accounts:soul_update", {"soul": "Test"}),
            ("accounts:org_soul_update", {"soul": "Test"}),
            ("accounts:org_description_update", {"description": "Test"}),
            ("accounts:org_name_update", {"name": "Test Org"}),
        ]
        # Buckets are per-view (the group derives from the view's dotted path),
        # so one user's posts to different endpoints never interfere.
        for url_name, payload in endpoints:
            with self.subTest(endpoint=url_name):
                for _ in range(10):
                    response = self._post(url_name, payload)
                    self.assertEqual(response.status_code, 200)
                response = self._post(url_name, payload)
                self.assertEqual(response.status_code, 429)
                self.assertEqual(response["Content-Type"], "application/json")
                self.assertIn("error", response.json())

    def test_throttle_is_per_user(self) -> None:
        for _ in range(11):
            self._post("accounts:profile_update", {"description": "Test"})

        member = User.objects.create_user(
            email="member@test.com", password=self.password
        )
        member.email_verified = True
        member.save(update_fields=["email_verified"])
        self.client.logout()
        self.client.login(email=member.email, password=self.password)
        response = self._post("accounts:profile_update", {"description": "Test"})
        self.assertEqual(response.status_code, 200)
