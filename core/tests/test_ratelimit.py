"""Unit tests for the django-ratelimit helpers in core/ratelimit.py."""
from django.test import RequestFactory, SimpleTestCase

from core.ratelimit import client_ip, login_username_or_ip


class ClientIpTests(SimpleTestCase):
    def setUp(self) -> None:
        self.factory = RequestFactory()

    def test_no_xff_falls_back_to_remote_addr(self) -> None:
        request = self.factory.get("/")
        self.assertEqual(client_ip(request), "127.0.0.1")

    def test_single_entry_returned(self) -> None:
        request = self.factory.get("/", HTTP_X_FORWARDED_FOR="203.0.113.7")
        self.assertEqual(client_ip(request), "203.0.113.7")

    def test_multi_entry_returns_last(self) -> None:
        # Heroku appends the true client IP last; earlier entries are spoofable.
        request = self.factory.get(
            "/", HTTP_X_FORWARDED_FOR="spoofed-junk, 10.0.0.1, 203.0.113.7"
        )
        self.assertEqual(client_ip(request), "203.0.113.7")

    def test_whitespace_stripped(self) -> None:
        request = self.factory.get(
            "/", HTTP_X_FORWARDED_FOR="10.0.0.1 ,  203.0.113.7  "
        )
        self.assertEqual(client_ip(request), "203.0.113.7")

    def test_empty_header_falls_back(self) -> None:
        request = self.factory.get("/", HTTP_X_FORWARDED_FOR="")
        self.assertEqual(client_ip(request), "127.0.0.1")

    def test_trailing_comma_falls_back(self) -> None:
        request = self.factory.get("/", HTTP_X_FORWARDED_FOR="10.0.0.1,")
        self.assertEqual(client_ip(request), "127.0.0.1")

    def test_ipv6_last_entry(self) -> None:
        request = self.factory.get(
            "/", HTTP_X_FORWARDED_FOR="10.0.0.1, 2001:db8::1"
        )
        self.assertEqual(client_ip(request), "2001:db8::1")


class LoginUsernameOrIpTests(SimpleTestCase):
    def setUp(self) -> None:
        self.factory = RequestFactory()

    def test_username_normalized(self) -> None:
        request = self.factory.post(
            "/accounts/login/", {"username": "  User@Example.COM "}
        )
        self.assertEqual(
            login_username_or_ip("group", request), "username:user@example.com"
        )

    def test_missing_username_falls_back_to_ip(self) -> None:
        request = self.factory.post("/accounts/login/", {})
        self.assertEqual(login_username_or_ip("group", request), "ip:127.0.0.1")

    def test_empty_username_falls_back_to_ip(self) -> None:
        request = self.factory.post("/accounts/login/", {"username": "   "})
        self.assertEqual(login_username_or_ip("group", request), "ip:127.0.0.1")

    def test_get_request_falls_back_to_ip(self) -> None:
        request = self.factory.get("/accounts/login/")
        self.assertEqual(login_username_or_ip("group", request), "ip:127.0.0.1")

    def test_ip_fallback_uses_last_xff_entry(self) -> None:
        request = self.factory.post(
            "/accounts/login/", {}, HTTP_X_FORWARDED_FOR="spoofed, 203.0.113.7"
        )
        self.assertEqual(login_username_or_ip("group", request), "ip:203.0.113.7")
