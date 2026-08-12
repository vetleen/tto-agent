"""Tests for CanonicalHostMiddleware (www -> bare canonical host redirect)."""
from django.http import HttpResponse
from django.test import RequestFactory, SimpleTestCase, override_settings

from core.middleware import CanonicalHostMiddleware

# Sentinel response returned when the middleware passes the request through
# instead of redirecting — lets a test assert "not redirected" by identity.
_PASSTHROUGH = HttpResponse("ok")


def _get_response(request):
    return _PASSTHROUGH


@override_settings(
    ALLOWED_HOSTS=[
        "wilfred.work",
        "www.wilfred.work",
        "wilfred-production-74f112c1b31b.herokuapp.com",
    ]
)
class CanonicalHostMiddlewareTests(SimpleTestCase):
    def setUp(self):
        self.rf = RequestFactory()
        self.mw = CanonicalHostMiddleware(_get_response)

    @override_settings(CANONICAL_HOST="wilfred.work")
    def test_www_redirected_to_bare_host(self):
        resp = self.mw(self.rf.get("/dashboard/", HTTP_HOST="www.wilfred.work"))
        self.assertEqual(resp.status_code, 301)
        self.assertEqual(resp["Location"], "http://wilfred.work/dashboard/")

    @override_settings(CANONICAL_HOST="wilfred.work")
    def test_query_string_preserved(self):
        resp = self.mw(self.rf.get("/search?q=hi&p=2", HTTP_HOST="www.wilfred.work"))
        self.assertEqual(resp.status_code, 301)
        self.assertEqual(resp["Location"], "http://wilfred.work/search?q=hi&p=2")

    @override_settings(CANONICAL_HOST="wilfred.work")
    def test_https_scheme_preserved(self):
        # secure=True -> request.scheme == "https"; the redirect must keep it.
        resp = self.mw(self.rf.get("/", secure=True, HTTP_HOST="www.wilfred.work"))
        self.assertEqual(resp.status_code, 301)
        self.assertEqual(resp["Location"], "https://wilfred.work/")

    @override_settings(CANONICAL_HOST="wilfred.work")
    def test_case_insensitive_host_match(self):
        resp = self.mw(self.rf.get("/", HTTP_HOST="WWW.Wilfred.Work"))
        self.assertEqual(resp.status_code, 301)
        self.assertEqual(resp["Location"], "http://wilfred.work/")

    @override_settings(CANONICAL_HOST="wilfred.work")
    def test_bare_canonical_host_passes_through(self):
        resp = self.mw(self.rf.get("/", HTTP_HOST="wilfred.work"))
        self.assertIs(resp, _PASSTHROUGH)

    @override_settings(CANONICAL_HOST="wilfred.work")
    def test_herokuapp_host_passes_through(self):
        resp = self.mw(
            self.rf.get(
                "/", HTTP_HOST="wilfred-production-74f112c1b31b.herokuapp.com"
            )
        )
        self.assertIs(resp, _PASSTHROUGH)

    @override_settings(CANONICAL_HOST="")
    def test_disabled_when_unset(self):
        # Staging / local leave the setting empty: www is not touched.
        resp = self.mw(self.rf.get("/", HTTP_HOST="www.wilfred.work"))
        self.assertIs(resp, _PASSTHROUGH)
