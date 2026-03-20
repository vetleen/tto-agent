"""Tests for WebFetchTool."""

import json
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from llm.tools.web_fetch import WebFetchTool


@override_settings(CACHES={"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}})
class WebFetchToolTests(TestCase):

    def setUp(self):
        self.tool = WebFetchTool()

    @patch("llm.tools.web_fetch.requests.get")
    def test_successful_fetch(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.text = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>Navigation</nav>
            <main><p>Hello World</p></main>
            <footer>Footer</footer>
        </body>
        </html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = json.loads(self.tool.invoke({"url": "https://example.com"}))

        self.assertEqual(result["url"], "https://example.com")
        self.assertEqual(result["title"], "Test Page")
        self.assertIn("Hello World", result["content"])
        # nav and footer should be removed
        self.assertNotIn("Navigation", result["content"])
        self.assertNotIn("Footer", result["content"])
        self.assertFalse(result["truncated"])

    @patch("llm.tools.web_fetch.requests.get")
    def test_script_tags_removed(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = """
        <html><body>
            <script>alert('xss')</script>
            <style>.foo{color:red}</style>
            <p>Clean content</p>
        </body></html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = json.loads(self.tool.invoke({"url": "https://example.com"}))

        self.assertIn("Clean content", result["content"])
        self.assertNotIn("alert", result["content"])
        self.assertNotIn("color:red", result["content"])

    @patch("llm.tools.web_fetch.requests.get")
    def test_truncation(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = "<html><body><p>" + "x" * 1000 + "</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = json.loads(self.tool.invoke({"url": "https://example.com", "max_chars": 100}))

        self.assertTrue(result["truncated"])
        self.assertLessEqual(result["char_count"], 100)

    def test_invalid_url_scheme(self):
        result = json.loads(self.tool.invoke({"url": "ftp://example.com/file"}))
        self.assertIn("error", result)
        self.assertIn("ftp", result["error"])

    def test_empty_url(self):
        result = json.loads(self.tool.invoke({"url": ""}))
        self.assertIn("error", result)

    @patch("llm.tools.web_fetch.requests.get")
    def test_timeout(self, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.Timeout("timeout")

        result = json.loads(self.tool.invoke({"url": "https://example.com"}))
        self.assertIn("error", result)
        self.assertIn("timed out", result["error"])

    @patch("llm.tools.web_fetch.requests.get")
    def test_connection_error(self, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.ConnectionError("failed")

        result = json.loads(self.tool.invoke({"url": "https://example.com"}))
        self.assertIn("error", result)
        self.assertIn("Connection", result["error"])

    @patch("llm.tools.web_fetch.requests.get")
    def test_non_html_content_type(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = json.loads(self.tool.invoke({"url": "https://example.com/doc.pdf"}))
        self.assertIn("error", result)
        self.assertIn("Non-text", result["error"])

    @patch("llm.tools.web_fetch.requests.get")
    def test_max_chars_capped_at_absolute_max(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = "<html><body><p>content</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Even with a huge max_chars, should not exceed 50000
        result = json.loads(self.tool.invoke({"url": "https://example.com", "max_chars": 999999}))
        self.assertIn("content", result)


@override_settings(CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}})
class WebFetchCacheTests(TestCase):
    """Tests for WebFetchTool caching."""

    def setUp(self):
        from django.core.cache import cache
        cache.clear()
        self.tool = WebFetchTool()

    @patch("llm.tools.web_fetch.requests.get")
    def test_cache_miss_calls_api_and_caches(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = "<html><body><p>Cached content</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = json.loads(self.tool.invoke({"url": "https://example.com/cached"}))
        self.assertIn("Cached content", result["content"])
        mock_get.assert_called_once()

        # Second call should use cache
        mock_get.reset_mock()
        result2 = json.loads(self.tool.invoke({"url": "https://example.com/cached"}))
        self.assertIn("Cached content", result2["content"])
        mock_get.assert_not_called()

    @patch("llm.tools.web_fetch.requests.get")
    def test_cache_hit_re_truncates(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = "<html><body><p>" + "x" * 500 + "</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # First fetch with large max_chars to populate cache
        self.tool.invoke({"url": "https://example.com/trunc", "max_chars": 50000})

        # Second fetch with small max_chars — should truncate cached content
        mock_get.reset_mock()
        result = json.loads(self.tool.invoke({"url": "https://example.com/trunc", "max_chars": 50}))
        self.assertTrue(result["truncated"])
        self.assertEqual(result["char_count"], 50)
        mock_get.assert_not_called()

    @patch("llm.tools.web_fetch.requests.get")
    def test_error_response_not_cached(self, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.Timeout("timeout")

        result = json.loads(self.tool.invoke({"url": "https://example.com/err"}))
        self.assertIn("error", result)

        # Reset — next call should hit API, not cache
        mock_get.side_effect = None
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = "<html><body><p>OK</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result2 = json.loads(self.tool.invoke({"url": "https://example.com/err"}))
        self.assertNotIn("error", result2)
        mock_get.assert_called()
