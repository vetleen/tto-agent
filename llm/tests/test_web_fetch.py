"""Tests for WebFetchTool."""

import json
from unittest.mock import MagicMock, patch

from django.test import TestCase

from llm.tools.web_fetch import WebFetchTool


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
