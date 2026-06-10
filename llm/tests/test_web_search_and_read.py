"""Tests for WebSearchAndReadTool."""

import json
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from llm.tools.web_search_and_read import WebSearchAndReadTool


def _no_ssrf_check(url):
    return None


SEARCH_RESULTS = {
    "results": [
        {"title": "Page One", "url": "https://example.com/one", "description": "First result"},
        {"title": "Page Two", "url": "https://example.com/two", "description": "Second result"},
    ],
    "count": 2,
}


@override_settings(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}},
    BRAVE_SEARCH_API_KEY="test-key",
    JINA_API_KEY="",  # disable the Jina fallback so fetch-error tests make no real network call
)
@patch("llm.tools.web_fetch._check_url_ssrf", _no_ssrf_check)
class WebSearchAndReadTests(TestCase):

    def setUp(self):
        self.tool = WebSearchAndReadTool()

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search.BraveSearchTool._run")
    def test_search_and_fetch(self, mock_search, mock_fetch_get):
        mock_search.return_value = json.dumps(SEARCH_RESULTS)

        def make_response(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"Content-Type": "text/html"}
            resp.is_redirect = False
            resp.raise_for_status = MagicMock()
            if "one" in url:
                resp.text = "<html><body><article><p>Content of page one</p></article></body></html>"
            else:
                resp.text = "<html><body><article><p>Content of page two</p></article></body></html>"
            return resp

        mock_fetch_get.side_effect = make_response
        result = json.loads(self.tool.invoke({"query": "test query", "count": 2}))

        self.assertEqual(result["query"], "test query")
        self.assertEqual(result["count"], 2)
        self.assertIn("Content of page one", result["results"][0]["content"])
        self.assertIn("Content of page two", result["results"][1]["content"])
        self.assertEqual(result["results"][0]["title"], "Page One")
        self.assertEqual(result["results"][0]["url"], "https://example.com/one")

    @patch("llm.tools.brave_search.BraveSearchTool._run")
    def test_search_error_propagated(self, mock_search):
        mock_search.return_value = json.dumps({"error": "API unavailable", "results": []})

        result = json.loads(self.tool.invoke({"query": "test"}))
        self.assertIn("error", result)

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search.BraveSearchTool._run")
    def test_fetch_error_included_per_result(self, mock_search, mock_fetch_get):
        mock_search.return_value = json.dumps(SEARCH_RESULTS)

        import requests as req_lib
        mock_fetch_get.side_effect = req_lib.exceptions.ConnectionError("refused")

        result = json.loads(self.tool.invoke({"query": "test"}))
        self.assertEqual(result["count"], 2)
        for r in result["results"]:
            self.assertIn("fetch_error", r)
            self.assertEqual(r["content"], "")

    @patch("llm.tools.brave_search.BraveSearchTool._run")
    def test_empty_results(self, mock_search):
        mock_search.return_value = json.dumps({"results": [], "count": 0})

        result = json.loads(self.tool.invoke({"query": "nothing"}))
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["results"], [])

    def test_count_capped_at_10(self):
        with patch("llm.tools.brave_search.BraveSearchTool._run") as mock_search, \
             patch("llm.tools.web_fetch._pinned_get"):
            mock_search.return_value = json.dumps({"results": [], "count": 0})
            self.tool.invoke({"query": "test", "count": 20})
            call_args = json.loads(mock_search.call_args[0][0]) if mock_search.call_args[0] else {}
            called_count = mock_search.call_args[1].get("count", mock_search.call_args[0][1] if len(mock_search.call_args[0]) > 1 else 5)
            self.assertLessEqual(called_count, 10)

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search.BraveSearchTool._run")
    def test_preserves_search_metadata(self, mock_search, mock_fetch_get):
        mock_search.return_value = json.dumps(SEARCH_RESULTS)

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.text = "<html><body><p>Content</p></body></html>"
        resp.is_redirect = False
        resp.raise_for_status = MagicMock()
        mock_fetch_get.return_value = resp

        result = json.loads(self.tool.invoke({"query": "test"}))
        for r in result["results"]:
            self.assertIn("title", r)
            self.assertIn("url", r)
            self.assertIn("description", r)
