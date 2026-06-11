"""Tests for WebSearchAndReadTool."""

from unittest.mock import ANY, MagicMock, patch

from django.test import TestCase, override_settings

from llm.tools.web_search_and_read import WebSearchAndReadTool


def _no_ssrf_check(url):
    return None


def _search_data(results=None):
    """Build a _search_core-shaped result dict."""
    if results is None:
        results = [
            {"type": "web", "title": "Page One", "url": "https://example.com/one",
             "description": "First result", "age": "2 days ago", "extra_snippets": []},
            {"type": "web", "title": "Page Two", "url": "https://example.com/two",
             "description": "Second result", "age": "", "extra_snippets": []},
        ]
    return {"query": "test query", "results": results, "count": len(results)}


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
    @patch("llm.tools.brave_search._search_core")
    def test_search_and_fetch(self, mock_search, mock_fetch_get):
        mock_search.return_value = _search_data()

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
        result = self.tool.invoke({"query": "test query", "count": 2})

        self.assertIn("Search: test query", result)
        self.assertIn("Results: 2", result)
        self.assertIn("Content of page one", result)
        self.assertIn("Content of page two", result)
        self.assertIn("Source 1: Page One", result)
        self.assertIn("Source 2: Page Two", result)
        self.assertIn("URL: https://example.com/one", result)
        self.assertIn("(2 days ago)", result)

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search._search_core")
    def test_delimiters_wrap_output(self, mock_search, mock_fetch_get):
        mock_search.return_value = _search_data()

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.text = "<html><body><p>Content</p></body></html>"
        resp.is_redirect = False
        resp.raise_for_status = MagicMock()
        mock_fetch_get.return_value = resp

        result = self.tool.invoke({"query": "test"})

        self.assertIn("=== BEGIN EXTERNAL WEB CONTENT", result)
        self.assertIn("=== END EXTERNAL WEB CONTENT ===", result)
        # Single outer delimiter pair, not one per source
        self.assertEqual(result.count("=== BEGIN EXTERNAL WEB CONTENT"), 1)

    @patch("llm.tools.brave_search._search_core")
    def test_search_error_propagated(self, mock_search):
        mock_search.return_value = {"error": "API unavailable", "results": []}

        result = self.tool.invoke({"query": "test"})
        self.assertIn("Search error", result)
        self.assertIn("API unavailable", result)

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search._search_core")
    def test_fetch_error_included_per_result(self, mock_search, mock_fetch_get):
        mock_search.return_value = _search_data()

        import requests as req_lib
        mock_fetch_get.side_effect = req_lib.exceptions.ConnectionError("refused")

        result = self.tool.invoke({"query": "test"})
        self.assertEqual(result.count("(Could not fetch content:"), 2)
        # Search snippets still present so the model can use them
        self.assertIn("First result", result)
        self.assertIn("Second result", result)

    @patch("llm.tools.brave_search._search_core")
    def test_empty_results(self, mock_search):
        mock_search.return_value = {"query": "nothing", "results": [], "count": 0}

        result = self.tool.invoke({"query": "nothing"})
        self.assertIn("No results found", result)

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search._search_core")
    def test_count_capped_at_20(self, mock_search, mock_fetch_get):
        mock_search.return_value = {"query": "test", "results": [], "count": 0}

        self.tool.invoke({"query": "test", "count": 50})

        mock_search.assert_called_once_with(
            "test", count=20, freshness="", categories=["web"], context=ANY,
        )

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search._search_core")
    def test_freshness_forwarded(self, mock_search, mock_fetch_get):
        mock_search.return_value = {"query": "test", "results": [], "count": 0}

        self.tool.invoke({"query": "test", "freshness": "pm"})

        mock_search.assert_called_once_with(
            "test", count=5, freshness="pm", categories=["web"], context=ANY,
        )

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search._search_core")
    def test_invalid_freshness_dropped(self, mock_search, mock_fetch_get):
        mock_search.return_value = {"query": "test", "results": [], "count": 0}

        self.tool.invoke({"query": "test", "freshness": "recent"})

        mock_search.assert_called_once_with(
            "test", count=5, freshness="", categories=["web"], context=ANY,
        )

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search._search_core")
    def test_truncation_pointer_to_web_fetch(self, mock_search, mock_fetch_get):
        mock_search.return_value = _search_data([
            {"type": "web", "title": "Long Page", "url": "https://example.com/long",
             "description": "Long result", "age": "", "extra_snippets": []},
        ])

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.text = "<html><body><p>" + "word " * 6000 + "</p></body></html>"  # ~30k chars
        resp.is_redirect = False
        resp.raise_for_status = MagicMock()
        mock_fetch_get.return_value = resp

        result = self.tool.invoke({"query": "test"})

        self.assertIn("Content truncated at", result)
        self.assertIn("Use web_fetch with start_index=", result)

    @patch("llm.tools.web_fetch._pinned_get")
    @patch("llm.tools.brave_search._search_core")
    def test_preserves_search_metadata(self, mock_search, mock_fetch_get):
        mock_search.return_value = _search_data()

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.text = "<html><body><p>Content</p></body></html>"
        resp.is_redirect = False
        resp.raise_for_status = MagicMock()
        mock_fetch_get.return_value = resp

        result = self.tool.invoke({"query": "test"})
        self.assertIn("Page One", result)
        self.assertIn("Page Two", result)
        self.assertIn("https://example.com/one", result)
        self.assertIn("First result", result)
