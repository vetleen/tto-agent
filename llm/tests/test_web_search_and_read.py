"""Tests for WebSearchAndReadTool."""

from unittest.mock import ANY, MagicMock, patch

from django.test import TestCase, override_settings

from llm.tools.web_search_and_read import WebSearchAndReadTool


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
    WEB_SEARCH_READ_BACKEND="fetch",
)
class WebSearchAndReadTests(TestCase):
    """The fetch backend (rollback path): Brave web search + fetch each page."""

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


# ---------------------------------------------------------------------------
# LLM Context backend
# ---------------------------------------------------------------------------

_DUMMY_CACHE = {"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}}
_LOCMEM_CACHE = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                             "LOCATION": "web-search-read-ctx-tests"}}


def _ctx_payload(generic=None, sources=None):
    """A Brave /res/v1/llm/context response body."""
    if generic is None:
        generic = [
            {"url": "https://example.com/one", "title": "Page One",
             "snippets": ["Relevant passage one.", "Another passage from one."]},
            {"url": "https://example.com/two", "title": "Page Two",
             "snippets": ["Relevant passage two."]},
        ]
    if sources is None:
        sources = {
            "https://example.com/one": {
                "title": "Page One", "hostname": "example.com",
                "age": ["Monday, March 2, 2026", "2026-03-02", "2 days ago", "2026-03-02T10:00:00Z"],
            },
            "https://example.com/two": {"title": "Page Two", "hostname": "example.com", "age": []},
        }
    return {"grounding": {"generic": generic}, "sources": sources}


def _mock_response(payload=None, status=200):
    import requests

    resp = MagicMock()
    resp.status_code = status
    resp.headers = {}
    resp.iter_content.return_value = [b"{}"]
    resp.json.return_value = payload if payload is not None else _ctx_payload()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
    else:
        resp.raise_for_status = MagicMock()
    return resp


@override_settings(
    CACHES=_DUMMY_CACHE,
    BRAVE_SEARCH_API_KEY="test-key",
    WEB_SEARCH_READ_BACKEND="llm_context",
)
class LLMContextBackendTests(TestCase):

    def setUp(self):
        self.tool = WebSearchAndReadTool()
        patcher = patch("llm.tools.brave_search._brave_rate_limiter")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _params(self, mock_get):
        return mock_get.call_args.kwargs["params"]

    # --- request ---------------------------------------------------------

    @patch("llm.tools.brave_search.requests.get")
    def test_calls_llm_context_endpoint_not_fetch(self, mock_get):
        mock_get.return_value = _mock_response()
        with patch("llm.tools.web_fetch._fetch_core") as mock_fetch, \
                patch("llm.tools.brave_search._search_core") as mock_search:
            self.tool.invoke({"query": "test query"})
        mock_fetch.assert_not_called()
        mock_search.assert_not_called()
        self.assertEqual(mock_get.call_args.args[0], "https://api.search.brave.com/res/v1/llm/context")
        self.assertEqual(mock_get.call_args.kwargs["headers"]["X-Subscription-Token"], "test-key")
        self.assertEqual(self._params(mock_get)["q"], "test query")

    @patch("llm.tools.brave_search.requests.get")
    def test_default_depth_is_standard(self, mock_get):
        mock_get.return_value = _mock_response()
        self.tool.invoke({"query": "q"})
        self.assertEqual(self._params(mock_get)["maximum_number_of_tokens"], 8192)

    @patch("llm.tools.brave_search.requests.get")
    def test_depth_budgets(self, mock_get):
        mock_get.return_value = _mock_response()
        for depth, budget in (("quick", 2048), ("standard", 8192), ("deep", 16384)):
            self.tool.invoke({"query": "q", "depth": depth})
            self.assertEqual(self._params(mock_get)["maximum_number_of_tokens"], budget, depth)

    @patch("llm.tools.brave_search.requests.get")
    def test_invalid_depth_falls_back_to_standard(self, mock_get):
        mock_get.return_value = _mock_response()
        result = self.tool.invoke({"query": "q", "depth": "exhaustive"})
        self.assertIn("Relevant passage one.", result)
        self.assertEqual(self._params(mock_get)["maximum_number_of_tokens"], 8192)

    @patch("llm.tools.brave_search.requests.get")
    def test_budget_constants_are_the_source_of_truth(self, mock_get):
        mock_get.return_value = _mock_response()
        with patch.dict("llm.tools.web_search_and_read.DEPTH_TOKEN_BUDGETS", {"quick": 1500}):
            self.tool.invoke({"query": "q", "depth": "quick"})
        self.assertEqual(self._params(mock_get)["maximum_number_of_tokens"], 1500)

    @patch("llm.tools.brave_search.requests.get")
    def test_count_maps_to_max_urls(self, mock_get):
        mock_get.return_value = _mock_response()
        self.tool.invoke({"query": "q"})
        self.assertEqual(self._params(mock_get)["maximum_number_of_urls"], 10)
        self.tool.invoke({"query": "q", "count": 3})
        self.assertEqual(self._params(mock_get)["maximum_number_of_urls"], 3)
        self.tool.invoke({"query": "q", "count": 50})
        self.assertEqual(self._params(mock_get)["maximum_number_of_urls"], 20)
        self.tool.invoke({"query": "q", "count": 0})
        self.assertEqual(self._params(mock_get)["maximum_number_of_urls"], 1)

    @patch("llm.tools.brave_search.requests.get")
    def test_freshness_forwarded_and_invalid_dropped(self, mock_get):
        mock_get.return_value = _mock_response()
        self.tool.invoke({"query": "q", "freshness": "pm"})
        self.assertEqual(self._params(mock_get)["freshness"], "pm")
        self.tool.invoke({"query": "q", "freshness": "recent"})
        self.assertNotIn("freshness", self._params(mock_get))

    def test_empty_query(self):
        self.assertIn("empty search query", self.tool.invoke({"query": "   "}))

    # --- output ----------------------------------------------------------

    @patch("llm.tools.brave_search.requests.get")
    def test_formats_sources_and_snippets(self, mock_get):
        from llm.tools._text_cleaning import EXTERNAL_CONTENT_BEGIN, EXTERNAL_CONTENT_END

        mock_get.return_value = _mock_response()
        result = self.tool.invoke({"query": "test query"})

        self.assertTrue(result.startswith(EXTERNAL_CONTENT_BEGIN))
        self.assertTrue(result.rstrip().endswith(EXTERNAL_CONTENT_END))
        self.assertIn("Search: test query", result)
        self.assertIn("Results: 2", result)
        self.assertIn("--- Source 1: Page One (2 days ago) ---", result)
        self.assertIn("--- Source 2: Page Two ---", result)
        self.assertIn("URL: https://example.com/one", result)
        self.assertIn("Relevant passage one.\n\nAnother passage from one.", result)
        self.assertIn("Relevant passage two.", result)
        self.assertIn("Use web_fetch(url) to read a page in full.", result)

    @patch("llm.tools.brave_search.requests.get")
    def test_structured_snippet_kept_as_json_and_text_cleaned(self, mock_get):
        mock_get.return_value = _mock_response(_ctx_payload(
            generic=[{"url": "https://example.com/t", "title": "Caf&eacute; <strong>table</strong>",
                      "snippets": [{"col": ["a", "b"]}, "zero​width", "   "]}],
            sources={},
        ))
        result = self.tool.invoke({"query": "q"})
        self.assertIn('{"col": ["a", "b"]}', result)
        self.assertIn("zerowidth", result)
        self.assertIn("Café **table**", result)

    @patch("llm.tools.brave_search.requests.get")
    def test_sources_without_snippets_skipped(self, mock_get):
        mock_get.return_value = _mock_response(_ctx_payload(
            generic=[{"url": "https://example.com/empty", "title": "Empty", "snippets": []},
                     {"url": "https://example.com/ok", "title": "Ok", "snippets": ["text"]}],
            sources={},
        ))
        result = self.tool.invoke({"query": "q"})
        self.assertIn("Results: 1", result)
        self.assertNotIn("Empty", result)

    @patch("llm.tools.brave_search.requests.get")
    def test_empty_grounding(self, mock_get):
        mock_get.return_value = _mock_response({"grounding": {"generic": []}, "sources": {}})
        self.assertEqual(self.tool.invoke({"query": "q"}), "No relevant content found.")

    # --- errors ----------------------------------------------------------

    @patch("llm.tools.brave_search.requests.get")
    def test_client_error_not_retried(self, mock_get):
        mock_get.return_value = _mock_response({"error": {"detail": "bad param"}}, status=422)
        result = self.tool.invoke({"query": "q"})
        self.assertTrue(result.startswith("Search error: Brave Search API error 422: bad param"))
        mock_get.assert_called_once()

    @patch("llm.tools.brave_search.time.sleep")
    @patch("llm.tools.brave_search.requests.get")
    def test_rate_limited_after_retries(self, mock_get, _sleep):
        mock_get.return_value = _mock_response({}, status=429)
        result = self.tool.invoke({"query": "q"})
        self.assertIn("Search error: Brave Search rate limited after retries", result)
        self.assertGreater(mock_get.call_count, 1)

    @patch("llm.tools.brave_search.time.sleep")
    @patch("llm.tools.brave_search.requests.get")
    def test_timeout_then_success(self, mock_get, _sleep):
        import requests

        mock_get.side_effect = [requests.exceptions.Timeout(), _mock_response()]
        result = self.tool.invoke({"query": "q"})
        self.assertIn("Relevant passage one.", result)
        self.assertEqual(mock_get.call_count, 2)

    # --- cache + guardrail ----------------------------------------------

    @override_settings(CACHES=_LOCMEM_CACHE)
    @patch("llm.tools.brave_search.requests.get")
    def test_cache_hit_skips_request(self, mock_get):
        from django.core.cache import cache

        cache.clear()
        self.addCleanup(cache.clear)
        mock_get.return_value = _mock_response()
        first = self.tool.invoke({"query": "cached q"})
        second = self.tool.invoke({"query": "cached q"})
        self.assertEqual(first, second)
        mock_get.assert_called_once()
        # A different depth is a different request.
        self.tool.invoke({"query": "cached q", "depth": "deep"})
        self.assertEqual(mock_get.call_count, 2)

    @override_settings(CACHES=_LOCMEM_CACHE)
    @patch("llm.tools.brave_search.requests.get")
    def test_errors_not_cached(self, mock_get):
        from django.core.cache import cache

        cache.clear()
        self.addCleanup(cache.clear)
        mock_get.return_value = _mock_response({"error": {"detail": "x"}}, status=400)
        self.tool.invoke({"query": "err q"})
        mock_get.return_value = _mock_response()
        result = self.tool.invoke({"query": "err q"})
        self.assertIn("Relevant passage one.", result)

    @patch("guardrails.web_content.scan_web_content_from_tool")
    @patch("llm.tools.brave_search.requests.get")
    def test_guardrail_scan_labelled(self, mock_get, mock_scan):
        mock_get.return_value = _mock_response()
        self.tool.invoke({"query": "q"})
        mock_scan.assert_called_once()
        text = mock_scan.call_args.args[0]
        self.assertIn("Relevant passage one.", text)
        self.assertEqual(mock_scan.call_args.kwargs["source_label"], "brave_llm_context")


class BackendSelectionTests(TestCase):

    @override_settings(WEB_SEARCH_READ_BACKEND="llm_context")
    def test_llm_context_description_and_timeout(self):
        tool = WebSearchAndReadTool()
        self.assertIn("web_fetch(url)", tool.description)
        self.assertEqual(tool.timeout_seconds, 45)

    @override_settings(WEB_SEARCH_READ_BACKEND="fetch")
    def test_fetch_description_and_timeout(self):
        tool = WebSearchAndReadTool()
        self.assertIn("full content of each result page", tool.description)
        self.assertEqual(tool.timeout_seconds, 80)

    @override_settings(WEB_SEARCH_READ_BACKEND="nonsense")
    def test_unknown_backend_means_llm_context(self):
        from llm.tools.web_search_and_read import _backend

        self.assertEqual(_backend(), "llm_context")

    def test_schema_advertises_depth_enum(self):
        from llm.tools.web_search_and_read import WebSearchAndReadInput

        prop = WebSearchAndReadInput.model_json_schema()["properties"]["depth"]
        self.assertEqual(prop["enum"], ["quick", "standard", "deep"])
        self.assertEqual(prop["default"], "standard")
