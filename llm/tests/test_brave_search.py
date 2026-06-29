"""Tests for BraveSearchTool."""

import time
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from llm.tools.brave_search import (
    _RATE_LIMIT_BACKOFF_SCHEDULE,
    BraveSearchTool,
    _normalize_categories,
    _parse_rate_limit_reset,
    _parse_retry_after,
    _TokenBucketRateLimiter,
    _validate_freshness,
)


def _mock_ok(payload=None):
    """Build a mocked 200 response with the given JSON payload."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Shape needed by the size-cap guard (_enforce_size_and_buffer): a real
    # headers dict (no Content-Length) and a small streamed body.
    mock_response.headers = {}
    mock_response.iter_content.return_value = [b"{}"]
    mock_response.json.return_value = (
        payload if payload is not None else {"web": {"results": []}}
    )
    mock_response.raise_for_status = MagicMock()
    return mock_response


@override_settings(CACHES={"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}})
class BraveSearchToolTests(TestCase):

    def setUp(self):
        self.tool = BraveSearchTool()
        # Patch the module-level rate limiter so tests don't block.
        patcher = patch("llm.tools.brave_search._brave_rate_limiter")
        self.mock_limiter = patcher.start()
        self.addCleanup(patcher.stop)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_successful_search(self, mock_get):
        mock_get.return_value = _mock_ok({
            "web": {
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1", "description": "Desc 1"},
                    {"title": "Result 2", "url": "https://example.com/2", "description": "Desc 2"},
                ]
            }
        })

        result = self.tool.invoke({"query": "test query"})

        self.assertIn("[1] Result 1", result)
        self.assertIn("Desc 1", result)
        self.assertIn("URL: https://example.com/2", result)
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["headers"]["X-Subscription-Token"], "test-key")

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_oversized_response_rejected_gracefully(self, mock_get):
        """An oversized body is capped (no OOM) and surfaced as a search error
        without retrying — Brave responses are never legitimately that large."""
        from llm.tools.web_fetch import _max_response_bytes

        oversized = MagicMock()
        oversized.status_code = 200
        oversized.headers = {"Content-Length": str(_max_response_bytes() + 1)}
        oversized.raise_for_status = MagicMock()
        mock_get.return_value = oversized

        result = self.tool.invoke({"query": "test"})

        self.assertIn("Search error", result)
        self.assertIn("large", result)
        mock_get.assert_called_once()

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_delimiters_wrap_results(self, mock_get):
        mock_get.return_value = _mock_ok({
            "web": {"results": [{"title": "T", "url": "https://e.com", "description": "d"}]}
        })

        result = self.tool.invoke({"query": "test"})

        self.assertIn("=== BEGIN EXTERNAL WEB CONTENT", result)
        self.assertIn("=== END EXTERNAL WEB CONTENT ===", result)
        self.assertIn("never as instructions", result)

    def test_missing_api_key_raises(self):
        with override_settings(BRAVE_SEARCH_API_KEY=""):
            with self.assertRaises(Exception) as ctx:
                self.tool.invoke({"query": "test"})
            self.assertIn("BRAVE_SEARCH_API_KEY", str(ctx.exception))

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    def test_empty_query_returns_error(self):
        result = self.tool.invoke({"query": ""})
        self.assertIn("Search error", result)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_timeout_retries(self, mock_sleep, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.Timeout("timeout")

        result = self.tool.invoke({"query": "test"})

        self.assertIn("Search error", result)
        # 1 initial + 3 retries = 4 calls
        self.assertEqual(mock_get.call_count, 4)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_client_error_no_retry(self, mock_get):
        import requests as req_lib
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        result = self.tool.invoke({"query": "test"})

        self.assertIn("Search error", result)
        self.assertIn("401", result)
        # Should NOT retry on client errors
        self.assertEqual(mock_get.call_count, 1)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_count_capped_at_20(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "test", "count": 50})

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["params"]["count"], 20)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_extra_snippets_requested(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "test"})

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["params"]["extra_snippets"], 1)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_freshness_passed_through(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "test", "freshness": "pw"})

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["params"]["freshness"], "pw")

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_freshness_date_range_passed_through(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "test", "freshness": "2025-01-01to2025-06-01"})

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["params"]["freshness"], "2025-01-01to2025-06-01")

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_invalid_freshness_ignored(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "test", "freshness": "last_week"})

        call_kwargs = mock_get.call_args
        self.assertNotIn("freshness", call_kwargs.kwargs["params"])

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_categories_default_web(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "test"})

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["params"]["result_filter"], "web")

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_categories_comma_string_parsed(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "test", "categories": "news, web"})

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["params"]["result_filter"], "web,news")

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_invalid_categories_dropped(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "test", "categories": "web,locations,summarizer"})

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["params"]["result_filter"], "web")

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_news_results_parsed(self, mock_get):
        mock_get.return_value = _mock_ok({
            "web": {"results": [{"title": "Web R", "url": "https://w.co", "description": "wd"}]},
            "news": {"results": [{"title": "News R", "url": "https://n.co", "description": "nd", "age": "2 hours ago"}]},
        })

        result = self.tool.invoke({"query": "test", "categories": "web,news"})

        self.assertIn("Web R", result)
        self.assertIn("News R", result)
        self.assertIn("(2 hours ago)", result)
        self.assertIn("[news]", result)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_faq_results_parsed(self, mock_get):
        mock_get.return_value = _mock_ok({
            "faq": {"results": [{"question": "What is X?", "answer": "X is Y.", "url": "https://f.co"}]},
        })

        result = self.tool.invoke({"query": "test", "categories": "faq"})

        self.assertIn("What is X?", result)
        self.assertIn("X is Y.", result)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_strong_tags_converted_to_markdown(self, mock_get):
        mock_get.return_value = _mock_ok({
            "web": {
                "results": [
                    {
                        "title": "The <strong>best</strong> tool",
                        "url": "https://e.com",
                        "description": "It is <strong>very good</strong> indeed.",
                    }
                ]
            }
        })

        result = self.tool.invoke({"query": "test"})

        self.assertIn("The **best** tool", result)
        self.assertIn("It is **very good** indeed.", result)
        self.assertNotIn("<strong>", result)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_html_entities_unescaped(self, mock_get):
        mock_get.return_value = _mock_ok({
            "web": {
                "results": [
                    {
                        "title": "Ohio&#x27;s universities",
                        "url": "https://e.com",
                        "description": "The &quot;Bayh-Dole&quot; Act &amp; licensing.",
                    }
                ]
            }
        })

        result = self.tool.invoke({"query": "test"})

        self.assertIn("Ohio's universities", result)
        self.assertIn('The "Bayh-Dole" Act & licensing.', result)
        self.assertNotIn("&#x27;", result)
        self.assertNotIn("&quot;", result)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_zero_width_entity_still_stripped(self, mock_get):
        """A zero-width char smuggled in as an HTML entity must still be removed."""
        mock_get.return_value = _mock_ok({
            "web": {
                "results": [
                    {"title": "Re&#8203;sult", "url": "https://e.com", "description": "d"},
                ]
            }
        })

        result = self.tool.invoke({"query": "test"})

        self.assertIn("Result", result)
        self.assertNotIn("​", result)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_extra_snippets_and_age_in_output(self, mock_get):
        mock_get.return_value = _mock_ok({
            "web": {
                "results": [
                    {
                        "title": "T",
                        "url": "https://e.com",
                        "description": "d",
                        "age": "3 days ago",
                        "extra_snippets": ["First extra snippet.", "Second extra snippet."],
                    }
                ]
            }
        })

        result = self.tool.invoke({"query": "test"})

        self.assertIn("(3 days ago)", result)
        self.assertIn("> First extra snippet.", result)
        self.assertIn("> Second extra snippet.", result)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_api_version_header_sent_when_set(self, mock_get):
        mock_get.return_value = _mock_ok()

        with patch.dict("os.environ", {"BRAVE_API_VERSION": "2025-01-01"}):
            self.tool.invoke({"query": "test"})

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["headers"]["Api-Version"], "2025-01-01")

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_api_version_header_absent_by_default(self, mock_get):
        mock_get.return_value = _mock_ok()

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("BRAVE_API_VERSION", None)
            self.tool.invoke({"query": "test"})

        call_kwargs = mock_get.call_args
        self.assertNotIn("Api-Version", call_kwargs.kwargs["headers"])

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_rate_limit_429_retries(self, mock_sleep, mock_get):
        """429 is retried with rate-limit backoff, not treated as a client error."""
        import requests as req_lib

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {}
        mock_429.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_429)

        mock_ok = _mock_ok({"web": {"results": [{"title": "OK", "url": "https://example.com", "description": "d"}]}})

        mock_get.side_effect = [mock_429, mock_ok]

        result = self.tool.invoke({"query": "test"})

        self.assertIn("OK", result)
        self.assertEqual(mock_get.call_count, 2)
        # Rate-limit backoff schedule[0] = 5.0
        mock_sleep.assert_called_once_with(5.0)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_rate_limit_429_exhausted(self, mock_sleep, mock_get):
        """All attempts return 429 → descriptive rate-limit error message."""
        import requests as req_lib

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {}
        mock_429.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_429)
        mock_get.return_value = mock_429

        result = self.tool.invoke({"query": "test"})

        self.assertIn("rate limited", result)
        # 1 initial + 3 retries = 4 calls
        self.assertEqual(mock_get.call_count, 4)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_rate_limit_respects_x_ratelimit_reset(self, mock_sleep, mock_get):
        """X-RateLimit-Reset: 7 header → sleeps 7s."""
        import requests as req_lib

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"X-RateLimit-Reset": "7"}
        mock_429.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_429)

        mock_get.side_effect = [mock_429, _mock_ok()]

        self.tool.invoke({"query": "test"})

        mock_sleep.assert_called_once_with(7.0)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_x_ratelimit_reset_preferred_over_retry_after(self, mock_sleep, mock_get):
        """When both headers are present, X-RateLimit-Reset wins."""
        import requests as req_lib

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"X-RateLimit-Reset": "7", "Retry-After": "3"}
        mock_429.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_429)

        mock_get.side_effect = [mock_429, _mock_ok()]

        self.tool.invoke({"query": "test"})

        mock_sleep.assert_called_once_with(7.0)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_rate_limit_respects_retry_after(self, mock_sleep, mock_get):
        """Legacy Retry-After: 3 header → sleeps 3s when X-RateLimit-Reset absent."""
        import requests as req_lib

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"Retry-After": "3"}
        mock_429.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_429)

        mock_get.side_effect = [mock_429, _mock_ok()]

        self.tool.invoke({"query": "test"})

        mock_sleep.assert_called_once_with(3.0)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_rate_limit_header_capped(self, mock_sleep, mock_get):
        """X-RateLimit-Reset: 120 → capped at schedule max (60s)."""
        import requests as req_lib

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"X-RateLimit-Reset": "120"}
        mock_429.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_429)

        mock_get.side_effect = [mock_429, _mock_ok()]

        self.tool.invoke({"query": "test"})

        mock_sleep.assert_called_once_with(60.0)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_403_no_retry(self, mock_get):
        """Regression guard: 403 still fails immediately without retry."""
        import requests as req_lib

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        result = self.tool.invoke({"query": "test"})

        self.assertIn("403", result)
        self.assertEqual(mock_get.call_count, 1)

    def test_parse_retry_after_invalid(self):
        """Non-numeric Retry-After header returns None."""
        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "not-a-number"}
        self.assertIsNone(_parse_retry_after(mock_response))

    def test_parse_retry_after_missing(self):
        """Missing Retry-After header returns None."""
        mock_response = MagicMock()
        mock_response.headers = {}
        self.assertIsNone(_parse_retry_after(mock_response))

    def test_parse_rate_limit_reset_valid(self):
        mock_response = MagicMock()
        mock_response.headers = {"X-RateLimit-Reset": "12"}
        self.assertEqual(_parse_rate_limit_reset(mock_response), 12.0)

    def test_parse_rate_limit_reset_invalid(self):
        mock_response = MagicMock()
        mock_response.headers = {"X-RateLimit-Reset": "soon"}
        self.assertIsNone(_parse_rate_limit_reset(mock_response))

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_rate_limiter_acquire_called(self, mock_get):
        """acquire() is called before each request attempt."""
        import requests as req_lib

        mock_timeout = req_lib.exceptions.Timeout("timeout")

        # First call times out, second succeeds
        mock_get.side_effect = [mock_timeout, _mock_ok()]

        with patch("llm.tools.brave_search.time.sleep"):
            self.tool.invoke({"query": "test"})

        # acquire() should have been called once per attempt
        self.assertEqual(self.mock_limiter.acquire.call_count, 2)

    def test_backoff_schedule_values(self):
        """Regression guard: backoff schedule matches expected values."""
        self.assertEqual(_RATE_LIMIT_BACKOFF_SCHEDULE, [5.0, 15.0, 30.0, 60.0])

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("guardrails.web_content.scan_web_content")
    def test_scan_web_content_called_with_results(self, mock_scan, mock_get):
        """scan_web_content should be called with the combined result text."""
        mock_get.return_value = _mock_ok({
            "web": {
                "results": [
                    {"title": "Title 1", "url": "https://example.com/1", "description": "Desc 1"},
                    {"title": "Title 2", "url": "https://example.com/2", "description": "Desc 2"},
                ]
            }
        })

        from llm.types.context import RunContext
        ctx = RunContext.create(user_id=42, conversation_id="thread-abc")
        self.tool.set_context(ctx)
        self.tool.invoke({"query": "test query"})

        mock_scan.assert_called_once()
        call_kwargs = mock_scan.call_args
        self.assertIn("Title 1", call_kwargs[0][0])
        self.assertIn("Desc 2", call_kwargs[0][0])
        self.assertEqual(call_kwargs[1]["user_id"], "42")
        self.assertEqual(call_kwargs[1]["thread_id"], "thread-abc")
        self.assertEqual(call_kwargs[1]["source_label"], "brave_search")

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("guardrails.web_content.scan_web_content", side_effect=RuntimeError("scan boom"))
    def test_scan_web_content_error_does_not_break_tool(self, mock_scan, mock_get):
        """If scan_web_content raises, the tool should still return valid results."""
        mock_get.return_value = _mock_ok({
            "web": {"results": [{"title": "OK", "url": "https://example.com", "description": "Fine"}]}
        })

        result = self.tool.invoke({"query": "test"})
        self.assertIn("OK", result)


class FreshnessValidationTests(TestCase):
    """Unit tests for _validate_freshness."""

    def test_presets_valid(self):
        for v in ("pd", "pw", "pm", "py"):
            self.assertEqual(_validate_freshness(v), v)

    def test_date_range_valid(self):
        self.assertEqual(
            _validate_freshness("2025-01-01to2025-06-01"),
            "2025-01-01to2025-06-01",
        )

    def test_invalid_values_ignored(self):
        for v in ("px", "yesterday", "2025-01-01", "2025-01-01to", "p"):
            self.assertEqual(_validate_freshness(v), "")

    def test_empty_and_none(self):
        self.assertEqual(_validate_freshness(""), "")
        self.assertEqual(_validate_freshness(None), "")


class CategoryNormalizationTests(TestCase):
    """Unit tests for _normalize_categories."""

    def test_default_web(self):
        self.assertEqual(_normalize_categories(None), ["web"])
        self.assertEqual(_normalize_categories(""), ["web"])

    def test_comma_string(self):
        self.assertEqual(_normalize_categories("news, web"), ["web", "news"])

    def test_invalid_dropped(self):
        self.assertEqual(_normalize_categories("web,bogus"), ["web"])

    def test_all_invalid_falls_back_to_web(self):
        self.assertEqual(_normalize_categories("bogus,locations"), ["web"])

    def test_list_input(self):
        self.assertEqual(_normalize_categories(["faq", "news"]), ["news", "faq"])

    def test_case_insensitive(self):
        self.assertEqual(_normalize_categories("NEWS"), ["news"])


class BraveSearchCacheTests(TestCase):
    """Tests for BraveSearchTool caching."""

    def setUp(self):
        from django.core.cache import cache
        cache.clear()
        self.tool = BraveSearchTool()
        patcher = patch("llm.tools.brave_search._brave_rate_limiter")
        self.mock_limiter = patcher.start()
        self.addCleanup(patcher.stop)

    @override_settings(
        BRAVE_SEARCH_API_KEY="test-key",
        CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    )
    @patch("llm.tools.brave_search.requests.get")
    def test_cache_miss_calls_api_and_caches(self, mock_get):
        mock_get.return_value = _mock_ok(
            {"web": {"results": [{"title": "CachedResult", "url": "https://r.co", "description": "d"}]}}
        )

        result = self.tool.invoke({"query": "cache me"})

        self.assertIn("CachedResult", result)
        mock_get.assert_called_once()

        # Second call should use cache
        mock_get.reset_mock()
        result2 = self.tool.invoke({"query": "cache me"})
        self.assertIn("CachedResult", result2)
        mock_get.assert_not_called()

    @override_settings(
        BRAVE_SEARCH_API_KEY="test-key",
        CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    )
    @patch("llm.tools.brave_search.requests.get")
    def test_cache_key_varies_with_freshness(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "freshness query"})
        self.tool.invoke({"query": "freshness query", "freshness": "pw"})

        # Different freshness → different cache key → two API calls
        self.assertEqual(mock_get.call_count, 2)

    @override_settings(
        BRAVE_SEARCH_API_KEY="test-key",
        CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    )
    @patch("llm.tools.brave_search.requests.get")
    def test_cache_key_varies_with_categories(self, mock_get):
        mock_get.return_value = _mock_ok()

        self.tool.invoke({"query": "categories query"})
        self.tool.invoke({"query": "categories query", "categories": "web,news"})

        self.assertEqual(mock_get.call_count, 2)

    @override_settings(
        BRAVE_SEARCH_API_KEY="test-key",
        CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    )
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_error_response_not_cached(self, mock_sleep, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.Timeout("timeout")

        result = self.tool.invoke({"query": "fail query"})
        self.assertIn("Search error", result)

        # Reset and make a successful call — should hit API, not cache
        mock_get.side_effect = None
        mock_get.return_value = _mock_ok()

        result2 = self.tool.invoke({"query": "fail query"})
        self.assertNotIn("Search error", result2)
        mock_get.assert_called()

    @override_settings(
        BRAVE_SEARCH_API_KEY="test-key",
        CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    )
    @patch("llm.tools.brave_search.requests.get")
    def test_cache_connection_error_falls_through(self, mock_get):
        mock_get.return_value = _mock_ok(
            {"web": {"results": [{"title": "RedisDownResult", "url": "https://r.co", "description": "d"}]}}
        )

        with patch("django.core.cache.cache.get", side_effect=ConnectionError("Redis SSL EOF")), \
             patch("django.core.cache.cache.set", side_effect=ConnectionError("Redis SSL EOF")):
            result = self.tool.invoke({"query": "redis down"})

        self.assertIn("RedisDownResult", result)
        mock_get.assert_called_once()


class TokenBucketRateLimiterTests(TestCase):
    """Unit tests for _TokenBucketRateLimiter."""

    def test_first_acquire_immediate(self):
        """First acquire should return immediately (burst=1 gives one token)."""
        limiter = _TokenBucketRateLimiter(requests_per_second=1.0, burst=1)
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.1)

    def test_second_acquire_waits(self):
        """Second acquire must wait ~1/rps seconds."""
        limiter = _TokenBucketRateLimiter(requests_per_second=10.0, burst=1)
        limiter.acquire()  # consume the burst token
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        # Should wait ~0.1s (1/10 rps)
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertLess(elapsed, 0.5)

    def test_burst_allows_multiple_immediate(self):
        """With burst=3, three acquires should be immediate."""
        limiter = _TokenBucketRateLimiter(requests_per_second=1.0, burst=3)
        start = time.monotonic()
        for _ in range(3):
            limiter.acquire()
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.1)


@override_settings(BRAVE_SEARCH_API_KEY="test-key")
@override_settings(CACHES={"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}})
class BraveSearchNormalizationTests(TestCase):
    """Tests for text normalization on search results."""

    def setUp(self):
        self.tool = BraveSearchTool()
        patcher = patch("llm.tools.brave_search._brave_rate_limiter")
        self.mock_limiter = patcher.start()
        self.addCleanup(patcher.stop)

    @patch("llm.tools.brave_search.requests.get")
    def test_descriptions_have_zero_width_chars_stripped(self, mock_get):
        """Zero-width characters in titles/descriptions should be removed."""
        mock_get.return_value = _mock_ok({
            "web": {
                "results": [
                    {
                        "title": "Re​sult",
                        "url": "https://example.com",
                        "description": "De‌sc‍ription",
                    },
                ]
            }
        })

        result = self.tool.invoke({"query": "test"})

        self.assertIn("Result", result)
        self.assertIn("Description", result)
        self.assertNotIn("​", result)
        self.assertNotIn("‌", result)
