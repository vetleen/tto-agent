"""Tests for BraveSearchTool."""

import json
import time
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from llm.tools.brave_search import BraveSearchTool, _TokenBucketRateLimiter


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
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1", "description": "Desc 1"},
                    {"title": "Result 2", "url": "https://example.com/2", "description": "Desc 2"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = json.loads(self.tool.invoke({"query": "test query"}))

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["results"][0]["title"], "Result 1")
        self.assertEqual(result["results"][1]["url"], "https://example.com/2")
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["headers"]["X-Subscription-Token"], "test-key")

    def test_missing_api_key_raises(self):
        with override_settings(BRAVE_SEARCH_API_KEY=""):
            with self.assertRaises(Exception) as ctx:
                self.tool.invoke({"query": "test"})
            self.assertIn("BRAVE_SEARCH_API_KEY", str(ctx.exception))

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    def test_empty_query_returns_error(self):
        result = json.loads(self.tool.invoke({"query": ""}))
        self.assertIn("error", result)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_timeout_retries(self, mock_sleep, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.Timeout("timeout")

        result = json.loads(self.tool.invoke({"query": "test"}))

        self.assertIn("error", result)
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

        result = json.loads(self.tool.invoke({"query": "test"}))

        self.assertIn("error", result)
        self.assertIn("401", result["error"])
        # Should NOT retry on client errors
        self.assertEqual(mock_get.call_count, 1)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_count_capped_at_10(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        self.tool.invoke({"query": "test", "count": 50})

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs["params"]["count"], 10)

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

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"web": {"results": [{"title": "OK", "url": "https://example.com", "description": "d"}]}}
        mock_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_429, mock_ok]

        result = json.loads(self.tool.invoke({"query": "test"}))

        self.assertEqual(result["count"], 1)
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

        result = json.loads(self.tool.invoke({"query": "test"}))

        self.assertIn("rate limited", result["error"])
        # 1 initial + 3 retries = 4 calls
        self.assertEqual(mock_get.call_count, 4)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_rate_limit_respects_retry_after(self, mock_sleep, mock_get):
        """Retry-After: 3 header → sleeps 3s."""
        import requests as req_lib

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"Retry-After": "3"}
        mock_429.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_429)

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"web": {"results": []}}
        mock_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_429, mock_ok]

        self.tool.invoke({"query": "test"})

        mock_sleep.assert_called_once_with(3.0)

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_rate_limit_retry_after_capped(self, mock_sleep, mock_get):
        """Retry-After: 120 → capped at schedule max (60s)."""
        import requests as req_lib

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"Retry-After": "120"}
        mock_429.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_429)

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"web": {"results": []}}
        mock_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_429, mock_ok]

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

        result = json.loads(self.tool.invoke({"query": "test"}))

        self.assertIn("403", result["error"])
        self.assertEqual(mock_get.call_count, 1)

    def test_parse_retry_after_invalid(self):
        """Non-numeric Retry-After header returns None."""
        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "not-a-number"}
        self.assertIsNone(BraveSearchTool._parse_retry_after(mock_response))

    def test_parse_retry_after_missing(self):
        """Missing Retry-After header returns None."""
        mock_response = MagicMock()
        mock_response.headers = {}
        self.assertIsNone(BraveSearchTool._parse_retry_after(mock_response))

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    def test_rate_limiter_acquire_called(self, mock_get):
        """acquire() is called before each request attempt."""
        import requests as req_lib

        mock_timeout = req_lib.exceptions.Timeout("timeout")
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"web": {"results": []}}
        mock_ok.raise_for_status = MagicMock()

        # First call times out, second succeeds
        mock_get.side_effect = [mock_timeout, mock_ok]

        with patch("llm.tools.brave_search.time.sleep"):
            self.tool.invoke({"query": "test"})

        # acquire() should have been called once per attempt
        self.assertEqual(self.mock_limiter.acquire.call_count, 2)

    def test_backoff_schedule_values(self):
        """Regression guard: backoff schedule matches expected values."""
        self.assertEqual(
            self.tool._RATE_LIMIT_BACKOFF_SCHEDULE,
            [5.0, 15.0, 30.0, 60.0],
        )

    @override_settings(BRAVE_SEARCH_API_KEY="test-key")
    @patch("llm.tools.brave_search.requests.get")
    @patch("guardrails.web_content.scan_web_content")
    def test_scan_web_content_called_with_results(self, mock_scan, mock_get):
        """scan_web_content should be called with the combined result text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Title 1", "url": "https://example.com/1", "description": "Desc 1"},
                    {"title": "Title 2", "url": "https://example.com/2", "description": "Desc 2"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

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
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "OK", "url": "https://example.com", "description": "Fine"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = json.loads(self.tool.invoke({"query": "test"}))
        self.assertEqual(result["count"], 1)


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
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"web": {"results": [{"title": "R", "url": "https://r.co", "description": "d"}]}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = json.loads(self.tool.invoke({"query": "test"}))

        self.assertEqual(result["count"], 1)
        mock_get.assert_called_once()

        # Second call should use cache
        mock_get.reset_mock()
        result2 = json.loads(self.tool.invoke({"query": "test"}))
        self.assertEqual(result2["count"], 1)
        mock_get.assert_not_called()

    @override_settings(
        BRAVE_SEARCH_API_KEY="test-key",
        CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    )
    @patch("llm.tools.brave_search.requests.get")
    @patch("llm.tools.brave_search.time.sleep")
    def test_error_response_not_cached(self, mock_sleep, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.Timeout("timeout")

        result = json.loads(self.tool.invoke({"query": "fail query"}))
        self.assertIn("error", result)

        # Reset and make a successful call — should hit API, not cache
        mock_get.side_effect = None
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result2 = json.loads(self.tool.invoke({"query": "fail query"}))
        self.assertNotIn("error", result2)
        mock_get.assert_called()


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
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Re\u200bsult",
                        "url": "https://example.com",
                        "description": "De\u200csc\u200dription",
                    },
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = json.loads(self.tool.invoke({"query": "test"}))

        self.assertEqual(result["results"][0]["title"], "Result")
        self.assertEqual(result["results"][0]["description"], "Description")
