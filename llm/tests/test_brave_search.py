"""Tests for BraveSearchTool."""

import json
from unittest.mock import MagicMock, call, patch

from django.test import TestCase, override_settings

from llm.tools.brave_search import BraveSearchTool


class BraveSearchToolTests(TestCase):

    def setUp(self):
        self.tool = BraveSearchTool()

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
        # 1 initial + 2 retries = 3 calls
        self.assertEqual(mock_get.call_count, 3)

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
        # Rate-limit backoff: 2.0 * (2 ** 0) = 2.0
        mock_sleep.assert_called_once_with(2.0)

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
        self.assertEqual(mock_get.call_count, 3)

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
        """Retry-After: 60 → capped at _RATE_LIMIT_MAX_WAIT (10s)."""
        import requests as req_lib

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"Retry-After": "60"}
        mock_429.raise_for_status.side_effect = req_lib.exceptions.HTTPError(response=mock_429)

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"web": {"results": []}}
        mock_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_429, mock_ok]

        self.tool.invoke({"query": "test"})

        mock_sleep.assert_called_once_with(10.0)

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
