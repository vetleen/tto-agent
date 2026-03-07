"""Tests for BraveSearchTool."""

import json
from unittest.mock import MagicMock, patch

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
