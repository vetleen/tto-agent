"""Test utilities for the llm app."""

from __future__ import annotations

import os
import unittest


def require_test_apis(reason: str = "Set TEST_APIS=True in the environment to run live API tests."):
    """
    Decorator to skip a test unless TEST_APIS is set to True (case-insensitive).

    Use for tests that call real provider APIs (OpenAI, Anthropic, Gemini).
    """
    test_apis = os.environ.get("TEST_APIS", "").strip().lower() == "true"
    return unittest.skipUnless(test_apis, reason)
