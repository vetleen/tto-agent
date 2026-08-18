"""Tests for config.settings helpers."""

import os
import warnings
from unittest.mock import patch

from django.test import SimpleTestCase

from config.settings import _env_int


class EnvIntTests(SimpleTestCase):
    """_env_int must never crash the app on a missing/empty/garbage env var —
    Heroku's config:set can silently write an empty string on this setup, and a
    bare int("") would raise at settings import and take the whole app down."""

    def test_missing_uses_default(self):
        self.assertEqual(_env_int("DEFINITELY_NOT_A_REAL_ENV_VAR_9x", 42), 42)

    def test_valid_value_is_parsed(self):
        with patch.dict(os.environ, {"WF_TEST_INT": "7"}):
            self.assertEqual(_env_int("WF_TEST_INT", 42), 7)

    def test_empty_string_uses_default(self):
        with patch.dict(os.environ, {"WF_TEST_INT": ""}):
            self.assertEqual(_env_int("WF_TEST_INT", 42), 42)

    def test_whitespace_uses_default(self):
        with patch.dict(os.environ, {"WF_TEST_INT": "   "}):
            self.assertEqual(_env_int("WF_TEST_INT", 42), 42)

    def test_non_numeric_uses_default(self):
        with patch.dict(os.environ, {"WF_TEST_INT": "not-a-number"}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(_env_int("WF_TEST_INT", 42), 42)

    def test_string_default_is_coerced_to_int(self):
        self.assertEqual(_env_int("DEFINITELY_NOT_A_REAL_ENV_VAR_9x", "100"), 100)

    def test_underscore_default_is_valid(self):
        # Mirrors DOCUMENT_UPLOAD_MAX_SIZE_BYTES' "50_000_000" default.
        self.assertEqual(_env_int("DEFINITELY_NOT_A_REAL_ENV_VAR_9x", "50_000_000"), 50_000_000)
