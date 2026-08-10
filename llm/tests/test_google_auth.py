"""Tests for the Gemini Vertex AI credentials helper (llm/core/google_auth.py)."""

import json
from unittest.mock import patch

from django.test import SimpleTestCase

from llm.core.google_auth import clear_credentials_cache, get_vertex_config
from llm.service.errors import LLMConfigurationError

# A structurally-valid-looking service account JSON. from_service_account_info is
# patched in every test, so the values (and the fake key) are never parsed by
# real crypto.
_FAKE_SA = json.dumps(
    {
        "type": "service_account",
        "project_id": "sa-embedded-project",
        "private_key_id": "abc",
        "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----\n",
        "client_email": "svc@example.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
)


class _FakeCreds:
    """Stand-in for service_account.Credentials with a project_id attribute."""

    def __init__(self, project_id="sa-embedded-project"):
        self.project_id = project_id


class GetVertexConfigTests(SimpleTestCase):
    def setUp(self):
        clear_credentials_cache()

    def tearDown(self):
        clear_credentials_cache()

    @patch.dict("os.environ", {}, clear=False)
    def test_returns_none_without_service_account(self):
        import os

        os.environ.pop("GOOGLE_SERVICE_ACCOUNT_JSON", None)
        self.assertIsNone(get_vertex_config())

    @patch.dict("os.environ", {"GOOGLE_SERVICE_ACCOUNT_JSON": "   "}, clear=False)
    def test_blank_service_account_is_treated_as_unset(self):
        self.assertIsNone(get_vertex_config())

    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    @patch.dict(
        "os.environ",
        {
            "GOOGLE_SERVICE_ACCOUNT_JSON": _FAKE_SA,
            "GOOGLE_CLOUD_PROJECT": "wilfred-505110",
            "GOOGLE_CLOUD_LOCATION": "eu",
        },
        clear=False,
    )
    def test_returns_vertex_config_when_configured(self, mock_from_info):
        creds = _FakeCreds()
        mock_from_info.return_value = creds

        config = get_vertex_config()
        self.assertEqual(config["vertexai"], True)
        self.assertEqual(config["project"], "wilfred-505110")
        self.assertEqual(config["location"], "eu")
        self.assertIs(config["credentials"], creds)

        # Scoped to cloud-platform.
        _, kwargs = mock_from_info.call_args
        self.assertEqual(
            kwargs["scopes"], ["https://www.googleapis.com/auth/cloud-platform"]
        )

    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    @patch.dict(
        "os.environ",
        {"GOOGLE_SERVICE_ACCOUNT_JSON": _FAKE_SA, "GOOGLE_CLOUD_PROJECT": "p"},
        clear=False,
    )
    def test_location_defaults_to_global(self, mock_from_info):
        import os

        os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
        mock_from_info.return_value = _FakeCreds()
        self.assertEqual(get_vertex_config()["location"], "global")

    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    @patch.dict(
        "os.environ", {"GOOGLE_SERVICE_ACCOUNT_JSON": _FAKE_SA}, clear=False
    )
    def test_project_falls_back_to_credentials_project_id(self, mock_from_info):
        import os

        os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        mock_from_info.return_value = _FakeCreds(project_id="sa-embedded-project")
        self.assertEqual(get_vertex_config()["project"], "sa-embedded-project")

    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_service_account_from_file_path(self, mock_from_file):
        """Locally the env var is a path to the key file, not inline JSON."""
        import os
        import tempfile

        mock_from_file.return_value = _FakeCreds(project_id="wilfred-505110")
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with patch.dict(
                "os.environ",
                {"GOOGLE_SERVICE_ACCOUNT_JSON": path, "GOOGLE_CLOUD_PROJECT": "p"},
                clear=False,
            ):
                config = get_vertex_config()
            self.assertEqual(config["vertexai"], True)
            self.assertIs(config["credentials"], mock_from_file.return_value)
            mock_from_file.assert_called_once()
            _, kwargs = mock_from_file.call_args
            self.assertEqual(
                kwargs["scopes"], ["https://www.googleapis.com/auth/cloud-platform"]
            )
        finally:
            os.remove(path)

    @patch.dict(
        "os.environ",
        {"GOOGLE_SERVICE_ACCOUNT_JSON": "{not valid json"},
        clear=False,
    )
    def test_malformed_json_raises_configuration_error(self):
        with self.assertRaises(LLMConfigurationError):
            get_vertex_config()

    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    @patch.dict(
        "os.environ",
        {"GOOGLE_SERVICE_ACCOUNT_JSON": _FAKE_SA, "GOOGLE_CLOUD_PROJECT": "p"},
        clear=False,
    )
    def test_credentials_are_memoized(self, mock_from_info):
        mock_from_info.return_value = _FakeCreds()

        first = get_vertex_config()
        second = get_vertex_config()

        # Built once, same object identity (keeps the factory client cache stable).
        mock_from_info.assert_called_once()
        self.assertIs(first["credentials"], second["credentials"])
