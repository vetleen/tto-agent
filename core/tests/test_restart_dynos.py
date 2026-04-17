"""Tests for the restart_dynos management command.

The command runs only in production (via Heroku Scheduler), so tests stub
``requests.delete`` and verify the URL, headers, and error handling rather
than hitting the real Heroku API.
"""
from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase


class RestartDynosCommandTests(SimpleTestCase):
    def _fake_response(self, status_code=202, text=""):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        return resp

    @patch.dict("os.environ", {"HEROKU_API_KEY": "tok", "HEROKU_APP_NAME": "wilfred-production"})
    @patch("core.management.commands.restart_dynos.requests.delete")
    def test_default_restarts_web_formation(self, mock_delete):
        mock_delete.return_value = self._fake_response(202)
        out = StringIO()
        call_command("restart_dynos", stdout=out)

        mock_delete.assert_called_once()
        url = mock_delete.call_args[0][0]
        self.assertEqual(url, "https://api.heroku.com/apps/wilfred-production/dynos/web")
        headers = mock_delete.call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer tok")
        self.assertIn("vnd.heroku+json", headers["Accept"])
        self.assertIn("type=web", out.getvalue())

    @patch.dict("os.environ", {"HEROKU_API_KEY": "tok", "HEROKU_APP_NAME": "wilfred-production"})
    @patch("core.management.commands.restart_dynos.requests.delete")
    def test_all_targets_app_dynos_collection(self, mock_delete):
        mock_delete.return_value = self._fake_response(202)
        call_command("restart_dynos", "--type", "all", stdout=StringIO())
        url = mock_delete.call_args[0][0]
        self.assertEqual(url, "https://api.heroku.com/apps/wilfred-production/dynos")

    @patch.dict("os.environ", {"HEROKU_API_KEY": "tok"})
    def test_missing_app_raises(self):
        with self.assertRaises(CommandError) as ctx:
            call_command("restart_dynos", stdout=StringIO())
        self.assertIn("App name unknown", str(ctx.exception))

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        with self.assertRaises(CommandError) as ctx:
            call_command("restart_dynos", "--app", "x", stdout=StringIO())
        self.assertIn("HEROKU_API_KEY", str(ctx.exception))

    @patch.dict("os.environ", {"HEROKU_API_KEY": "tok", "HEROKU_APP_NAME": "x"})
    @patch("core.management.commands.restart_dynos.requests.delete")
    def test_api_error_raises_without_leaking_token(self, mock_delete):
        mock_delete.return_value = self._fake_response(401, text="unauthorized")
        with self.assertRaises(CommandError) as ctx:
            call_command("restart_dynos", stdout=StringIO())
        err = str(ctx.exception)
        self.assertIn("401", err)
        self.assertNotIn("tok", err)  # token must never surface in the error
