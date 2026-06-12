"""Tests for Sentry initialization gating in ``config.settings``.

Sentry must initialize only in deployed environments (``DEBUG=False`` and not a
test run). Local development (``DEBUG=True``) and the test suite must never
initialize the SDK, otherwise intentional error-handling-test noise ("db boom",
"LLM down", mock artifacts) and routine local management commands (``check``,
``makemigrations``, ``runserver``, one-off shells) flood the shared Sentry
project with ``local_dev`` events that drown out real production issues.

Regression guard for the local_dev noise behind WILFRED-C / WILFRED-P /
WILFRED-5N / WILFRED-61, all of which originated from a developer machine.

The gate lives in module-level code in ``config/settings.py``, so each case is
exercised by importing the settings module in a fresh subprocess (mirroring
``accounts/tests/test_email_config.py``) and reporting whether the global Sentry
client ended up active.
"""
import os
import subprocess
import sys

from django.conf import settings
from django.test import SimpleTestCase

# Syntactically valid but non-routable DSN. ``sentry_sdk.init()`` parses it and
# the client reports active, but nothing is sent on import (no events captured).
_FAKE_DSN = "https://examplePublicKey@o0.ingest.sentry.io/0"

# Import settings (running its module-level Sentry block), then report whether
# the global client is active. Printed on its own final stdout line.
_PROBE = (
    "import sentry_sdk; "
    "from config import settings; "  # noqa: F401 — import triggers the gate
    "print('ACTIVE' if sentry_sdk.get_client().is_active() else 'INACTIVE')"
)


def _probe_sentry(env_overrides: dict) -> str:
    """Import ``config.settings`` in a fresh process; return ``'ACTIVE'``/``'INACTIVE'``."""
    env = {**os.environ, **env_overrides}
    # Minimal env so the module imports cleanly before/after the Sentry block:
    # a secret key (required when DEBUG=False) and a safe email backend (avoids
    # the unrelated production email-backend guard further down in settings).
    env.setdefault("DJANGO_SECRET_KEY", "x" * 50)
    env.setdefault("DJANGO_EMAIL_BACKEND", "django.core.mail.backends.locmem.EmailBackend")
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        cwd=str(settings.BASE_DIR),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(f"settings import failed: {result.stderr}")
    return result.stdout.strip().splitlines()[-1]


class SentryInitGatingTests(SimpleTestCase):
    """``config.settings`` initializes Sentry only in deployed, non-test runs."""

    def test_initializes_in_production(self) -> None:
        """DEBUG=False + DSN + not a test run → Sentry active (deploys still report)."""
        self.assertEqual(
            _probe_sentry({"DJANGO_DEBUG": "False", "SENTRY_DSN": _FAKE_DSN}),
            "ACTIVE",
        )

    def test_skipped_in_local_development(self) -> None:
        """DEBUG=True (developer machine) → Sentry never initializes, even with a DSN."""
        self.assertEqual(
            _probe_sentry({"DJANGO_DEBUG": "True", "SENTRY_DSN": _FAKE_DSN}),
            "INACTIVE",
        )

    def test_skipped_without_dsn(self) -> None:
        """No DSN → Sentry never initializes, even in a deployed-style run."""
        self.assertEqual(
            _probe_sentry({"DJANGO_DEBUG": "False", "SENTRY_DSN": ""}),
            "INACTIVE",
        )
