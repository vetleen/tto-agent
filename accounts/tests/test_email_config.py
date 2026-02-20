"""Tests for email configuration (django-anymail, DEFAULT_FROM_EMAIL, production validation).

No real API or domain required: uses locmem backend and subprocess to test validation.
"""
import subprocess
import sys
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse

User = get_user_model()


def _subprocess_load_settings(env_overrides: dict) -> subprocess.CompletedProcess:
    """Run a fresh Python process that loads config.settings with given env; return result."""
    env = {**dict(settings._wrapped.__dict__.get("_original_settings", {}) or {}), **env_overrides}
    # Ensure minimal env so settings load doesn't fail before email validation
    env.setdefault("DJANGO_SECRET_KEY", "x" * 50)
    return subprocess.run(
        [sys.executable, "-c", "from config import settings"],
        env={**__import__("os").environ, **env},
        cwd=str(settings.BASE_DIR),
        capture_output=True,
        text=True,
        timeout=10,
    )


@override_settings(ALLOWED_HOSTS=["testserver"], EMAIL_VERIFICATION_REQUIRED=True)
@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class EmailConfigTests(TestCase):
    """Test that email sending uses DEFAULT_FROM_EMAIL and works without any API."""

    def test_default_from_email_used_in_verification_email(self) -> None:
        with override_settings(DEFAULT_FROM_EMAIL="noreply@example.com"):
            self.client.post(
                reverse("accounts:signup"),
                {
                    "email": "new@example.com",
                    "password1": "secure-pass-123!",
                    "password2": "secure-pass-123!",
                },
            )
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].from_email, "noreply@example.com")

    def test_default_from_email_fallback_when_set(self) -> None:
        """With locmem and no override, sent mail uses settings default (webmaster@localhost)."""
        self.client.post(
            reverse("accounts:signup"),
            {
                "email": "user@example.com",
                "password1": "secure-pass-123!",
                "password2": "secure-pass-123!",
            },
        )
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].from_email, settings.DEFAULT_FROM_EMAIL)

    def test_send_mail_works_with_locmem_no_api_required(self) -> None:
        """Sending with locmem backend does not require any external API or domain."""
        from django.core.mail import send_mail

        send_mail(
            subject="Test",
            message="Body",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=["to@example.com"],
            fail_silently=False,
        )
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].subject, "Test")
        self.assertEqual(mail.outbox[0].to, ["to@example.com"])


@override_settings(ALLOWED_HOSTS=["testserver"])
class EmailValidationSubprocessTests(TestCase):
    """Test production email validation by loading settings in a subprocess with env vars."""

    def test_validation_fails_when_production_and_real_backend_without_enabled(self) -> None:
        result = _subprocess_load_settings({
            "DJANGO_DEBUG": "False",
            "DJANGO_EMAIL_BACKEND": "anymail.backends.mailgun.EmailBackend",
            "DEFAULT_FROM_EMAIL": "noreply@example.com",
            "MAILGUN_API_KEY": "key",
            "MAILGUN_SENDER_DOMAIN": "example.com",
        })
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EMAIL_SENDING_ENABLED", result.stderr)

    def test_validation_fails_when_production_and_invalid_default_from_email(self) -> None:
        result = _subprocess_load_settings({
            "DJANGO_DEBUG": "False",
            "DJANGO_EMAIL_BACKEND": "anymail.backends.mailgun.EmailBackend",
            "EMAIL_SENDING_ENABLED": "true",
            "DEFAULT_FROM_EMAIL": "not-an-email",
            "MAILGUN_API_KEY": "key",
            "MAILGUN_SENDER_DOMAIN": "example.com",
        })
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DEFAULT_FROM_EMAIL", result.stderr)

    def test_validation_fails_when_mailgun_backend_missing_vars(self) -> None:
        result = _subprocess_load_settings({
            "DJANGO_DEBUG": "False",
            "DJANGO_EMAIL_BACKEND": "anymail.backends.mailgun.EmailBackend",
            "EMAIL_SENDING_ENABLED": "true",
            "DEFAULT_FROM_EMAIL": "noreply@example.com",
            # MAILGUN_API_KEY and MAILGUN_SENDER_DOMAIN not set
        })
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MAILGUN", result.stderr)

    def test_validation_passes_with_safe_backend_in_production(self) -> None:
        """With DEBUG=False but console/locmem backend, no validation error (safe backends)."""
        result = _subprocess_load_settings({
            "DJANGO_DEBUG": "False",
            "DJANGO_EMAIL_BACKEND": "django.core.mail.backends.locmem.EmailBackend",
        })
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
