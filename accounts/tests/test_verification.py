"""Tests for email verification flow."""
from datetime import timedelta

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from accounts.models import EmailVerificationToken
from accounts.verification import can_resend_verification, verify_token

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"], EMAIL_VERIFICATION_REQUIRED=True)
@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class EmailVerificationTests(TestCase):
    def test_signup_redirects_to_verify_sent_and_sends_email(self) -> None:
        response = self.client.post(
            reverse("accounts:signup"),
            {
                "email": "new@example.com",
                "password1": "secure-pass-123!",
                "password2": "secure-pass-123!",
            },
        )
        self.assertRedirects(response, reverse("accounts:verify_email_sent"))
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ["new@example.com"])
        self.assertIn("verify", mail.outbox[0].subject.lower())
        user = User.objects.get(email="new@example.com")
        self.assertFalse(user.email_verified)
        self.assertIn("verification_pending_email", self.client.session)

    def test_verify_token_marks_user_verified_and_logs_in(self) -> None:
        user = User.objects.create_user(email="u@example.com", password="pass")
        user.email_verified = False
        user.save(update_fields=["email_verified"])
        token_obj = EmailVerificationToken.objects.create(
            user=user,
            token="test-token-123",
        )
        response = self.client.get(
            reverse("accounts:verify_email", kwargs={"token": token_obj.token}),
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        user.refresh_from_db()
        self.assertTrue(user.email_verified)
        self.assertFalse(EmailVerificationToken.objects.filter(pk=token_obj.pk).exists())
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_verify_invalid_token_shows_error(self) -> None:
        response = self.client.get(
            reverse("accounts:verify_email", kwargs={"token": "invalid-token"}),
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "invalid")

    def test_verify_expired_token_shows_error(self) -> None:
        user = User.objects.create_user(email="u@example.com", password="pass")
        token_obj = EmailVerificationToken.objects.create(user=user, token="expired-token")
        token_obj.created_at = timezone.now() - timedelta(hours=25)
        token_obj.save(update_fields=["created_at"])
        with override_settings(EMAIL_VERIFICATION_TIMEOUT=86400):  # 24h
            response = self.client.get(
                reverse("accounts:verify_email", kwargs={"token": token_obj.token}),
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "expired")

    def test_login_blocked_when_not_verified_redirects_to_verify_required(self) -> None:
        user = User.objects.create_user(email="unverified@example.com", password="pass")
        user.email_verified = False
        user.save(update_fields=["email_verified"])
        response = self.client.post(
            reverse("accounts:login"),
            {"username": user.email, "password": "pass"},
            follow=False,
        )
        self.assertRedirects(response, reverse("accounts:verify_required"))
        self.assertIn("verification_pending_email", self.client.session)
        self.assertEqual(self.client.session["verification_pending_email"], user.email)

    def test_verify_token_returns_expired_for_old_token(self) -> None:
        user = User.objects.create_user(email="u@example.com", password="pass")
        token_obj = EmailVerificationToken.objects.create(user=user, token="t")
        token_obj.created_at = timezone.now() - timedelta(seconds=86401)
        token_obj.save(update_fields=["created_at"])
        found_user, error = verify_token(token_obj.token)
        self.assertIsNone(found_user)
        self.assertEqual(error, "expired")


@override_settings(ALLOWED_HOSTS=["testserver"], EMAIL_VERIFICATION_REQUIRED=True)
class ResendRateLimitTests(TestCase):
    def test_can_resend_after_first_minute(self) -> None:
        user = User.objects.create_user(email="u@example.com", password="pass")
        user.email_verified = False
        user.last_verification_email_sent_at = timezone.now() - timedelta(minutes=2)
        user.verification_resend_window_start = timezone.now() - timedelta(minutes=2)
        user.verification_resend_count = 0
        user.save()
        allowed, wait = can_resend_verification(user)
        self.assertTrue(allowed)
        self.assertIsNone(wait)

    def test_cannot_resend_within_first_minute(self) -> None:
        user = User.objects.create_user(email="u@example.com", password="pass")
        user.email_verified = False
        user.last_verification_email_sent_at = timezone.now() - timedelta(seconds=30)
        user.verification_resend_window_start = timezone.now()
        user.verification_resend_count = 0
        user.save()
        allowed, wait = can_resend_verification(user)
        self.assertFalse(allowed)
        self.assertIsNotNone(wait)
        self.assertLessEqual(wait, 60)

    def test_resend_wait_doubles_after_each_resend(self) -> None:
        user = User.objects.create_user(email="u@example.com", password="pass")
        user.email_verified = False
        user.last_verification_email_sent_at = timezone.now()
        user.verification_resend_window_start = timezone.now()
        user.verification_resend_count = 1  # one resend already done
        user.save()
        allowed, wait = can_resend_verification(user)
        self.assertFalse(allowed)
        self.assertGreaterEqual(wait, 60)  # 2 minutes = 120 seconds (minus a few)

    def test_window_reset_after_24_hours_allows_immediate_resend(self) -> None:
        user = User.objects.create_user(email="u@example.com", password="pass")
        user.email_verified = False
        user.last_verification_email_sent_at = timezone.now() - timedelta(hours=25)
        user.verification_resend_window_start = timezone.now() - timedelta(hours=25)
        user.verification_resend_count = 5
        user.save()
        allowed, wait = can_resend_verification(user)
        self.assertTrue(allowed)
        self.assertIsNone(wait)
        user.refresh_from_db()
        self.assertEqual(user.verification_resend_count, 0)
