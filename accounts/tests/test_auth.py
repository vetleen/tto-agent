from django.contrib.auth import get_user_model
from django.test import TestCase


class AccountsAuthTests(TestCase):
    def setUp(self) -> None:
        self.password = "test-pass-123"
        self.user = get_user_model().objects.create_user(
            email="tester@example.com",
            password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])

    def test_user_can_login_and_logout(self) -> None:
        logged_in = self.client.login(email=self.user.email, password=self.password)
        self.assertTrue(logged_in)

        response = self.client.get("/")
        self.assertTrue(response.wsgi_request.user.is_authenticated)

        self.client.logout()
        response = self.client.get("/")
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_user_email_can_be_updated(self) -> None:
        self.user.email = "new-email@example.com"
        self.user.save(update_fields=["email"])

        refreshed = get_user_model().objects.get(pk=self.user.pk)
        self.assertEqual(refreshed.email, "new-email@example.com")


class CaseInsensitiveEmailTests(TestCase):
    """Emails are stored lowercase and matched case-insensitively at login,
    consistent with PasswordResetForm's iexact lookup."""

    def setUp(self) -> None:
        self.password = "test-pass-123"
        self.user = get_user_model().objects.create_user(
            email="Mixed.Case@Example.COM",
            password=self.password,
        )

    def test_create_user_lowercases_email(self) -> None:
        self.assertEqual(self.user.email, "mixed.case@example.com")

    def test_login_with_case_variant_email(self) -> None:
        self.assertTrue(
            self.client.login(email="MIXED.CASE@example.com", password=self.password)
        )

    def test_get_by_natural_key_is_case_insensitive(self) -> None:
        found = get_user_model().objects.get_by_natural_key("Mixed.Case@example.com")
        self.assertEqual(found.pk, self.user.pk)

    def test_case_variant_duplicate_rejected(self) -> None:
        from django.db import IntegrityError

        with self.assertRaises(IntegrityError):
            # Bypass the manager's lowercasing to hit the DB constraint.
            get_user_model()(email="MIXED.case@example.com").save()
