import time
from urllib.parse import urlparse

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(ALLOWED_HOSTS=["testserver"])
class AuthViewsTests(TestCase):
    def setUp(self) -> None:
        self.password = "test-pass-123"
        self.user = get_user_model().objects.create_user(
            email="tester@example.com",
            password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])

    def test_login_and_logout_flow(self) -> None:
        response = self.client.get(reverse("accounts:login"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="username"')
        self.assertContains(response, 'name="password"')

        response = self.client.post(
            reverse("accounts:login"),
            {"username": self.user.email, "password": self.password},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.wsgi_request.user.is_authenticated)

        response = self.client.post(reverse("accounts:logout"), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_signup_like_flow_creates_user(self) -> None:
        user = get_user_model().objects.create_user(
            email="newuser@example.com",
            password="signup-pass-123",
        )
        user.email_verified = True
        user.save(update_fields=["email_verified"])
        self.assertIsNotNone(user.pk)
        logged_in = self.client.login(email=user.email, password="signup-pass-123")
        self.assertTrue(logged_in)

    def test_delete_account_flow(self) -> None:
        self.client.login(email=self.user.email, password=self.password)

        response = self.client.get(reverse("accounts:account_delete"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Delete account")

        response = self.client.post(reverse("accounts:account_delete"), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(get_user_model().objects.filter(pk=self.user.pk).exists())

    def test_password_change_flow(self) -> None:
        self.client.login(email=self.user.email, password=self.password)

        response = self.client.get(reverse("accounts:password_change"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="old_password"')
        self.assertContains(response, 'name="new_password1"')
        self.assertContains(response, 'name="new_password2"')

        response = self.client.post(
            reverse("accounts:password_change"),
            {
                "old_password": self.password,
                "new_password1": "new-pass-456",
                "new_password2": "new-pass-456",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)

        self.client.logout()
        logged_in = self.client.login(email=self.user.email, password="new-pass-456")
        self.assertTrue(logged_in)

    @override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
    def test_password_reset_flow(self) -> None:
        response = self.client.get(reverse("accounts:password_reset"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="email"')

        response = self.client.post(
            reverse("accounts:password_reset"),
            {"email": self.user.email},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(mail.outbox), 1)

        #time.sleep(5)

        reset_url = mail.outbox[0].body.strip().splitlines()[-1]
        reset_path = urlparse(reset_url).path

        response = self.client.get(reset_path, follow=True)
        self.assertEqual(response.status_code, 200)
        confirm_path = response.request["PATH_INFO"]
        self.assertContains(response, 'name="new_password1"')
        self.assertContains(response, 'name="new_password2"')

        response = self.client.post(
            confirm_path,
            {"new_password1": "reset-pass-789", "new_password2": "reset-pass-789"},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)

        self.client.logout()
        logged_in = self.client.login(email=self.user.email, password="reset-pass-789")
        self.assertTrue(logged_in)

    def test_password_reset_invalid_link_shows_message(self) -> None:
        response = self.client.get(
            reverse("accounts:password_reset_confirm", kwargs={"uidb64": "invalid", "token": "invalid-token"}),
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Reset link is invalid")
