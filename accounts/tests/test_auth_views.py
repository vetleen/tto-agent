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
        self.assertContains(response, 'name="password"')

        response = self.client.post(
            reverse("accounts:account_delete"),
            {"password": self.password},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(get_user_model().objects.filter(pk=self.user.pk).exists())

    def test_delete_account_wipes_chat_and_feedback_and_redacts_llm_logs(self) -> None:
        """GDPR Art. 17: deleting the account must CASCADE chat + feedback
        personal data and redact LLMCallLog content while preserving usage
        metadata (model, tokens) with a nulled user FK."""
        from chat.models import ChatMessage, ChatThread
        from feedback.models import Feedback
        from llm.models import LLMCallLog

        thread = ChatThread.objects.create(created_by=self.user, title="t")
        message = ChatMessage.objects.create(
            thread=thread, role=ChatMessage.Role.USER, content="hello world"
        )
        feedback = Feedback.objects.create(
            user=self.user, text="this is my feedback", user_agent="Mozilla"
        )
        llm_log = LLMCallLog.objects.create(
            user=self.user,
            model="gpt-5-mini",
            prompt=[{"role": "user", "content": "secret prompt"}],
            raw_output="secret response",
            tools=[{"name": "tool_a"}],
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )

        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            reverse("accounts:account_delete"),
            {"password": self.password},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)

        self.assertFalse(ChatThread.objects.filter(pk=thread.pk).exists())
        self.assertFalse(ChatMessage.objects.filter(pk=message.pk).exists())
        self.assertFalse(Feedback.objects.filter(pk=feedback.pk).exists())

        llm_log.refresh_from_db()
        self.assertEqual(llm_log.prompt, {"redacted": True})
        self.assertEqual(llm_log.raw_output, "")
        self.assertIsNone(llm_log.tools)
        self.assertIsNone(llm_log.user_id)
        self.assertEqual(llm_log.model, "gpt-5-mini")
        self.assertEqual(llm_log.input_tokens, 10)
        self.assertEqual(llm_log.output_tokens, 20)
        self.assertEqual(llm_log.total_tokens, 30)

    def test_delete_account_wrong_password_rejected(self) -> None:
        self.client.login(email=self.user.email, password=self.password)

        response = self.client.post(
            reverse("accounts:account_delete"),
            {"password": "wrong-password"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Incorrect password")
        # User must still exist
        self.assertTrue(get_user_model().objects.filter(pk=self.user.pk).exists())

    def test_delete_account_no_password_rejected(self) -> None:
        self.client.login(email=self.user.email, password=self.password)

        response = self.client.post(reverse("accounts:account_delete"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(get_user_model().objects.filter(pk=self.user.pk).exists())

    def test_delete_account_clears_session(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        self.client.post(
            reverse("accounts:account_delete"),
            {"password": self.password},
        )
        # Session should be cleared — subsequent requests must not be authenticated
        response = self.client.get(reverse("accounts:account_delete"))
        self.assertRedirects(response, reverse("accounts:login"))

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
