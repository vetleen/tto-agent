"""Tests for the pre_delete signal that redacts LLMCallLog on user deletion."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from llm.models import LLMCallLog

User = get_user_model()


class UserDeleteRedactsLLMCallLogTests(TestCase):
    def _make_log(self, user, prompt=None, raw_output="response body", tools=None):
        return LLMCallLog.objects.create(
            user=user,
            model="gpt-5-mini",
            prompt=prompt or [{"role": "user", "content": "hello"}],
            raw_output=raw_output,
            tools=tools or [{"name": "tool_a"}],
        )

    def test_prompt_and_raw_output_redacted_on_user_delete(self):
        user = User.objects.create_user(
            email="alice@example.com", password="pw12345!"
        )
        log = self._make_log(user)

        user.delete()

        log.refresh_from_db()
        self.assertEqual(log.prompt, {"redacted": True})
        self.assertEqual(log.raw_output, "")
        self.assertIsNone(log.tools)

    def test_user_fk_nulled_after_redaction(self):
        user = User.objects.create_user(
            email="bob@example.com", password="pw12345!"
        )
        log = self._make_log(user)

        user.delete()

        log.refresh_from_db()
        self.assertIsNone(log.user_id)

    def test_non_pii_columns_preserved(self):
        user = User.objects.create_user(
            email="carol@example.com", password="pw12345!"
        )
        log = LLMCallLog.objects.create(
            user=user,
            model="gpt-5-mini",
            prompt=[{"role": "user", "content": "hello"}],
            raw_output="response body",
            input_tokens=5,
            output_tokens=7,
            total_tokens=12,
        )

        user.delete()

        log.refresh_from_db()
        self.assertEqual(log.model, "gpt-5-mini")
        self.assertEqual(log.input_tokens, 5)
        self.assertEqual(log.output_tokens, 7)
        self.assertEqual(log.total_tokens, 12)

    def test_other_users_logs_unaffected(self):
        alice = User.objects.create_user(
            email="alice2@example.com", password="pw12345!"
        )
        bob = User.objects.create_user(
            email="bob2@example.com", password="pw12345!"
        )
        bob_log = self._make_log(bob, prompt=[{"role": "user", "content": "bob message"}])

        alice.delete()

        bob_log.refresh_from_db()
        self.assertEqual(bob_log.prompt, [{"role": "user", "content": "bob message"}])
        self.assertEqual(bob_log.raw_output, "response body")
