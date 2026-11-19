"""Deleting a chat thread redacts the content of its LLMCallLog rows
(chat/signals.py:redact_llm_call_logs_on_thread_delete)."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from chat.models import ChatThread, SubAgentRun
from llm.models import LLMCallLog

User = get_user_model()

PROMPT = [{"role": "user", "content": "secret question"}]


def _make_log(thread, user, **fields):
    defaults = dict(
        user=user,
        model="gpt-5-mini",
        prompt=PROMPT,
        raw_output="secret answer",
        tools=[{"name": "tool_a"}],
        conversation_id=str(thread.id) if thread else "",
        input_tokens=5,
        output_tokens=7,
        cost_usd=Decimal("0.25"),
    )
    defaults.update(fields)
    return LLMCallLog.objects.create(**defaults)


class ThreadDeleteRedactsLLMCallLogTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="thread@example.com", password="testpass")
        self.thread = ChatThread.objects.create(title="t", created_by=self.user)

    def assertRedacted(self, log):
        log.refresh_from_db()
        self.assertEqual(log.prompt, {"redacted": True})
        self.assertEqual(log.raw_output, "")
        self.assertIsNone(log.tools)

    def assertIntact(self, log):
        log.refresh_from_db()
        self.assertEqual(log.prompt, PROMPT)
        self.assertEqual(log.raw_output, "secret answer")

    def test_instance_delete_redacts_content(self):
        log = _make_log(self.thread, self.user)

        self.thread.delete()

        self.assertRedacted(log)

    def test_metadata_kept(self):
        log = _make_log(self.thread, self.user)
        created_at = log.created_at
        thread_id = str(self.thread.id)

        self.thread.delete()

        log.refresh_from_db()
        self.assertEqual(log.user_id, self.user.pk)
        self.assertEqual(log.conversation_id, thread_id)
        self.assertEqual(log.model, "gpt-5-mini")
        self.assertEqual(log.input_tokens, 5)
        self.assertEqual(log.output_tokens, 7)
        self.assertEqual(log.cost_usd, Decimal("0.25"))
        self.assertEqual(log.created_at, created_at)

    def test_subagent_rows_redacted(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="p", model_tier="medium",
        )
        log = _make_log(self.thread, self.user, run_id=str(run.id))

        self.thread.delete()

        self.assertRedacted(log)

    def test_other_threads_and_unlinked_rows_untouched(self):
        other = ChatThread.objects.create(title="other", created_by=self.user)
        other_log = _make_log(other, self.user)
        unlinked = _make_log(None, self.user)

        self.thread.delete()

        self.assertIntact(other_log)
        self.assertIntact(unlinked)

    def test_queryset_delete_redacts(self):
        other = ChatThread.objects.create(title="other", created_by=self.user)
        logs = [_make_log(self.thread, self.user), _make_log(other, self.user)]

        ChatThread.objects.filter(pk__in=[self.thread.pk, other.pk]).delete()

        for log in logs:
            self.assertRedacted(log)

    def test_enforce_retention_delete_redacts(self):
        log = _make_log(self.thread, self.user)
        ChatThread.objects.filter(pk=self.thread.pk).update(
            retain_until=timezone.now() - timedelta(days=1),
        )

        call_command("enforce_retention", "--target", "ChatThread", stdout=StringIO())

        self.assertFalse(ChatThread.objects.filter(pk=self.thread.pk).exists())
        self.assertRedacted(log)

    def test_user_cascade_redacts_rows_of_other_users_in_the_thread(self):
        # A row in the thread attributed to nobody (e.g. a shadow call) is not
        # covered by the user-delete signal; the thread cascade covers it.
        log = _make_log(self.thread, None)

        self.user.delete()

        self.assertRedacted(log)

    def test_already_redacted_row_is_left_alone(self):
        log = _make_log(self.thread, self.user, prompt={"redacted": True}, raw_output="", tools=None)

        self.thread.delete()

        self.assertRedacted(log)
