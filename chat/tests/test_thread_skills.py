"""Tests for chat.thread_skills — lock-serialized replacement of a thread's skills."""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.db import connection, transaction
from django.test import TestCase
from django.test.utils import CaptureQueriesContext

from agent_skills.models import AgentSkill
from chat.models import ChatThread, ChatThreadSkill
from chat.thread_skills import lock_thread, replace_thread_skills

User = get_user_model()


class ReplaceThreadSkillsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ts@example.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.a = AgentSkill.objects.create(
            slug="ts-a", name="A", instructions="x", level="user", created_by=self.user,
        )
        self.b = AgentSkill.objects.create(
            slug="ts-b", name="B", instructions="x", level="user", created_by=self.user,
        )

    def _ids(self):
        return list(
            ChatThreadSkill.objects.filter(thread=self.thread).values_list(
                "skill_id", flat=True
            )
        )

    def test_replaces_in_caller_order(self):
        replace_thread_skills(self.thread, [self.b.id, self.a.id])
        self.assertEqual(self._ids(), [self.b.id, self.a.id])
        replace_thread_skills(self.thread, [self.a.id])
        self.assertEqual(self._ids(), [self.a.id])

    def test_empty_list_detaches_all(self):
        replace_thread_skills(self.thread, [self.a.id])
        replace_thread_skills(self.thread, [])
        self.assertEqual(self._ids(), [])

    def test_already_attached_skill_is_replaced_not_duplicated(self):
        """WILFRED-8P shape: the desired set contains a skill another writer
        attached just before we took the lock. The replace drops and re-inserts
        it rather than tripping the (thread, skill) unique constraint."""
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.a)
        replace_thread_skills(self.thread, [self.a.id, self.b.id])
        self.assertEqual(self._ids(), [self.a.id, self.b.id])

    def test_lock_is_taken_before_delete_and_insert(self):
        """Query order: the thread-row lock precedes the DELETE and the INSERT,
        so a concurrent replace serializes on the lock, not on the rows."""
        with CaptureQueriesContext(connection) as cap:
            replace_thread_skills(self.thread, [self.a.id])
        sqls = [q["sql"] for q in cap.captured_queries]

        def first(pred):
            return next(i for i, s in enumerate(sqls) if pred(s))

        i_lock = first(
            lambda s: s.startswith("SELECT")
            and '"chat_chatthread"."id"' in s
            and "chat_chatthreadskill" not in s
        )
        i_delete = first(lambda s: s.startswith("DELETE") and "chat_chatthreadskill" in s)
        i_insert = first(lambda s: s.startswith("INSERT") and "chat_chatthreadskill" in s)
        self.assertLess(i_lock, i_delete)
        self.assertLess(i_delete, i_insert)

    def test_lock_thread_issues_for_update_on_supporting_backends(self):
        """SQLite has no row locks, so assert the query shape rather than its
        effect; on Postgres the same statement carries FOR UPDATE."""
        with transaction.atomic():
            with CaptureQueriesContext(connection) as cap:
                lock_thread(self.thread.pk)
        self.assertEqual(len(cap.captured_queries), 1)
        sql = cap.captured_queries[0]["sql"]
        self.assertIn('"chat_chatthread"', sql)
        if connection.features.has_select_for_update:
            self.assertIn("FOR UPDATE", sql)

    def test_replace_calls_lock_for_the_thread(self):
        with patch("chat.thread_skills.lock_thread") as lock:
            replace_thread_skills(self.thread, [self.a.id])
        lock.assert_called_once_with(self.thread.pk)
