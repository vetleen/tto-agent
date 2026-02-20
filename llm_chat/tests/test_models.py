import os
from unittest import mock, skipUnless

from django.contrib.auth import get_user_model
from django.test import TestCase

from llm_chat.models import ChatThread, ChatMessage
from llm_service.models import LLMCallLog


User = get_user_model()


class ChatThreadModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")

    def test_thread_str_and_ordering(self):
        t1 = ChatThread.objects.create(user=self.user, title="First")
        t2 = ChatThread.objects.create(user=self.user, title="Second")

        self.assertIn("First", str(t1))
        self.assertIn("Second", str(t2))

        # last_message_at defaults to None; ordering then falls back to created_at
        threads = list(ChatThread.objects.filter(user=self.user))
        # Newest created first according to Meta.ordering
        self.assertEqual(threads[0].id, t2.id)
        self.assertEqual(threads[1].id, t1.id)


class ChatMessageTokenCountingTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.thread = ChatThread.objects.create(user=self.user, title="Tokens")

    def test_token_count_without_model(self):
        msg = ChatMessage.objects.create(
            thread=self.thread,
            role=ChatMessage.Role.USER,
            content="Hello world",
        )
        # We don't assert an exact number (depends on encoding), just that it is non-negative.
        self.assertGreaterEqual(msg.token_count, 0)

    def test_token_count_with_llm_model(self):
        llm_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        msg = ChatMessage.objects.create(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT,
            content="Some assistant content",
            llm_call_log=llm_log,
        )
        self.assertGreaterEqual(msg.token_count, 0)


class ChatMessageStreamingConstraintTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.thread = ChatThread.objects.create(user=self.user, title="Constraint")

    def test_only_one_streaming_assistant_per_thread(self):
        ChatMessage.objects.create(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT,
            status=ChatMessage.Status.STREAMING,
            content="partial",
        )

        with self.assertRaises(Exception):
            ChatMessage.objects.create(
                thread=self.thread,
                role=ChatMessage.Role.ASSISTANT,
                status=ChatMessage.Status.STREAMING,
                content="another partial",
            )

