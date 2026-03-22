"""Tests for ChatThread and ChatMessage models."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.models import ChatMessage, ChatThread

User = get_user_model()


class ChatThreadModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", password="testpass123"
        )

    def test_create_thread(self):
        thread = ChatThread.objects.create(
            created_by=self.user, title="My thread"
        )
        self.assertIsNotNone(thread.id)
        self.assertEqual(thread.title, "My thread")
        self.assertEqual(thread.created_by, self.user)

    def test_thread_blank_title(self):
        thread = ChatThread.objects.create(
            created_by=self.user
        )
        self.assertEqual(thread.title, "")

    def test_thread_str(self):
        thread = ChatThread.objects.create(
            created_by=self.user, title="Chat about docs"
        )
        self.assertEqual(str(thread), "Chat about docs")

    def test_thread_str_no_title(self):
        thread = ChatThread.objects.create(
            created_by=self.user
        )
        self.assertIn("Thread", str(thread))

    def test_thread_ordering(self):
        t1 = ChatThread.objects.create(created_by=self.user, title="First")
        t2 = ChatThread.objects.create(created_by=self.user, title="Second")
        threads = list(ChatThread.objects.filter(created_by=self.user))
        # Most recently updated first
        self.assertEqual(threads[0].id, t2.id)
        self.assertEqual(threads[1].id, t1.id)


class ChatMessageModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", password="testpass123"
        )
        self.thread = ChatThread.objects.create(
            created_by=self.user
        )

    def test_create_message(self):
        msg = ChatMessage.objects.create(
            thread=self.thread, role="user", content="Hello"
        )
        self.assertIsNotNone(msg.id)
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")

    def test_message_roles(self):
        for role in ["system", "user", "assistant", "tool"]:
            msg = ChatMessage.objects.create(
                thread=self.thread, role=role, content=f"Test {role}"
            )
            self.assertEqual(msg.role, role)

    def test_message_ordering(self):
        m1 = ChatMessage.objects.create(thread=self.thread, role="user", content="First")
        m2 = ChatMessage.objects.create(thread=self.thread, role="assistant", content="Second")
        messages = list(self.thread.messages.all())
        self.assertEqual(messages[0].id, m1.id)
        self.assertEqual(messages[1].id, m2.id)

    def test_cascade_delete_with_thread(self):
        ChatMessage.objects.create(thread=self.thread, role="user", content="Hello")
        self.thread.delete()
        self.assertEqual(ChatMessage.objects.count(), 0)

    def test_tool_call_id(self):
        msg = ChatMessage.objects.create(
            thread=self.thread, role="tool", content="result",
            tool_call_id="call_abc123"
        )
        self.assertEqual(msg.tool_call_id, "call_abc123")

    def test_metadata_default(self):
        msg = ChatMessage.objects.create(
            thread=self.thread, role="user", content="Hello"
        )
        self.assertEqual(msg.metadata, {})

    def test_message_str(self):
        msg = ChatMessage.objects.create(
            thread=self.thread, role="user", content="Hello world"
        )
        self.assertIn("user", str(msg))
        self.assertIn("Hello", str(msg))

    def test_token_count_auto_computed_on_create(self):
        msg = ChatMessage.objects.create(
            thread=self.thread, role="user", content="Hello world"
        )
        self.assertGreater(msg.token_count, 0)

    def test_redacted_message_preserves_zero_token_count(self):
        """A redacted message with token_count=0 must not recompute on save."""
        msg = ChatMessage.objects.create(
            thread=self.thread, role="user", content="Original sensitive message"
        )
        original_tokens = msg.token_count
        self.assertGreater(original_tokens, 0)

        # Simulate redaction (uses .update() like the consumer)
        ChatMessage.objects.filter(pk=msg.pk).update(
            content="[This message was removed by the content safety system.]",
            is_redacted=True,
            token_count=0,
        )

        # Reload and re-save — token_count must stay 0
        msg.refresh_from_db()
        self.assertEqual(msg.token_count, 0)
        self.assertTrue(msg.is_redacted)

        msg.save()
        self.assertEqual(msg.token_count, 0)

    def test_explicit_token_count_not_overwritten(self):
        """If token_count is explicitly set, save() should not recompute."""
        msg = ChatMessage(
            thread=self.thread, role="assistant", content="A response",
            token_count=42,
        )
        msg.save()
        self.assertEqual(msg.token_count, 42)
