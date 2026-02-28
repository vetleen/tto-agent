"""Tests for ChatThread and ChatMessage models."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.models import ChatMessage, ChatThread
from documents.models import Project

User = get_user_model()


class ChatThreadModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", password="testpass123"
        )
        self.project = Project.objects.create(
            name="Test Project", slug="test-project", created_by=self.user
        )

    def test_create_thread(self):
        thread = ChatThread.objects.create(
            project=self.project, created_by=self.user, title="My thread"
        )
        self.assertIsNotNone(thread.id)
        self.assertEqual(thread.title, "My thread")
        self.assertEqual(thread.project, self.project)
        self.assertEqual(thread.created_by, self.user)

    def test_thread_blank_title(self):
        thread = ChatThread.objects.create(
            project=self.project, created_by=self.user
        )
        self.assertEqual(thread.title, "")

    def test_thread_str(self):
        thread = ChatThread.objects.create(
            project=self.project, created_by=self.user, title="Chat about docs"
        )
        self.assertEqual(str(thread), "Chat about docs")

    def test_thread_str_no_title(self):
        thread = ChatThread.objects.create(
            project=self.project, created_by=self.user
        )
        self.assertIn("Thread", str(thread))

    def test_thread_ordering(self):
        t1 = ChatThread.objects.create(project=self.project, created_by=self.user, title="First")
        t2 = ChatThread.objects.create(project=self.project, created_by=self.user, title="Second")
        threads = list(ChatThread.objects.filter(project=self.project))
        # Most recently updated first
        self.assertEqual(threads[0].id, t2.id)
        self.assertEqual(threads[1].id, t1.id)

    def test_cascade_delete_with_project(self):
        thread = ChatThread.objects.create(project=self.project, created_by=self.user)
        ChatMessage.objects.create(thread=thread, role="user", content="Hello")
        self.project.delete()
        self.assertEqual(ChatThread.objects.count(), 0)
        self.assertEqual(ChatMessage.objects.count(), 0)

    def test_related_name(self):
        ChatThread.objects.create(project=self.project, created_by=self.user)
        self.assertEqual(self.project.chat_threads.count(), 1)


class ChatMessageModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", password="testpass123"
        )
        self.project = Project.objects.create(
            name="Test Project", slug="test-project", created_by=self.user
        )
        self.thread = ChatThread.objects.create(
            project=self.project, created_by=self.user
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
