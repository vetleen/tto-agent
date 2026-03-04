"""Tests for chat thread delete, archive, restore, and auto-restore."""

import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatThread, ChatMessage
from documents.models import Project

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class ThreadDeleteTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.project = Project.objects.create(name="Test", slug="test", created_by=self.user)
        self.thread = ChatThread.objects.create(
            project=self.project, created_by=self.user, title="My thread",
        )
        ChatMessage.objects.create(thread=self.thread, role="user", content="Hello")
        self.other = User.objects.create_user(email="other@example.com", password="testpass")

    def _url(self, thread_id=None):
        return reverse("thread_delete", kwargs={
            "project_id": self.project.uuid,
            "thread_id": thread_id or self.thread.id,
        })

    def test_delete_thread(self):
        self.client.force_login(self.user)
        response = self.client.post(self._url())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertFalse(ChatThread.objects.filter(id=self.thread.id).exists())
        self.assertEqual(ChatMessage.objects.filter(thread=self.thread.id).count(), 0)

    def test_delete_requires_post(self):
        self.client.force_login(self.user)
        response = self.client.get(self._url())
        self.assertEqual(response.status_code, 405)

    def test_delete_other_user_blocked(self):
        self.client.force_login(self.other)
        response = self.client.post(self._url())
        self.assertEqual(response.status_code, 404)
        self.assertTrue(ChatThread.objects.filter(id=self.thread.id).exists())

    def test_delete_nonexistent_returns_404(self):
        self.client.force_login(self.user)
        response = self.client.post(self._url(thread_id=uuid.uuid4()))
        self.assertEqual(response.status_code, 404)


@override_settings(ALLOWED_HOSTS=["testserver"])
class ThreadArchiveTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.project = Project.objects.create(name="Test", slug="test", created_by=self.user)
        self.thread = ChatThread.objects.create(
            project=self.project, created_by=self.user, title="My thread",
        )
        self.other = User.objects.create_user(email="other@example.com", password="testpass")

    def _url(self, thread_id=None):
        return reverse("thread_archive", kwargs={
            "project_id": self.project.uuid,
            "thread_id": thread_id or self.thread.id,
        })

    def test_archive_thread(self):
        self.client.force_login(self.user)
        response = self.client.post(self._url())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["is_archived"])
        self.thread.refresh_from_db()
        self.assertTrue(self.thread.is_archived)

    def test_restore_thread(self):
        self.thread.is_archived = True
        self.thread.save(update_fields=["is_archived"])
        self.client.force_login(self.user)
        response = self.client.post(self._url())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertFalse(data["is_archived"])
        self.thread.refresh_from_db()
        self.assertFalse(self.thread.is_archived)

    def test_archive_requires_post(self):
        self.client.force_login(self.user)
        response = self.client.get(self._url())
        self.assertEqual(response.status_code, 405)

    def test_archive_other_user_blocked(self):
        self.client.force_login(self.other)
        response = self.client.post(self._url())
        self.assertEqual(response.status_code, 404)
        self.thread.refresh_from_db()
        self.assertFalse(self.thread.is_archived)


@override_settings(ALLOWED_HOSTS=["testserver"])
class ThreadAutoRestoreAndContextTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.project = Project.objects.create(name="Test", slug="test", created_by=self.user)
        self.active_thread = ChatThread.objects.create(
            project=self.project, created_by=self.user, title="Active",
        )
        self.archived_thread = ChatThread.objects.create(
            project=self.project, created_by=self.user, title="Archived", is_archived=True,
        )

    def _chat_url(self, thread_id=None):
        url = reverse("project_chat", kwargs={"project_id": self.project.uuid})
        if thread_id:
            url += f"?thread={thread_id}"
        return url

    def test_auto_restore_on_open(self):
        self.client.force_login(self.user)
        response = self.client.get(self._chat_url(thread_id=self.archived_thread.id))
        self.assertEqual(response.status_code, 200)
        self.archived_thread.refresh_from_db()
        self.assertFalse(self.archived_thread.is_archived)

    def test_active_threads_exclude_archived(self):
        self.client.force_login(self.user)
        response = self.client.get(self._chat_url())
        self.assertEqual(response.status_code, 200)
        threads = response.context["threads"]
        thread_ids = [t.id for t in threads]
        self.assertIn(self.active_thread.id, thread_ids)
        self.assertNotIn(self.archived_thread.id, thread_ids)

    def test_archived_threads_in_context(self):
        self.client.force_login(self.user)
        response = self.client.get(self._chat_url())
        self.assertEqual(response.status_code, 200)
        archived = response.context["archived_threads"]
        archived_ids = [t.id for t in archived]
        self.assertIn(self.archived_thread.id, archived_ids)
        self.assertNotIn(self.active_thread.id, archived_ids)
