"""Tests for chat thread delete, archive, restore, and emoji."""

import json
import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatThread, ChatMessage

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class ThreadDeleteTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.thread = ChatThread.objects.create(
            created_by=self.user, title="My thread",
        )
        ChatMessage.objects.create(thread=self.thread, role="user", content="Hello")
        self.other = User.objects.create_user(email="other@example.com", password="testpass")

    def _url(self, thread_id=None):
        return reverse("thread_delete", kwargs={
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
        self.thread = ChatThread.objects.create(
            created_by=self.user, title="My thread",
        )
        self.other = User.objects.create_user(email="other@example.com", password="testpass")

    def _url(self, thread_id=None):
        return reverse("thread_archive", kwargs={
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
class ThreadEmojiTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.thread = ChatThread.objects.create(
            created_by=self.user, title="My thread",
        )
        self.other = User.objects.create_user(email="other@example.com", password="testpass")

    def _url(self, thread_id=None):
        return reverse("thread_emoji", kwargs={
            "thread_id": thread_id or self.thread.id,
        })

    def test_set_emoji(self):
        self.client.force_login(self.user)
        response = self.client.post(
            self._url(),
            data=json.dumps({"emoji": "\u2B50"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["emoji"], "\u2B50")
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.emoji, "\u2B50")

    def test_clear_emoji(self):
        self.thread.emoji = "\u2B50"
        self.thread.save(update_fields=["emoji"])
        self.client.force_login(self.user)
        response = self.client.post(
            self._url(),
            data=json.dumps({"emoji": ""}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["emoji"], "")
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.emoji, "")

    def test_emoji_requires_post(self):
        self.client.force_login(self.user)
        response = self.client.get(self._url())
        self.assertEqual(response.status_code, 405)

    def test_emoji_other_user_blocked(self):
        self.client.force_login(self.other)
        response = self.client.post(
            self._url(),
            data=json.dumps({"emoji": "\u2B50"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 404)
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.emoji, "")

    def test_emoji_truncated_to_max_length(self):
        self.client.force_login(self.user)
        response = self.client.post(
            self._url(),
            data=json.dumps({"emoji": "A" * 20}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.thread.refresh_from_db()
        self.assertEqual(len(self.thread.emoji), 8)
