"""Tests for chat retention signals and save() override."""
from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.utils import timezone

from chat.models import ChatAttachment, ChatMessage, ChatThread, ChatThreadDataRoom
from documents.models import DataRoom

User = get_user_model()


class ChatThreadRetainUntilTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ret@example.com", password="testpass")

    def test_save_sets_retain_until(self):
        before = timezone.now()
        thread = ChatThread.objects.create(title="test", created_by=self.user)
        self.assertIsNotNone(thread.retain_until)
        self.assertGreaterEqual(thread.retain_until, before + timedelta(days=364))

    def test_save_with_update_fields_includes_retain_until(self):
        thread = ChatThread.objects.create(title="test", created_by=self.user)
        ChatThread.objects.filter(pk=thread.pk).update(
            retain_until=timezone.now() - timedelta(days=1),
        )
        thread.refresh_from_db()
        old_retain = thread.retain_until

        thread.title = "updated"
        thread.save(update_fields=["title"])
        thread.refresh_from_db()

        self.assertGreater(thread.retain_until, old_retain)

    def test_message_extends_thread_retain(self):
        thread = ChatThread.objects.create(title="test", created_by=self.user)
        ChatThread.objects.filter(pk=thread.pk).update(
            retain_until=timezone.now() - timedelta(days=1),
        )

        ChatMessage.objects.create(thread=thread, role="user", content="hello")
        thread.refresh_from_db()

        self.assertGreater(thread.retain_until, timezone.now() + timedelta(days=363))

    def test_message_extends_attached_dataroom_retain(self):
        thread = ChatThread.objects.create(title="test", created_by=self.user)
        dr = DataRoom.objects.create(name="DR", slug="dr-ret", created_by=self.user)
        ChatThreadDataRoom.objects.create(thread=thread, data_room=dr)
        DataRoom.objects.filter(pk=dr.pk).update(
            retain_until=timezone.now() - timedelta(days=1),
        )

        ChatMessage.objects.create(thread=thread, role="user", content="hello")
        dr.refresh_from_db()

        self.assertGreater(dr.retain_until, timezone.now() + timedelta(days=363))

    def test_attach_dataroom_extends_retain(self):
        thread = ChatThread.objects.create(title="test", created_by=self.user)
        dr = DataRoom.objects.create(name="DR2", slug="dr-ret2", created_by=self.user)
        DataRoom.objects.filter(pk=dr.pk).update(
            retain_until=timezone.now() - timedelta(days=1),
        )

        ChatThreadDataRoom.objects.create(thread=thread, data_room=dr)
        dr.refresh_from_db()

        self.assertGreater(dr.retain_until, timezone.now() + timedelta(days=363))


class ChatAttachmentFileCleanupTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._override = override_settings(MEDIA_ROOT=cls._tmpdir.name)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._override.disable()
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def setUp(self):
        self.user = User.objects.create_user(email="att@example.com", password="testpass")
        self.thread = ChatThread.objects.create(title="att", created_by=self.user)

    def test_delete_attachment_removes_file(self):
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="file")
        upload = SimpleUploadedFile("test.txt", b"content", content_type="text/plain")
        att = ChatAttachment.objects.create(
            thread=self.thread,
            message=msg,
            uploaded_by=self.user,
            file=upload,
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=7,
        )
        stored_path = Path(att.file.path)
        self.assertTrue(stored_path.exists())

        att.delete()
        self.assertFalse(stored_path.exists())

    def test_thread_delete_cascades_to_file_removal(self):
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="file")
        upload = SimpleUploadedFile("cascade.txt", b"data", content_type="text/plain")
        att = ChatAttachment.objects.create(
            thread=self.thread,
            message=msg,
            uploaded_by=self.user,
            file=upload,
            original_filename="cascade.txt",
            content_type="text/plain",
            size_bytes=4,
        )
        stored_path = Path(att.file.path)
        self.assertTrue(stored_path.exists())

        self.thread.delete()
        self.assertFalse(stored_path.exists())
