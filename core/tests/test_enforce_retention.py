"""Tests for the enforce_retention management command."""
from __future__ import annotations

import tempfile
from datetime import timedelta
from io import StringIO
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.models import EmailVerificationToken
from chat.models import ChatMessage, ChatThread
from documents.models import DataRoom, DataRoomDocument
from feedback.models import Feedback
from guardrails.models import GuardrailEvent
from meetings.models import Meeting

User = get_user_model()


def _backdate_retain(model_cls, pk, retain_until):
    model_cls.objects.filter(pk=pk).update(retain_until=retain_until)


class EnforceRetentionCommandTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ret@example.com", password="testpass")
        self.now = timezone.now()

    # ── ChatThread ──────────────────────────────────────────────────

    def test_expired_chatthread_deleted(self):
        thread = ChatThread.objects.create(title="old", created_by=self.user)
        _backdate_retain(ChatThread, thread.pk, self.now - timedelta(days=1))

        call_command("enforce_retention", stdout=StringIO())

        self.assertFalse(ChatThread.objects.filter(pk=thread.pk).exists())

    def test_active_chatthread_not_deleted(self):
        thread = ChatThread.objects.create(title="new", created_by=self.user)

        call_command("enforce_retention", stdout=StringIO())

        self.assertTrue(ChatThread.objects.filter(pk=thread.pk).exists())

    def test_null_retain_until_not_deleted(self):
        thread = ChatThread.objects.create(title="null", created_by=self.user)
        ChatThread.objects.filter(pk=thread.pk).update(retain_until=None)

        call_command("enforce_retention", stdout=StringIO())

        self.assertTrue(ChatThread.objects.filter(pk=thread.pk).exists())

    def test_expired_chatthread_cascades_messages(self):
        thread = ChatThread.objects.create(title="cascade", created_by=self.user)
        ChatMessage.objects.create(thread=thread, role="user", content="hello")
        _backdate_retain(ChatThread, thread.pk, self.now - timedelta(days=1))

        call_command("enforce_retention", stdout=StringIO())

        self.assertEqual(ChatMessage.objects.filter(thread=thread.pk).count(), 0)

    # ── DataRoom ────────────────────────────────────────────────────

    def test_expired_dataroom_deleted(self):
        dr = DataRoom.objects.create(name="Old DR", slug="old-dr", created_by=self.user)
        _backdate_retain(DataRoom, dr.pk, self.now - timedelta(days=1))

        call_command("enforce_retention", stdout=StringIO())

        self.assertFalse(DataRoom.objects.filter(pk=dr.pk).exists())

    # ── Meeting ─────────────────────────────────────────────────────

    def test_expired_meeting_deleted(self):
        m = Meeting.objects.create(name="Old meeting", slug="old-m", created_by=self.user)
        _backdate_retain(Meeting, m.pk, self.now - timedelta(days=1))

        call_command("enforce_retention", stdout=StringIO())

        self.assertFalse(Meeting.objects.filter(pk=m.pk).exists())

    # ── GuardrailEvent ──────────────────────────────────────────────

    def test_expired_guardrail_deleted(self):
        event = GuardrailEvent.objects.create(
            user=self.user,
            trigger_source="user_message",
            check_type="heuristic",
            severity="low",
            action_taken="logged",
            raw_input="test",
        )
        _backdate_retain(GuardrailEvent, event.pk, self.now - timedelta(days=1))

        call_command("enforce_retention", stdout=StringIO())

        self.assertFalse(GuardrailEvent.objects.filter(pk=event.pk).exists())

    # ── Feedback ────────────────────────────────────────────────────

    def test_expired_feedback_deleted(self):
        fb = Feedback.objects.create(user=self.user, text="test feedback")
        _backdate_retain(Feedback, fb.pk, self.now - timedelta(days=1))

        call_command("enforce_retention", stdout=StringIO())

        self.assertFalse(Feedback.objects.filter(pk=fb.pk).exists())

    # ── EmailVerificationToken ──────────────────────────────────────

    def test_expired_verification_token_deleted(self):
        token = EmailVerificationToken.objects.create(user=self.user, token="abc123")
        EmailVerificationToken.objects.filter(pk=token.pk).update(
            created_at=self.now - timedelta(days=2),
        )

        call_command("enforce_retention", stdout=StringIO())

        self.assertFalse(EmailVerificationToken.objects.filter(pk=token.pk).exists())

    def test_fresh_verification_token_not_deleted(self):
        token = EmailVerificationToken.objects.create(user=self.user, token="def456")

        call_command("enforce_retention", stdout=StringIO())

        self.assertTrue(EmailVerificationToken.objects.filter(pk=token.pk).exists())

    # ── Command flags ───────────────────────────────────────────────

    def test_dry_run_does_not_delete(self):
        thread = ChatThread.objects.create(title="dry", created_by=self.user)
        _backdate_retain(ChatThread, thread.pk, self.now - timedelta(days=1))

        out = StringIO()
        call_command("enforce_retention", "--dry-run", stdout=out)

        self.assertTrue(ChatThread.objects.filter(pk=thread.pk).exists())
        self.assertIn("dry-run", out.getvalue())

    def test_model_filter(self):
        thread = ChatThread.objects.create(title="filter", created_by=self.user)
        fb = Feedback.objects.create(user=self.user, text="filter test")
        _backdate_retain(ChatThread, thread.pk, self.now - timedelta(days=1))
        _backdate_retain(Feedback, fb.pk, self.now - timedelta(days=1))

        call_command("enforce_retention", "--target", "ChatThread", stdout=StringIO())

        self.assertFalse(ChatThread.objects.filter(pk=thread.pk).exists())
        self.assertTrue(Feedback.objects.filter(pk=fb.pk).exists())

    def test_batch_size_processes_all(self):
        threads = [
            ChatThread.objects.create(title=f"batch-{i}", created_by=self.user)
            for i in range(5)
        ]
        for t in threads:
            _backdate_retain(ChatThread, t.pk, self.now - timedelta(days=1))

        call_command("enforce_retention", "--batch-size", "2", stdout=StringIO())

        self.assertEqual(
            ChatThread.objects.filter(pk__in=[t.pk for t in threads]).count(), 0,
        )


class EnforceRetentionFileCleanupTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory()
        # Pin the storage backend, not just MEDIA_ROOT: when the local .env
        # sets AWS_STORAGE_BUCKET_NAME, default storage is S3 — the upload
        # below would hit the real bucket and .path would raise
        # NotImplementedError. Tests must stay on the local filesystem.
        cls._override = override_settings(
            MEDIA_ROOT=cls._tmpdir.name,
            STORAGES={
                "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
                "staticfiles": {
                    "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
                },
            },
        )
        cls._override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._override.disable()
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def setUp(self):
        self.user = User.objects.create_user(email="file@example.com", password="testpass")
        self.now = timezone.now()

    def test_expired_dataroom_cascades_to_file_removal(self):
        dr = DataRoom.objects.create(name="File DR", slug="file-dr", created_by=self.user)
        upload = SimpleUploadedFile("doc.txt", b"hello", content_type="text/plain")
        doc = DataRoomDocument.objects.create(
            data_room=dr,
            uploaded_by=self.user,
            original_file=upload,
            original_filename="doc.txt",
            status=DataRoomDocument.Status.UPLOADED,
        )
        stored_path = Path(doc.original_file.path)
        self.assertTrue(stored_path.exists())

        _backdate_retain(DataRoom, dr.pk, self.now - timedelta(days=1))
        call_command("enforce_retention", stdout=StringIO())

        self.assertFalse(stored_path.exists())
