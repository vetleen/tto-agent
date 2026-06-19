"""Tests for the compression indicator + re-attach UI."""

import tempfile

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatAttachment, ChatMessage, ChatThread

User = get_user_model()

_MEDIA = tempfile.mkdtemp()


@override_settings(MEDIA_ROOT=_MEDIA)
class ReattachAttachmentTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ra@test.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.client.force_login(self.user)

    def test_reattach_copies_file_to_new_pending_attachment(self):
        old = ChatAttachment.objects.create(
            thread=self.thread, uploaded_by=self.user,
            file=ContentFile(b"PNGDATA", name="chart.png"),
            original_filename="chart.png", content_type="image/png", size_bytes=7,
        )
        resp = self.client.post(
            reverse("chat_reattach_attachment", args=[self.thread.id, old.id])
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["filename"], "chart.png")
        self.assertEqual(data["content_type"], "image/png")

        new = ChatAttachment.objects.get(id=data["id"])
        self.assertNotEqual(new.id, old.id)
        # Copied, not shared — so the file-delete signal can't orphan the original.
        self.assertNotEqual(new.file.name, old.file.name)
        self.assertIsNone(new.message_id)  # pending until the next message is sent

    def test_reattach_denied_for_non_owner(self):
        other = User.objects.create_user(email="ra2@test.com", password="pw")
        old = ChatAttachment.objects.create(
            thread=self.thread, uploaded_by=self.user,
            file=ContentFile(b"x", name="x.png"),
            original_filename="x.png", content_type="image/png", size_bytes=1,
        )
        self.client.force_login(other)
        resp = self.client.post(
            reverse("chat_reattach_attachment", args=[self.thread.id, old.id])
        )
        self.assertEqual(resp.status_code, 404)


@override_settings(MEDIA_ROOT=_MEDIA)
class CompressionIndicatorTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ci@test.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.client.force_login(self.user)

    def test_divider_and_dropped_reattach_rendered(self):
        m1 = ChatMessage.objects.create(thread=self.thread, role="user", content="first")
        att = ChatAttachment.objects.create(
            thread=self.thread, message=m1, uploaded_by=self.user,
            file=ContentFile(b"x", name="chart.png"),
            original_filename="chart.png", content_type="image/png", size_bytes=1,
        )
        m1.metadata = {"attachment_ids": [str(att.id)]}
        m1.save(update_fields=["metadata"])
        ChatMessage.objects.create(thread=self.thread, role="assistant", content="reply")
        ChatMessage.objects.create(thread=self.thread, role="user", content="second")
        # Compression boundary = the first message.
        self.thread.summary = "a summary"
        self.thread.summary_up_to_message_id = m1.id
        self.thread.save(update_fields=["summary", "summary_up_to_message_id"])

        resp = self.client.get(reverse("chat_home") + f"?thread={self.thread.id}")
        self.assertEqual(resp.status_code, 200)
        html = resp.content.decode()
        self.assertIn("Earlier messages summarized", html)  # the divider
        self.assertIn("reattach-btn", html)  # dropped attachment gets re-attach
        self.assertIn("chart.png", html)

    def test_no_divider_without_summary(self):
        ChatMessage.objects.create(thread=self.thread, role="user", content="hi")
        resp = self.client.get(reverse("chat_home") + f"?thread={self.thread.id}")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("Earlier messages summarized", resp.content.decode())
