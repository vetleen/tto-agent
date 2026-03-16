"""Tests for chat image attachment upload, linking, and multimodal block construction."""

import io
import uuid

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatAttachment, ChatMessage, ChatThread
from chat.services import build_image_content_block

User = get_user_model()


def _tiny_png():
    """Return a minimal valid 1x1 PNG byte string."""
    import struct
    import zlib

    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\x00\x00\x00")
    idat = _chunk(b"IDAT", raw)
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
    DEFAULT_FILE_STORAGE="django.core.files.storage.InMemoryStorage",
)
class UploadAttachmentTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="att@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.url = reverse("chat_upload_attachments", args=[self.thread.id])

    def test_upload_valid_image(self):
        f = SimpleUploadedFile("test.png", _tiny_png(), content_type="image/png")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["attachments"]), 1)
        att = data["attachments"][0]
        self.assertEqual(att["filename"], "test.png")
        self.assertEqual(att["content_type"], "image/png")
        # Verify DB record
        self.assertTrue(ChatAttachment.objects.filter(id=att["id"]).exists())
        db_att = ChatAttachment.objects.get(id=att["id"])
        self.assertIsNone(db_att.message)
        self.assertEqual(db_att.thread, self.thread)

    def test_upload_multiple_images(self):
        f1 = SimpleUploadedFile("a.png", _tiny_png(), content_type="image/png")
        f2 = SimpleUploadedFile("b.png", _tiny_png(), content_type="image/png")
        resp = self.client.post(self.url, {"files": [f1, f2]})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["attachments"]), 2)

    def test_upload_wrong_type_rejected(self):
        f = SimpleUploadedFile("test.pdf", b"%PDF-1.4 test", content_type="application/pdf")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file type", resp.json()["error"])

    def test_upload_oversized_rejected(self):
        # 11 MB file
        big = b"\x00" * (11 * 1024 * 1024)
        f = SimpleUploadedFile("big.png", big, content_type="image/png")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("too large", resp.json()["error"])

    def test_upload_requires_auth(self):
        self.client.logout()
        f = SimpleUploadedFile("test.png", _tiny_png(), content_type="image/png")
        resp = self.client.post(self.url, {"files": f})
        self.assertIn(resp.status_code, [302, 403])

    def test_upload_enforces_thread_ownership(self):
        other_user = User.objects.create_user(email="other@example.com", password="testpass")
        other_user.email_verified = True
        other_user.save(update_fields=["email_verified"])
        other_thread = ChatThread.objects.create(created_by=other_user)
        url = reverse("chat_upload_attachments", args=[other_thread.id])
        f = SimpleUploadedFile("test.png", _tiny_png(), content_type="image/png")
        resp = self.client.post(url, {"files": f})
        self.assertEqual(resp.status_code, 404)

    def test_upload_no_files_returns_400(self):
        resp = self.client.post(self.url)
        self.assertEqual(resp.status_code, 400)


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
    DEFAULT_FILE_STORAGE="django.core.files.storage.InMemoryStorage",
)
class AttachmentLinkingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="link@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.message = ChatMessage.objects.create(
            thread=self.thread, role="user", content="test",
        )

    def test_link_attachments_to_message(self):
        att = ChatAttachment.objects.create(
            thread=self.thread,
            uploaded_by=self.user,
            file=SimpleUploadedFile("img.png", _tiny_png(), content_type="image/png"),
            original_filename="img.png",
            content_type="image/png",
            size_bytes=100,
        )
        self.assertIsNone(att.message)
        ChatAttachment.objects.filter(
            id=att.id,
            thread=self.thread,
            uploaded_by=self.user,
            message__isnull=True,
        ).update(message=self.message)
        att.refresh_from_db()
        self.assertEqual(att.message, self.message)

    def test_link_ignores_already_linked(self):
        """Attachments already linked to another message should not be re-linked."""
        other_msg = ChatMessage.objects.create(
            thread=self.thread, role="user", content="other",
        )
        att = ChatAttachment.objects.create(
            thread=self.thread,
            uploaded_by=self.user,
            message=other_msg,
            file=SimpleUploadedFile("img.png", _tiny_png(), content_type="image/png"),
            original_filename="img.png",
            content_type="image/png",
            size_bytes=100,
        )
        # Try to link to self.message — should not match because message is not null
        updated = ChatAttachment.objects.filter(
            id=att.id,
            thread=self.thread,
            uploaded_by=self.user,
            message__isnull=True,
        ).update(message=self.message)
        self.assertEqual(updated, 0)
        att.refresh_from_db()
        self.assertEqual(att.message, other_msg)


class BuildImageContentBlockTests(TestCase):
    def test_anthropic_format(self):
        block = build_image_content_block("abc123", "image/png", "anthropic")
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["source"]["type"], "base64")
        self.assertEqual(block["source"]["media_type"], "image/png")
        self.assertEqual(block["source"]["data"], "abc123")

    def test_openai_format(self):
        block = build_image_content_block("abc123", "image/jpeg", "openai")
        self.assertEqual(block["type"], "image_url")
        self.assertIn("data:image/jpeg;base64,abc123", block["image_url"]["url"])

    def test_gemini_format_uses_openai_style(self):
        block = build_image_content_block("xyz", "image/webp", "gemini")
        self.assertEqual(block["type"], "image_url")

    def test_unknown_provider_uses_openai_style(self):
        block = build_image_content_block("data", "image/gif", "")
        self.assertEqual(block["type"], "image_url")


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
)
class ChatHomeVisionContextTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="vis@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)

    def test_model_choices_include_supports_vision(self):
        import json

        response = self.client.get(reverse("chat_home"))
        choices = json.loads(response.context["model_choices_json"])
        for c in choices:
            self.assertIn("supports_vision", c)
            self.assertIsInstance(c["supports_vision"], bool)
