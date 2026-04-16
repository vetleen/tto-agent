"""Tests for chat attachment upload, linking, and multimodal block construction."""

import io
import uuid

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatAttachment, ChatMessage, ChatThread
from chat.services import (
    build_image_content_block,
    build_pdf_content_block,
    build_text_content_block,
    extract_docx_text,
)

User = get_user_model()


def _tiny_docx():
    """Return a minimal valid .docx byte string with 'Hello World' content."""
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '</Types>'
        ))
        zf.writestr("_rels/.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            '</Relationships>'
        ))
        zf.writestr("word/document.xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>Hello World</w:t></w:r></w:p></w:body>'
            '</w:document>'
        ))
    return buf.getvalue()


def _docx_with_image():
    """Return a valid .docx byte string containing an embedded PNG image."""
    import zipfile

    png_data = _tiny_png()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="png" ContentType="image/png"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '</Types>'
        ))
        zf.writestr("_rels/.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            '</Relationships>'
        ))
        zf.writestr("word/_rels/document.xml.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="media/image1.png"/>'
            '</Relationships>'
        ))
        zf.writestr("word/document.xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
            ' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
            ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
            ' xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
            ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<w:body>'
            '<w:p><w:r><w:t>Before image</w:t></w:r></w:p>'
            '<w:p><w:r><w:drawing>'
            '<wp:inline><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic><pic:blipFill><a:blip r:embed="rId1"/></pic:blipFill></pic:pic>'
            '</a:graphicData></a:graphic></wp:inline>'
            '</w:drawing></w:r></w:p>'
            '<w:p><w:r><w:t>After image</w:t></w:r></w:p>'
            '</w:body>'
            '</w:document>'
        ))
        zf.writestr("word/media/image1.png", png_data)
    return buf.getvalue()


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
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
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

    def test_upload_valid_pdf(self):
        f = SimpleUploadedFile("test.pdf", b"%PDF-1.4 test", content_type="application/pdf")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["attachments"][0]["content_type"], "application/pdf")

    def test_upload_valid_text_file(self):
        f = SimpleUploadedFile("readme.txt", b"Hello world", content_type="text/plain")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["attachments"][0]["content_type"], "text/plain")

    def test_upload_valid_csv(self):
        f = SimpleUploadedFile("data.csv", b"a,b,c\n1,2,3", content_type="text/csv")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["attachments"][0]["content_type"], "text/csv")

    def test_upload_valid_json(self):
        f = SimpleUploadedFile("data.json", b'{"key": "val"}', content_type="application/json")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["attachments"][0]["content_type"], "application/json")

    def test_upload_valid_docx(self):
        f = SimpleUploadedFile(
            "test.docx", _tiny_docx(),
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json()["attachments"][0]["content_type"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    def test_upload_docx_octet_stream_fallback(self):
        """Browsers may report .docx as application/octet-stream — accept by extension."""
        f = SimpleUploadedFile("report.docx", _tiny_docx(), content_type="application/octet-stream")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json()["attachments"][0]["content_type"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    def test_upload_exe_rejected(self):
        f = SimpleUploadedFile("malware.exe", b"\x00" * 100, content_type="application/x-msdownload")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file type", resp.json()["error"])

    def test_upload_pdf_oversized(self):
        # >30 MB PDF
        big = b"\x00" * (31 * 1024 * 1024)
        f = SimpleUploadedFile("huge.pdf", big, content_type="application/pdf")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("too large", resp.json()["error"])

    def test_upload_text_oversized(self):
        # >10 MB text
        big = b"x" * (11 * 1024 * 1024)
        f = SimpleUploadedFile("big.txt", big, content_type="text/plain")
        resp = self.client.post(self.url, {"files": f})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("too large", resp.json()["error"])

    def test_upload_image_oversized_rejected(self):
        # 11 MB image
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
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
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


class BuildPdfContentBlockTests(TestCase):
    def test_anthropic_format(self):
        block = build_pdf_content_block("abc123", "report.pdf", "anthropic")
        self.assertEqual(block["type"], "document")
        self.assertEqual(block["source"]["type"], "base64")
        self.assertEqual(block["source"]["media_type"], "application/pdf")
        self.assertEqual(block["source"]["data"], "abc123")

    def test_openai_format(self):
        block = build_pdf_content_block("abc123", "report.pdf", "openai")
        self.assertEqual(block["type"], "file")
        self.assertEqual(block["file"]["filename"], "report.pdf")
        self.assertIn("data:application/pdf;base64,abc123", block["file"]["file_data"])

    def test_gemini_format(self):
        block = build_pdf_content_block("abc123", "report.pdf", "gemini")
        self.assertEqual(block["type"], "image_url")
        self.assertIn("data:application/pdf;base64,abc123", block["image_url"]["url"])


class BuildTextContentBlockTests(TestCase):
    def test_wraps_text_with_filename(self):
        block = build_text_content_block("col1,col2\n1,2", "data.csv")
        self.assertEqual(block["type"], "text")
        self.assertIn("[Attached file: data.csv]", block["text"])
        self.assertIn("col1,col2", block["text"])


class ExtractDocxTextTests(TestCase):
    def test_extract_from_minimal_docx(self):
        text = extract_docx_text(_tiny_docx())
        self.assertIn("Hello World", text)

    def test_images_replaced_with_placeholder_no_user(self):
        """Without a user, images become [Image N] placeholders (no base64)."""
        docx_bytes = _docx_with_image()
        text = extract_docx_text(docx_bytes)
        self.assertIn("[Image 1]", text)
        self.assertNotIn("data:image", text)
        self.assertNotIn("base64", text)


class LoggerTruncationTests(TestCase):
    def test_truncate_document_block(self):
        from llm.service.logger import _truncate_base64_in_content

        long_data = "x" * 1000
        blocks = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": long_data}}]
        result = _truncate_base64_in_content(blocks)
        self.assertIn("1000 chars", result[0]["source"]["data"])

    def test_truncate_file_block(self):
        from llm.service.logger import _truncate_base64_in_content

        long_data = "data:application/pdf;base64," + "x" * 1000
        blocks = [{"type": "file", "file": {"filename": "test.pdf", "file_data": long_data}}]
        result = _truncate_base64_in_content(blocks)
        self.assertIn("chars", result[0]["file"]["file_data"])

    def test_short_document_block_untouched(self):
        from llm.service.logger import _truncate_base64_in_content

        blocks = [{"type": "document", "source": {"type": "base64", "data": "short"}}]
        result = _truncate_base64_in_content(blocks)
        self.assertEqual(result[0]["source"]["data"], "short")
