"""Tests for canvas ingress: PDF/text file import, and the agent tools
canvas_paste_user_text and chat_attachment_open_to_canvas (plus the context
manifests and listing helpers that feed them)."""

from __future__ import annotations

import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from chat.canvas_tools import PasteUserTextTool
from chat.models import ChatAttachment, ChatCanvas, ChatMessage, ChatThread
from chat.services import (
    CANVAS_MAX_CHARS,
    build_canvas_ingress_manifests,
    import_file_to_canvas,
    list_pasteable_user_messages,
    list_thread_attachments,
)
from chat.tools import AttachmentOpenToCanvasTool
from llm.types.context import RunContext

User = get_user_model()

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_IN_MEMORY_STORAGE = override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)


def _ctx(user_id, thread_id):
    return RunContext.create(user_id=user_id, conversation_id=str(thread_id))


def _invoke(tool_cls, args, ctx):
    tool = tool_cls()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


# ---------------------------------------------------------------------------
# import_file_to_canvas (service)
# ---------------------------------------------------------------------------

class ImportFileToCanvasTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ing@test.com", password="pass")

    def test_text_file_is_decoded(self):
        f = SimpleUploadedFile("notes.txt", b"hello world", content_type="text/plain")
        title, content, truncated = import_file_to_canvas(f, self.user)
        self.assertEqual(title, "notes")
        self.assertEqual(content, "hello world")
        self.assertFalse(truncated)

    def test_text_file_truncated_to_cap(self):
        big = b"x" * (CANVAS_MAX_CHARS + 50)
        f = SimpleUploadedFile("big.txt", big, content_type="text/plain")
        _title, content, truncated = import_file_to_canvas(f, self.user)
        self.assertTrue(truncated)
        self.assertEqual(len(content), CANVAS_MAX_CHARS)

    @patch("core.pdf.pdf_to_text", return_value="PDF BODY TEXT")
    def test_pdf_uses_pdf_extractor(self, mock_pdf):
        f = SimpleUploadedFile("doc.pdf", b"%PDF-1.4 fake", content_type="application/pdf")
        title, content, truncated = import_file_to_canvas(f, self.user)
        self.assertEqual(title, "doc")
        self.assertEqual(content, "PDF BODY TEXT")
        self.assertFalse(truncated)
        mock_pdf.assert_called_once()

    @patch("chat.services.import_docx_to_canvas", return_value=("d", "# Docx", False))
    def test_docx_delegates_to_docx_importer(self, mock_docx):
        f = SimpleUploadedFile("d.docx", b"PK fake", content_type=_DOCX_MIME)
        title, content, _truncated = import_file_to_canvas(f, self.user)
        self.assertEqual((title, content), ("d", "# Docx"))
        mock_docx.assert_called_once()


# ---------------------------------------------------------------------------
# canvas_import view — widened to PDF/text
# ---------------------------------------------------------------------------

class CanvasImportViewWidenedTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ingview@test.com", password="pass")
        self.client.login(email="ingview@test.com", password="pass")
        # Non-empty title so the view skips LLM title generation.
        self.thread = ChatThread.objects.create(created_by=self.user, title="T")

    def _url(self):
        return f"/chat/threads/{self.thread.id}/canvas/import/"

    def test_text_import_creates_canvas(self):
        f = SimpleUploadedFile("brief.txt", b"# Brief\n\nBody.", content_type="text/plain")
        resp = self.client.post(self._url(), {"file": f}, format="multipart")
        self.assertEqual(resp.status_code, 200)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="brief")
        self.assertIn("Body.", canvas.content)

    @patch("chat.services.import_file_to_canvas", return_value=("report", "PDF TEXT", False))
    def test_pdf_import_accepted(self, _mock):
        f = SimpleUploadedFile("report.pdf", b"%PDF fake", content_type="application/pdf")
        resp = self.client.post(self._url(), {"file": f}, format="multipart")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(ChatCanvas.objects.filter(thread=self.thread, title="report").exists())

    def test_pptx_rejected(self):
        f = SimpleUploadedFile(
            "deck.pptx", b"PK fake",
            content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
        resp = self.client.post(self._url(), {"file": f}, format="multipart")
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(ChatCanvas.objects.filter(thread=self.thread).exists())

    @patch("chat.services.import_file_to_canvas", return_value=("scan", "   ", False))
    def test_blank_extraction_returns_400_and_no_canvas(self, _mock):
        f = SimpleUploadedFile("scan.pdf", b"%PDF fake", content_type="application/pdf")
        resp = self.client.post(self._url(), {"file": f}, format="multipart")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("no extractable text", resp.json()["error"])
        self.assertFalse(ChatCanvas.objects.filter(thread=self.thread).exists())


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------

@_IN_MEMORY_STORAGE
class ListingHelperTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="list@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_user_message_numbering_excludes_noise(self):
        ChatMessage.objects.create(thread=self.thread, role="user", content="first")
        ChatMessage.objects.create(thread=self.thread, role="assistant", content="reply")
        ChatMessage.objects.create(thread=self.thread, role="user", content="second")
        ChatMessage.objects.create(thread=self.thread, role="user", content="redacted", is_redacted=True)
        ChatMessage.objects.create(thread=self.thread, role="user", content="hidden", is_hidden_from_user=True)
        ChatMessage.objects.create(thread=self.thread, role="user", content="uionly", metadata={"ui_only": True})

        pairs = list_pasteable_user_messages(str(self.thread.id))
        self.assertEqual([(n, m.content) for n, m in pairs], [(1, "first"), (2, "second")])

    def test_attachment_numbering(self):
        for name in ("a.txt", "b.pdf"):
            ChatAttachment.objects.create(
                thread=self.thread, uploaded_by=self.user,
                file=SimpleUploadedFile(name, b"x"), original_filename=name,
                content_type="text/plain", size_bytes=1,
            )
        pairs = list_thread_attachments(str(self.thread.id))
        self.assertEqual([(n, a.original_filename) for n, a in pairs], [(1, "a.txt"), (2, "b.pdf")])


@_IN_MEMORY_STORAGE
class BuildCanvasIngressManifestsTests(TestCase):
    """Regression: manifests must be gated on tool-NAME strings (what the consumer
    passes), not tool objects — the latter silently dropped both manifests."""

    def setUp(self):
        self.user = User.objects.create_user(email="man@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        ChatMessage.objects.create(thread=self.thread, role="user", content="hello there")
        ChatAttachment.objects.create(
            thread=self.thread, uploaded_by=self.user,
            file=SimpleUploadedFile("deck.pdf", b"%PDF"), original_filename="deck.pdf",
            content_type="application/pdf", size_bytes=4,
        )
        ChatAttachment.objects.create(
            thread=self.thread, uploaded_by=self.user,
            file=SimpleUploadedFile("pic.png", b"\x89PNG"), original_filename="pic.png",
            content_type="image/png", size_bytes=4,
        )

    def test_both_manifests_when_both_tools_selected(self):
        pasteable, attachments = build_canvas_ingress_manifests(
            str(self.thread.id),
            {"canvas_write", "canvas_paste_user_text", "chat_attachment_open_to_canvas"},
        )
        self.assertEqual(pasteable["messages"], [{"number": 1, "preview": "hello there"}])
        self.assertEqual(
            [(a["number"], a["filename"], a["is_image"]) for a in attachments["items"]],
            [(1, "deck.pdf", False), (2, "pic.png", True)],
        )
        self.assertTrue(attachments["can_open_to_canvas"])
        self.assertFalse(attachments["can_view"])

    def test_attachment_manifest_gated_on_view_tool_too(self):
        _, attachments = build_canvas_ingress_manifests(
            str(self.thread.id), {"chat_attachment_view"},
        )
        self.assertTrue(attachments["can_view"])
        self.assertFalse(attachments["can_open_to_canvas"])
        self.assertEqual(attachments["items"][0]["kind_label"], "PDF")
        self.assertEqual(attachments["items"][1]["kind_label"], "image")

    def test_draft_and_redacted_flags(self):
        # Both setUp attachments are unsent drafts (message=NULL).
        _, attachments = build_canvas_ingress_manifests(
            str(self.thread.id), {"chat_attachment_view"},
        )
        self.assertFalse(attachments["items"][0]["is_sent"])
        redacted = ChatMessage.objects.create(
            thread=self.thread, role="user", content="x", is_redacted=True,
        )
        ChatAttachment.objects.filter(original_filename="deck.pdf").update(message=redacted)
        _, attachments = build_canvas_ingress_manifests(
            str(self.thread.id), {"chat_attachment_view"},
        )
        self.assertTrue(attachments["items"][0]["is_sent"])
        self.assertTrue(attachments["items"][0]["is_redacted"])
        self.assertFalse(attachments["items"][1]["is_redacted"])

    def test_no_manifests_when_tools_absent(self):
        pasteable, attachments = build_canvas_ingress_manifests(
            str(self.thread.id), {"canvas_write", "web_search"},
        )
        self.assertIsNone(pasteable)
        self.assertIsNone(attachments)

    def test_each_manifest_gated_independently(self):
        pasteable, attachments = build_canvas_ingress_manifests(
            str(self.thread.id), {"canvas_paste_user_text"},
        )
        self.assertIsNotNone(pasteable)
        self.assertIsNone(attachments)


# ---------------------------------------------------------------------------
# canvas_paste_user_text tool
# ---------------------------------------------------------------------------

class PasteUserTextToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="paste@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.m1 = ChatMessage.objects.create(thread=self.thread, role="user", content="First message")
        self.m2 = ChatMessage.objects.create(thread=self.thread, role="user", content="Second message")

    def _ctx(self):
        return _ctx(self.user.pk, self.thread.id)

    def test_paste_auto_creates_canvas(self):
        result = _invoke(
            PasteUserTextTool,
            {"message_number": 2, "canvas_name": "Draft"},
            self._ctx(),
        )
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["inserted_after_anchor"])
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Draft")
        self.assertEqual(canvas.content, "Second message")

    def test_paste_appends_to_existing_canvas(self):
        ChatCanvas.objects.create(thread=self.thread, title="Draft", content="Existing")
        result = _invoke(
            PasteUserTextTool, {"message_number": 1, "canvas_name": "Draft"}, self._ctx()
        )
        self.assertEqual(result["status"], "ok")
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Draft")
        self.assertEqual(canvas.content, "Existing\n\nFirst message")

    def test_paste_after_anchor(self):
        ChatCanvas.objects.create(
            thread=self.thread, title="Draft", content="Intro. Background: rest."
        )
        result = _invoke(
            PasteUserTextTool,
            {"message_number": 1, "canvas_name": "Draft", "anchor": "Background:"},
            self._ctx(),
        )
        self.assertTrue(result["inserted_after_anchor"])
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Draft")
        self.assertEqual(canvas.content, "Intro. Background:\n\nFirst message rest.")

    def test_missing_anchor_falls_back_and_notes(self):
        ChatCanvas.objects.create(thread=self.thread, title="Draft", content="Existing")
        result = _invoke(
            PasteUserTextTool,
            {"message_number": 1, "canvas_name": "Draft", "anchor": "nope"},
            self._ctx(),
        )
        self.assertFalse(result["inserted_after_anchor"])
        self.assertIn("note", result)

    def test_out_of_range_message_number_errors(self):
        result = _invoke(PasteUserTextTool, {"message_number": 99}, self._ctx())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["latest_message_number"], 2)

    def test_no_context_errors(self):
        tool = PasteUserTextTool()
        result = json.loads(tool.invoke({"message_number": 1}))
        self.assertEqual(result["status"], "error")

    def test_truncates_to_cap(self):
        ChatMessage.objects.create(
            thread=self.thread, role="user", content="y" * (CANVAS_MAX_CHARS + 100)
        )
        result = _invoke(
            PasteUserTextTool, {"message_number": 3, "canvas_name": "Big"}, self._ctx()
        )
        self.assertTrue(result.get("truncated"))
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Big")
        self.assertEqual(len(canvas.content), CANVAS_MAX_CHARS)


# ---------------------------------------------------------------------------
# chat_attachment_open_to_canvas tool
# ---------------------------------------------------------------------------

@_IN_MEMORY_STORAGE
class AttachmentOpenToCanvasToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="att@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _ctx(self):
        return _ctx(self.user.pk, self.thread.id)

    def _make(self, name, content_type, *, body=b"x", extracted=""):
        return ChatAttachment.objects.create(
            thread=self.thread, uploaded_by=self.user,
            file=SimpleUploadedFile(name, body), original_filename=name,
            content_type=content_type, size_bytes=len(body),
            extracted_content=extracted,
        )

    def test_text_attachment_decoded_into_canvas(self):
        self._make("notes.txt", "text/plain", body=b"line1\nline2")
        result = _invoke(AttachmentOpenToCanvasTool, {"attachment_number": 1}, self._ctx())
        self.assertEqual(result["status"], "ok")
        canvas = ChatCanvas.objects.get(thread=self.thread, title="notes")
        self.assertEqual(canvas.content, "line1\nline2")

    def test_docx_uses_cached_extracted_content(self):
        self._make("d.docx", _DOCX_MIME, body=b"PK", extracted="# Cached body")
        result = _invoke(AttachmentOpenToCanvasTool, {"attachment_number": 1}, self._ctx())
        self.assertEqual(result["status"], "ok")
        canvas = ChatCanvas.objects.get(thread=self.thread, title="d")
        self.assertEqual(canvas.content, "# Cached body")

    def test_image_attachment_refused(self):
        self._make("pic.png", "image/png", body=b"\x89PNG")
        result = _invoke(AttachmentOpenToCanvasTool, {"attachment_number": 1}, self._ctx())
        self.assertIn("error", result)
        self.assertIn("image", result["error"].lower())
        self.assertFalse(ChatCanvas.objects.filter(thread=self.thread).exists())

    def test_missing_attachment_lists_available(self):
        self._make("notes.txt", "text/plain")
        result = _invoke(AttachmentOpenToCanvasTool, {"attachment_number": 5}, self._ctx())
        self.assertIn("error", result)
        self.assertEqual(result["available_attachments"][0]["filename"], "notes.txt")


# ---------------------------------------------------------------------------
# build_dynamic_context manifests
# ---------------------------------------------------------------------------

class DynamicContextManifestTests(TestCase):
    def test_manifests_render_when_provided(self):
        from chat.prompts import build_dynamic_context

        out = build_dynamic_context(
            pasteable_messages={
                "messages": [{"number": 1, "preview": "hello there"}],
                "total": 1, "omitted": 0,
            },
            attachments={
                "items": [
                    {"number": 1, "filename": "a.pdf", "kind": "pdf", "kind_label": "PDF, 3 pages",
                     "size_bytes": 2048, "is_image": False, "is_sent": True, "is_redacted": False},
                    {"number": 2, "filename": "p.png", "kind": "image", "kind_label": "image",
                     "size_bytes": 1024, "is_image": True, "is_sent": False, "is_redacted": False},
                    {"number": 3, "filename": "gone.txt", "kind": "text", "kind_label": "text file",
                     "size_bytes": 10, "is_image": False, "is_sent": True, "is_redacted": True},
                ],
                "can_view": True,
                "can_open_to_canvas": False,
            },
        )
        self.assertIn("# Your messages", out)
        self.assertIn("canvas_paste_user_text", out)
        self.assertIn("# Attachments", out)
        self.assertIn("Files attached to the current message are shown to you automatically", out)
        self.assertIn("1. a.pdf (PDF, 3 pages, 2 KB)", out)
        self.assertIn("2. p.png (image, 1 KB) — not yet sent", out)
        self.assertIn("3. gone.txt (text file, 0 KB) — removed", out)
        self.assertIn("chat_attachment_view", out)
        self.assertNotIn("chat_attachment_open_to_canvas", out)
        self.assertNotIn("[[image:uuid]]", out)

    def test_manifests_absent_when_none(self):
        from chat.prompts import build_dynamic_context

        out = build_dynamic_context()
        self.assertNotIn("# Your messages", out)
        self.assertNotIn("# Attachments", out)
