"""chat_attachment_view: resolve a chat attachment by number and show it to the
model natively (images/PDFs) or as extracted text, stating which form it got."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, TransactionTestCase, override_settings

from chat.attachment_tools import ATTACHMENT_VIEW_TEXT_CAP, AttachmentViewTool
from chat.consumers import ChatConsumer
from chat.models import ChatAttachment, ChatMessage, ChatThread
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


def _pdf_bytes(pages: int = 2) -> bytes:
    """A real (image-only) multi-page PDF generated with Pillow."""
    from PIL import Image

    imgs = [Image.new("RGB", (200, 260), (255, 255, 255)) for _ in range(pages)]
    buf = io.BytesIO()
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@_IN_MEMORY_STORAGE
class AttachmentViewToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="view@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.msg = ChatMessage.objects.create(thread=self.thread, role="user", content="here")

    def _ctx(self, model_id=None):
        ctx = RunContext.create(user_id=self.user.pk, conversation_id=str(self.thread.id))
        ctx.model_id = model_id
        return ctx

    def _make(self, name, content_type, body=b"x", *, sent=True, extracted=""):
        return ChatAttachment.objects.create(
            thread=self.thread, uploaded_by=self.user,
            message=self.msg if sent else None,
            file=SimpleUploadedFile(name, body), original_filename=name,
            content_type=content_type, size_bytes=len(body),
            extracted_content=extracted,
        )

    def _view(self, args, ctx=None):
        ctx = ctx or self._ctx()
        tool = AttachmentViewTool()
        tool.set_context(ctx)
        return json.loads(tool.invoke(args)), ctx

    # --- metadata ---------------------------------------------------------

    def test_is_always_on_main_only(self):
        tool = AttachmentViewTool()
        self.assertEqual(tool.section, "chat")
        self.assertEqual(tool.audience, "main")
        self.assertIn("only visible to you during the reply", tool.description)

    # --- resolution / refusals ---------------------------------------------

    def test_missing_number_lists_available(self):
        self._make("notes.txt", "text/plain")
        result, _ = self._view({"attachment_number": 9})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["available_attachments"], [{"number": 1, "filename": "notes.txt"}])

    def test_draft_attachment_refused(self):
        self._make("draft.txt", "text/plain", sent=False)
        result, _ = self._view({"attachment_number": 1})
        self.assertEqual(result["status"], "error")
        self.assertIn("draft", result["message"])

    def test_redacted_attachment_refused(self):
        self.msg.is_redacted = True
        self.msg.save(update_fields=["is_redacted"])
        self._make("x.txt", "text/plain")
        result, _ = self._view({"attachment_number": 1})
        self.assertEqual(result["status"], "error")
        self.assertIn("removed", result["message"])

    def test_numbering_matches_open_to_canvas(self):
        self._make("a.txt", "text/plain", b"AAA")
        self._make("b.txt", "text/plain", b"BBB")
        result, _ = self._view({"attachment_number": 2})
        self.assertEqual(result["filename"], "b.txt")
        tool = AttachmentOpenToCanvasTool()
        tool.set_context(self._ctx())
        canvas_result = json.loads(tool.invoke({"attachment_number": 2}))
        self.assertEqual(canvas_result["filename"], "b.txt")

    # --- images -------------------------------------------------------------

    def test_image_is_native(self):
        self._make("pic.png", "image/png", _png_bytes())
        result, ctx = self._view({"attachment_number": 1, "mode": "extracted"})
        self.assertEqual(result["representation"], "native")
        self.assertIn("always shown natively", result["note"])
        self.assertEqual(len(ctx.pending_native_assets), 1)
        self.assertEqual(ctx.pending_native_assets[0]["_pathway"], "attachment")

    def test_image_on_non_vision_model_is_error(self):
        self._make("pic.png", "image/png", _png_bytes())
        with patch("llm.display.supports_modality", return_value=False):
            result, ctx = self._view({"attachment_number": 1}, self._ctx(model_id="text-only"))
        self.assertEqual(result["status"], "error")
        self.assertIn("cannot view images", result["message"])
        self.assertEqual(ctx.pending_native_assets, [])

    @override_settings(NATIVE_ASSET_BUDGET_B64_BYTES=10)
    def test_image_over_budget_is_error(self):
        self._make("pic.png", "image/png", _png_bytes())
        result, _ = self._view({"attachment_number": 1})
        self.assertEqual(result["status"], "error")
        self.assertIn("budget", result["message"])

    # --- PDFs ---------------------------------------------------------------

    def test_pdf_native_with_page_count(self):
        att = self._make("deck.pdf", "application/pdf", _pdf_bytes(2), extracted="BODY")
        result, ctx = self._view({"attachment_number": 1})
        self.assertEqual(result["representation"], "native")
        self.assertEqual(result["total_pages"], 2)
        item = ctx.pending_native_assets[0]
        self.assertEqual(item["kind"], "pdf")
        self.assertEqual(item["pages"], 2)
        self.assertEqual(item["_pathway"], "attachment")
        att.refresh_from_db()
        self.assertEqual(att.page_count, 2)

    def test_pdf_page_selection_native(self):
        self._make("deck.pdf", "application/pdf", _pdf_bytes(3), extracted="BODY")
        result, ctx = self._view({"attachment_number": 1, "pages": "2-3"})
        self.assertEqual(result["representation"], "native")
        self.assertEqual(result["pages"], "2-3")
        self.assertEqual(ctx.pending_native_assets[0]["pages"], 2)

    @override_settings(NATIVE_REQUEST_MAX_PDF_PAGES=1)
    def test_pdf_over_page_cap_attaches_page_images(self):
        self._make("deck.pdf", "application/pdf", _pdf_bytes(2), extracted="BODY TEXT")
        result, ctx = self._view({"attachment_number": 1})
        self.assertEqual(result["representation"], "native_pages_as_images")
        self.assertEqual(result["pages_attached"], 2)
        self.assertIn("too many pages", result["reason"])
        self.assertIn("BODY TEXT", result["content"])
        self.assertTrue(all(i["kind"] == "image" for i in ctx.pending_native_assets))

    @override_settings(NATIVE_ASSET_BUDGET_B64_BYTES=10)
    def test_pdf_budget_floor_is_extracted_with_reason(self):
        self._make("deck.pdf", "application/pdf", _pdf_bytes(2), extracted="BODY TEXT")
        result, ctx = self._view({"attachment_number": 1})
        self.assertEqual(result["representation"], "extracted")
        self.assertIn("budget", result["reason"])
        self.assertEqual(result["content"], "BODY TEXT")
        self.assertEqual(ctx.pending_native_assets, [])

    def test_pdf_non_pdf_model_is_extracted_with_reason(self):
        self._make("deck.pdf", "application/pdf", _pdf_bytes(1), extracted="BODY")
        with patch("llm.display.supports_modality", return_value=False):
            result, ctx = self._view({"attachment_number": 1}, self._ctx(model_id="text-only"))
        self.assertEqual(result["representation"], "extracted")
        self.assertIn("cannot view PDFs", result["reason"])
        self.assertEqual(ctx.pending_native_assets, [])

    def test_pdf_extracted_mode_pages_are_page_scoped(self):
        self._make("deck.pdf", "application/pdf", _pdf_bytes(3), extracted="FULL")
        result, ctx = self._view({"attachment_number": 1, "mode": "extracted", "pages": "2"})
        self.assertEqual(result["representation"], "extracted")
        self.assertEqual(result["pages"], "2")
        self.assertIn("--- page 2 ---", result["content"])
        self.assertNotIn("--- page 1 ---", result["content"])
        self.assertEqual(ctx.pending_native_assets, [])

    # --- extracted text / paging ------------------------------------------

    def test_extracted_text_pages_with_char_offset(self):
        body = ("a" * ATTACHMENT_VIEW_TEXT_CAP + "TAIL").encode()
        self._make("long.txt", "text/plain", body)
        first, _ = self._view({"attachment_number": 1})
        self.assertEqual(first["representation"], "extracted")
        self.assertTrue(first["truncated"])
        self.assertEqual(first["next_char_offset"], ATTACHMENT_VIEW_TEXT_CAP)
        second, _ = self._view({"attachment_number": 1, "char_offset": first["next_char_offset"]})
        self.assertEqual(second["content"], "TAIL")
        self.assertFalse(second["truncated"])

    def test_char_offset_past_end_is_error(self):
        self._make("s.txt", "text/plain", b"short")
        result, _ = self._view({"attachment_number": 1, "char_offset": 999})
        self.assertEqual(result["status"], "error")

    def test_docx_native_mode_explains_fallback(self):
        self._make("d.docx", _DOCX_MIME, b"PK", extracted="# Cached body")
        result, _ = self._view({"attachment_number": 1, "mode": "native"})
        self.assertEqual(result["representation"], "extracted")
        self.assertEqual(result["content"], "# Cached body")
        self.assertIn("can't be shown natively", result["reason"])

    # --- labels ---------------------------------------------------------------

    def test_end_labels(self):
        tool = AttachmentViewTool()
        self.assertEqual(
            tool.end_label_for_result({"status": "ok", "filename": "t.pdf", "representation": "native", "pages": "1-5"}),
            "Viewed t.pdf (native, p. 1–5)",
        )
        self.assertEqual(
            tool.end_label_for_result({"status": "ok", "filename": "t.pdf", "representation": "native"}),
            "Viewed t.pdf (native)",
        )
        self.assertEqual(
            tool.end_label_for_result({
                "status": "ok", "filename": "t.pdf", "representation": "native_pages_as_images",
                "pages_attached": 20, "total_pages": 140,
            }),
            "Viewed t.pdf (first 20 of 140 pages as images)",
        )
        self.assertEqual(
            tool.end_label_for_result({"status": "ok", "filename": "d.docx", "representation": "extracted"}),
            "Read d.docx (extracted text)",
        )
        self.assertEqual(
            tool.end_label_for_result({"status": "error", "message": "x"}),
            "Couldn't view attachment",
        )


class AttachmentViewGatingTests(TransactionTestCase):
    """The tool is offered only when the thread has a sent attachment."""

    def setUp(self):
        self.user = User.objects.create_user(email="gate@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _consumer(self):
        c = ChatConsumer()
        c.scope = {"user": self.user}
        c.user = self.user
        c.resolved_prefs = None
        c.data_room_ids = []
        c.active_skill_ids = []
        return c

    async def test_hidden_without_sent_attachment(self):
        await database_sync_to_async(ChatAttachment.objects.create)(
            thread=self.thread, uploaded_by=self.user,
            file=SimpleUploadedFile("d.txt", b"x"), original_filename="d.txt",
            content_type="text/plain", size_bytes=1,
        )  # draft only
        tools, _ = await self._consumer()._resolve_selected_tools(
            is_loop_turn=False, thread_id=str(self.thread.id)
        )
        self.assertNotIn("chat_attachment_view", tools)

    async def test_offered_with_sent_attachment(self):
        msg = await database_sync_to_async(ChatMessage.objects.create)(
            thread=self.thread, role="user", content="hi",
        )
        await database_sync_to_async(ChatAttachment.objects.create)(
            thread=self.thread, uploaded_by=self.user, message=msg,
            file=SimpleUploadedFile("d.txt", b"x"), original_filename="d.txt",
            content_type="text/plain", size_bytes=1,
        )
        tools, _ = await self._consumer()._resolve_selected_tools(
            is_loop_turn=False, thread_id=str(self.thread.id)
        )
        self.assertIn("chat_attachment_view", tools)


@_IN_MEMORY_STORAGE
class AttachmentViewVectorPagesTests(TestCase):
    """OpenAI models also get rendered images of vector-graphic pages."""

    def setUp(self):
        self.user = User.objects.create_user(email="vecview@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="here")
        from weasyprint import HTML

        chart = "".join(f'<rect x="{40 + i * 60}" y="40" width="40" height="160" fill="#36c"/>' for i in range(4))
        data = HTML(string=f"<html><body><h1>Sales</h1><svg width='320' height='240'>{chart}</svg></body></html>").write_pdf()
        ChatAttachment.objects.create(
            thread=self.thread, uploaded_by=self.user, message=msg,
            file=SimpleUploadedFile("chart.pdf", data), original_filename="chart.pdf",
            content_type="application/pdf", size_bytes=len(data), extracted_content="Sales",
        )

    def _view(self, model_id):
        ctx = RunContext.create(user_id=self.user.pk, conversation_id=str(self.thread.id))
        ctx.model_id = model_id
        tool = AttachmentViewTool()
        tool.set_context(ctx)
        return json.loads(tool.invoke({"attachment_number": 1})), ctx

    def test_openai_result_lists_page_images(self):
        result, ctx = self._view("openai/gpt-6-luna")
        self.assertEqual(result["representation"], "native")
        self.assertEqual(result["page_images"], [1])
        self.assertEqual(result["note"], "The PDF and rendered images of page 1 are attached below for you to view.")
        self.assertEqual([i["kind"] for i in ctx.pending_native_assets], ["pdf", "image"])

    def test_anthropic_result_has_no_page_images(self):
        result, ctx = self._view("anthropic/claude-haiku-4-5")
        self.assertNotIn("page_images", result)
        self.assertEqual(result["note"], "The PDF is attached below for you to view.")
        self.assertEqual([i["kind"] for i in ctx.pending_native_assets], ["pdf"])
