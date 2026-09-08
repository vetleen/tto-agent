"""Tests for same-turn native viewing: the pipeline injection helper and the
document_view_native tool (images + PDFs)."""

import tempfile

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from llm.pipelines.simple_chat import SimpleChatPipeline
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest

User = get_user_model()

_MEDIA = tempfile.mkdtemp()


def _req(model, pending):
    ctx = RunContext.create(user_id=1)
    ctx.pending_native_assets = pending
    return ChatRequest(
        messages=[Message(role="user", content="hi")],
        model=model,
        stream=False,
        tools=[],
        context=ctx,
    ), ctx


class AppendPendingNativeAssetsTests(TestCase):
    def test_vision_model_gets_native_image_block(self):
        req, ctx = _req("anthropic/claude-opus-4-8", [
            {"asset_id": "", "b64": "AAAA", "media_type": "image/png", "description": "a bar chart"},
        ])
        new_messages = []
        SimpleChatPipeline._append_pending_native_assets(new_messages, req)

        self.assertEqual(len(new_messages), 1)
        msg = new_messages[0]
        self.assertEqual(msg.role, "user")
        self.assertIsInstance(msg.content, list)
        self.assertTrue(any(isinstance(b, dict) and b.get("type") == "image" for b in msg.content))
        # Collector is drained so it isn't re-injected next iteration.
        self.assertEqual(ctx.pending_native_assets, [])

    def test_non_vision_model_gets_text_fallback(self):
        req, ctx = _req("openai/whisper-1", [
            {"asset_id": "", "b64": "AAAA", "media_type": "image/png", "description": "a bar chart"},
        ])
        new_messages = []
        SimpleChatPipeline._append_pending_native_assets(new_messages, req)

        self.assertEqual(len(new_messages), 1)
        msg = new_messages[0]
        self.assertFalse(any(isinstance(b, dict) and b.get("type") == "image" for b in msg.content))
        self.assertTrue(any("no vision" in b.get("text", "") for b in msg.content))

    def test_no_pending_is_noop(self):
        req, ctx = _req("anthropic/claude-opus-4-8", [])
        new_messages = []
        SimpleChatPipeline._append_pending_native_assets(new_messages, req)
        self.assertEqual(new_messages, [])

    def test_pdf_model_gets_native_document_block(self):
        req, ctx = _req("anthropic/claude-opus-4-8", [
            {"kind": "pdf", "b64": "AAAA", "filename": "report.pdf",
             "description": "q3 deck", "extracted_text": "fallback text"},
        ])
        new_messages = []
        SimpleChatPipeline._append_pending_native_assets(new_messages, req)
        msg = new_messages[0]
        self.assertTrue(any(isinstance(b, dict) and b.get("type") == "document" for b in msg.content))
        self.assertEqual(ctx.pending_native_assets, [])

    def test_non_pdf_model_falls_back_to_text(self):
        req, ctx = _req("openai/whisper-1", [
            {"kind": "pdf", "b64": "AAAA", "filename": "report.pdf",
             "description": "q3 deck", "extracted_text": "fallback text here"},
        ])
        new_messages = []
        SimpleChatPipeline._append_pending_native_assets(new_messages, req)
        msg = new_messages[0]
        self.assertFalse(any(isinstance(b, dict) and b.get("type") == "document" for b in msg.content))
        self.assertTrue(any("fallback text here" in b.get("text", "") for b in msg.content))


@override_settings(MEDIA_ROOT=_MEDIA)
class DocumentViewNativeToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="siv@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-siv", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="chart.png", mime_type="image/png",
            doc_index=1, status=DataRoomDocument.Status.READY,
        )
        version = DataRoomDocumentVersion.objects.create(
            document=self.doc, parser_type="image", mime_type="image/png",
            native_blob=ContentFile(b"\x89PNG fake-image", name="chart.png"),
        )
        self.doc.current_version = version
        self.doc.save(update_fields=["current_version"])

    def _tool(self, data_room_ids):
        from chat.tools import DocumentViewNativeTool

        tool = DocumentViewNativeTool()
        tool.set_context(RunContext.create(user_id=self.user.pk, data_room_ids=data_room_ids))
        return tool

    def test_attaches_image_as_document(self):
        tool = self._tool([self.room.pk])
        result = tool._run([1])
        self.assertIn("attached", result.lower())
        pending = tool.context.pending_native_assets
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["media_type"], "image/png")
        self.assertTrue(pending[0]["b64"])

    def test_result_surfaces_embed_token(self):
        # The model gets a reusable [[image:uuid|label]] token to embed in a
        # canvas or its reply, both in the result text and as asset_id metadata.
        tool = self._tool([self.room.pk])
        result = tool._run([1])
        self.assertIn("[[image:", result)
        self.assertTrue(tool.context.pending_native_assets[0]["asset_id"].startswith("[[image:"))

    def test_inaccessible_room_is_denied(self):
        other = User.objects.create_user(email="siv-other@test.com", password="pw")
        room2 = DataRoom.objects.create(name="R2", slug="r2-siv", created_by=other)
        tool = self._tool([room2.pk])  # this user does not own room2
        result = tool._run([1])
        self.assertEqual(tool.context.pending_native_assets, [])
        self.assertNotIn("attached", result.lower())

    def test_attaches_image_from_original_file_when_native_blob_empty(self):
        # Production shape: a freshly-uploaded image keeps its bytes on
        # doc.original_file with an EMPTY native_blob. document_view_native must still
        # find the image, not report "no viewable image".
        doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="photo.jpg", mime_type="image/jpeg",
            doc_index=2, status=DataRoomDocument.Status.READY,
        )
        doc.original_file.save("photo.jpg", ContentFile(b"\xff\xd8\xff real-jpeg"), save=True)
        version = DataRoomDocumentVersion.objects.create(
            document=doc, parser_type="image", mime_type="image/jpeg",
        )
        doc.current_version = version
        doc.save(update_fields=["current_version"])

        tool = self._tool([self.room.pk])
        result = tool._run([2])
        self.assertIn("attached", result.lower())
        pending = tool.context.pending_native_assets
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["media_type"], "image/jpeg")
        self.assertEqual(pending[0]["kind"], "image")
        self.assertTrue(pending[0]["b64"])

    def test_attaches_pdf_natively(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="report.pdf", mime_type="application/pdf",
            doc_index=3, status=DataRoomDocument.Status.READY,
        )
        version = DataRoomDocumentVersion.objects.create(
            document=doc, parser_type="pypdf", mime_type="application/pdf",
            native_blob=ContentFile(b"%PDF-1.4 fake-pdf", name="report.pdf"),
        )
        doc.current_version = version
        doc.save(update_fields=["current_version"])

        tool = self._tool([self.room.pk])
        result = tool._run([3])
        self.assertIn("attached", result.lower())
        pending = tool.context.pending_native_assets
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["kind"], "pdf")
        self.assertEqual(pending[0]["filename"], "report.pdf")
        self.assertTrue(pending[0]["b64"])


@override_settings(MEDIA_ROOT=_MEDIA)
class PdfAttachChainTests(TestCase):
    """The staged degrade chain for PDFs that bust the native-asset budget:
    as-is → lossless compress → page render → extracted-text floor."""

    def setUp(self):
        self.user = User.objects.create_user(email="pdfchain@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-pdfchain", created_by=self.user)
        self.pdf_bytes = b"%PDF-1.4 " + b"x" * 1000
        doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="big.pdf", mime_type="application/pdf",
            doc_index=1, status=DataRoomDocument.Status.READY,
        )
        version = DataRoomDocumentVersion.objects.create(
            document=doc, parser_type="pypdf", mime_type="application/pdf",
            native_blob=ContentFile(self.pdf_bytes, name="big.pdf"),
        )
        doc.current_version = version
        doc.save(update_fields=["current_version"])

    def _tool(self, remaining_budget):
        from llm.types.context import NATIVE_ASSET_BUDGET_B64_CHARS
        from chat.tools import DocumentViewNativeTool

        tool = DocumentViewNativeTool()
        ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[self.room.pk])
        ctx._native_asset_b64_used = NATIVE_ASSET_BUDGET_B64_CHARS - remaining_budget
        tool.set_context(ctx)
        return tool

    def test_compress_stage_attaches_when_it_fits(self):
        from unittest.mock import patch

        tool = self._tool(remaining_budget=300)  # original b64 (~1350) won't fit
        with patch("chat.pdf_attach.compress_pdf_lossless", return_value=b"tiny-pdf") as mock_c:
            result = tool._run([1])
        mock_c.assert_called_once()
        self.assertIn("losslessly compressed", result)
        pending = tool.context.pending_native_assets
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["kind"], "pdf")

    def test_render_stage_attaches_page_images_with_truncation_note(self):
        from unittest.mock import patch

        tool = self._tool(remaining_budget=300)
        with patch("chat.pdf_attach.compress_pdf_lossless", side_effect=lambda b: b), \
             patch("chat.pdf_attach.render_pdf_pages_to_jpegs",
                   return_value=([b"jpeg-one", b"jpeg-two"], 5)):
            result = tool._run([1])
        self.assertIn("attached the first 2 of 5 pages", result)
        self.assertIn("TRUNCATED", result)
        pending = tool.context.pending_native_assets
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0]["kind"], "image")
        self.assertEqual(pending[0]["media_type"], "image/jpeg")
        self.assertIn("page 1 of 5", pending[0]["description"])

    def test_floor_returns_extracted_text_only(self):
        from unittest.mock import patch

        tool = self._tool(remaining_budget=3)  # nothing fits, not even a page
        with patch("chat.pdf_attach.compress_pdf_lossless", side_effect=lambda b: b), \
             patch("chat.pdf_attach.render_pdf_pages_to_jpegs", return_value=([], 0)):
            result = tool._run([1])
        self.assertIn("attachment budget for this run is exhausted", result)
        self.assertIn("Extracted text", result)
        self.assertEqual(tool.context.pending_native_assets, [])

    def test_doc_image_rejected_when_budget_exhausted(self):
        from llm.types.context import NATIVE_ASSET_BUDGET_B64_CHARS
        from chat.tools import DocumentViewNativeTool

        img_doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="chart.png", mime_type="image/png",
            doc_index=2, status=DataRoomDocument.Status.READY,
        )
        version = DataRoomDocumentVersion.objects.create(
            document=img_doc, parser_type="image", mime_type="image/png",
            native_blob=ContentFile(b"\x89PNG fake-image", name="chart.png"),
        )
        img_doc.current_version = version
        img_doc.save(update_fields=["current_version"])

        tool = DocumentViewNativeTool()
        ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[self.room.pk])
        ctx._native_asset_b64_used = NATIVE_ASSET_BUDGET_B64_CHARS
        tool.set_context(ctx)
        result = tool._run([2])
        self.assertIn("attached 0 of 1", result)
        self.assertIn("attachment budget for this run is exhausted", result)
        self.assertEqual(tool.context.pending_native_assets, [])


@override_settings(MEDIA_ROOT=_MEDIA)
class AttachmentMarkersTests(TestCase):
    """The summariser surfaces shared-file markers so compression doesn't
    silently lose that the user shared an image."""

    def test_marks_messages_with_attachments(self):
        from django.core.files.base import ContentFile

        from chat.models import ChatAttachment, ChatMessage, ChatThread
        from chat.services import _attachment_markers

        user = User.objects.create_user(email="amk@test.com", password="pw")
        thread = ChatThread.objects.create(created_by=user)
        msg = ChatMessage.objects.create(thread=thread, role="user", content="look at this")
        att = ChatAttachment.objects.create(
            thread=thread, message=msg, uploaded_by=user,
            file=ContentFile(b"x", name="chart.png"),
            original_filename="chart.png", content_type="image/png", size_bytes=1,
        )
        msg.metadata = {"attachment_ids": [str(att.id)]}
        msg.save(update_fields=["metadata"])

        markers = _attachment_markers([msg])
        self.assertIn(str(msg.id), markers)
        self.assertIn("chart.png", markers[str(msg.id)])

    def test_no_attachments_is_empty(self):
        from chat.models import ChatMessage, ChatThread
        from chat.services import _attachment_markers

        user = User.objects.create_user(email="amk2@test.com", password="pw")
        thread = ChatThread.objects.create(created_by=user)
        msg = ChatMessage.objects.create(thread=thread, role="user", content="hi")
        self.assertEqual(_attachment_markers([msg]), {})
