"""Tests for chat/attachment_processing.py, the attachment-owned Asset arm, the
processing task's failure settling, and the attachment status endpoint."""
from __future__ import annotations

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError, transaction
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.assets import user_can_access_asset
from chat.attachment_processing import (
    dispatch_processing,
    initial_processing_state,
    mark_for_reprocessing,
    process_attachment,
)
from chat.models import Asset, ChatAttachment, ChatMessage, ChatThread
from chat.tests.test_attachments import _docx_with_image, _pdf_with_image, _tiny_docx, _tiny_png
from documents.services import page_render as pr
from documents.tests.test_page_render import _FakeConvert, _deck, _jpeg, _png

User = get_user_model()

_IN_MEMORY_STORAGE = override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
_PDF_MIME = "application/pdf"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

State = ChatAttachment.ProcessingState
Render = ChatAttachment.PageRenderState


def _make_attachment(thread, user, name, ct, body, *, state=State.PENDING, message=None):
    return ChatAttachment.objects.create(
        thread=thread,
        message=message,
        uploaded_by=user,
        file=SimpleUploadedFile(name, body, content_type=ct),
        original_filename=name,
        content_type=ct,
        size_bytes=len(body),
        processing_state=state,
    )


@_IN_MEMORY_STORAGE
class AssetAttachmentOwnerTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="owner@example.com", password="pw")
        self.other = User.objects.create_user(email="other@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.att = _make_attachment(self.thread, self.user, "a.pdf", _PDF_MIME, b"%PDF-1.4 x")

    def test_attachment_owned_asset_saves_and_follows_thread_access(self):
        asset = Asset.objects.create(attachment=self.att, content_type="image/png", size_bytes=4)
        self.assertEqual(asset.attachment_id, self.att.id)
        self.assertTrue(user_can_access_asset(self.user, asset))
        self.assertFalse(user_can_access_asset(self.other, asset))

    def test_two_owners_are_rejected(self):
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="hi")
        with self.assertRaises(IntegrityError), transaction.atomic():
            Asset.objects.create(attachment=self.att, message=msg, content_type="image/png")

    def test_duplicate_page_render_per_attachment_is_rejected(self):
        Asset.objects.create(attachment=self.att, role=Asset.ROLE_PAGE_RENDER, page_number=1, content_type="image/jpeg")
        with self.assertRaises(IntegrityError), transaction.atomic():
            Asset.objects.create(attachment=self.att, role=Asset.ROLE_PAGE_RENDER, page_number=1, content_type="image/jpeg")

    def test_deleting_the_attachment_removes_its_assets(self):
        Asset.objects.create(attachment=self.att, content_type="image/png")
        self.att.delete()
        self.assertEqual(Asset.objects.count(), 0)

    def test_initial_processing_state_by_type(self):
        self.assertEqual(initial_processing_state(_PDF_MIME), State.PENDING)
        self.assertEqual(initial_processing_state(_DOCX_MIME), State.PENDING)
        self.assertEqual(initial_processing_state(_PPTX_MIME), State.PENDING)
        self.assertEqual(initial_processing_state("image/png"), State.READY)
        self.assertEqual(initial_processing_state("text/plain"), State.READY)


@_IN_MEMORY_STORAGE
@override_settings(DOCUMENT_RENDER_SERVICE_URL="http://render.test", DOCUMENT_RENDER_BATCH_SIZE=8)
class ProcessAttachmentTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="proc@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _make(self, name, ct, body, **kw):
        return _make_attachment(self.thread, self.user, name, ct, body, **kw)

    def test_pdf_is_extracted_with_attachment_owned_images_and_page_count(self):
        att = self._make("r.pdf", _PDF_MIME, _pdf_with_image())
        with patch("chat.services.describe_image", return_value="A red rectangle"):
            self.assertEqual(process_attachment(str(att.id)), "ready")
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.READY)
        self.assertEqual(att.page_count, 1)
        assets = list(Asset.objects.filter(attachment=att, role=Asset.ROLE_EMBEDDED))
        self.assertEqual(len(assets), 1)
        self.assertIn(f"[[image:{assets[0].id}|", att.extracted_content)
        self.assertEqual(att.page_render_state, Render.NONE)

    def test_docx_is_extracted(self):
        att = self._make("n.docx", _DOCX_MIME, _tiny_docx())
        self.assertEqual(process_attachment(str(att.id)), "ready")
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.READY)
        self.assertIn("Hello World", att.extracted_content)

    def test_image_needs_no_work(self):
        att = self._make("p.png", "image/png", _tiny_png())
        self.assertEqual(process_attachment(str(att.id)), "ready")
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.READY)
        self.assertEqual(att.extracted_content, "")

    def test_pptx_extracts_text_and_renders_slides_owned_by_the_attachment(self):
        att = self._make("deck.pptx", _PPTX_MIME, _deck(3, notes=False))
        fake = _FakeConvert()
        with patch.object(pr.GotenbergClient, "convert", fake):
            self.assertEqual(process_attachment(str(att.id)), "ready")
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.READY)
        self.assertEqual(att.page_render_state, Render.READY)
        self.assertEqual(att.page_count, 3)
        self.assertIn("## Slide 1", att.extracted_content)
        renders = Asset.objects.filter(attachment=att, role=Asset.ROLE_PAGE_RENDER).order_by("page_number")
        self.assertEqual(list(renders.values_list("page_number", flat=True)), [1, 2, 3])
        self.assertTrue(all(a.blob for a in renders))
        # Opaque upload name: attachment trace + batch number only.
        self.assertEqual(fake.calls[0][2], f"a-{att.id}-1.pptx")

    def test_pptx_with_picture_stores_an_embedded_asset(self):
        att = self._make("deck.pptx", _PPTX_MIME, _deck(1, picture=_png(64, 64), notes=False))
        with patch("chat.services.describe_image", return_value="A red square"), \
             patch.object(pr.GotenbergClient, "convert", _FakeConvert()):
            process_attachment(str(att.id))
        att.refresh_from_db()
        embedded = Asset.objects.filter(attachment=att, role=Asset.ROLE_EMBEDDED)
        self.assertEqual(embedded.count(), 1)
        self.assertIn(f"[[image:{embedded.first().id}|", att.extracted_content)

    def test_pptx_over_the_slide_cap_is_skipped_without_contacting_the_service(self):
        att = self._make("deck.pptx", _PPTX_MIME, _deck(3, notes=False))
        fake = _FakeConvert()
        with override_settings(CHAT_ATTACHMENT_RENDER_MAX_SLIDES=2), patch.object(pr.GotenbergClient, "convert", fake):
            self.assertEqual(process_attachment(str(att.id)), "ready")
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.READY)
        self.assertEqual(att.page_render_state, Render.SKIPPED)
        self.assertEqual(att.page_count, 3)
        self.assertEqual(fake.calls, [])

    def test_pptx_with_render_service_off_is_ready_as_text(self):
        att = self._make("deck.pptx", _PPTX_MIME, _deck(2, notes=False))
        with override_settings(DOCUMENT_RENDER_SERVICE_URL=""), patch.object(pr.GotenbergClient, "convert", _FakeConvert()):
            self.assertEqual(process_attachment(str(att.id)), "ready")
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.READY)
        self.assertEqual(att.page_render_state, Render.FAILED)
        self.assertIn("not configured", att.processing_error)

    def test_busy_render_service_propagates_and_keeps_the_row_pending(self):
        att = self._make("deck.pptx", _PPTX_MIME, _deck(2, notes=False))
        fake = _FakeConvert(lambda *_: pr.RenderBusy("full"))
        with patch.object(pr.GotenbergClient, "convert", fake), patch.object(pr.time, "sleep"):
            with self.assertRaises(pr.RenderBusy):
                process_attachment(str(att.id))
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.PENDING)
        # Extraction is cached, so the retry only renders.
        self.assertIn("## Slide 1", att.extracted_content)

    def test_corrupt_file_fails_without_retry(self):
        att = self._make("bad.pdf", _PDF_MIME, b"%PDF-1.4 not really a pdf")
        self.assertEqual(process_attachment(str(att.id)), "failed")
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.FAILED)
        self.assertTrue(att.processing_error)

    def test_unreadable_file_fails(self):
        att = self._make("gone.pdf", _PDF_MIME, b"%PDF-1.4 x")
        att.file.storage.delete(att.file.name)
        self.assertEqual(process_attachment(str(att.id)), "failed")
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.FAILED)
        self.assertIn("could not be read", att.processing_error)

    def test_missing_and_already_ready_rows_are_noops(self):
        self.assertEqual(process_attachment("00000000-0000-0000-0000-000000000000"), "missing")
        att = self._make("r.pdf", _PDF_MIME, b"%PDF-1.4 x", state=State.READY)
        with patch("chat.services.get_or_extract_attachment_text") as extract:
            self.assertEqual(process_attachment(str(att.id)), "ready")
        extract.assert_not_called()


@_IN_MEMORY_STORAGE
class TaskSettleAndDispatchTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="settle@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _on_failure(self, att, exc):
        from chat.tasks import _AttachmentProcessingTask

        _AttachmentProcessingTask().on_failure(exc, "task-id", [str(att.id)], {}, None)

    def test_render_only_failure_leaves_the_file_usable_as_text(self):
        att = _make_attachment(self.thread, self.user, "d.pptx", _PPTX_MIME, b"PK")
        ChatAttachment.objects.filter(pk=att.pk).update(extracted_content="## Slide 1", page_render_state=Render.PENDING)
        self._on_failure(att, pr.RenderUnavailable("down"))
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.READY)
        self.assertEqual(att.page_render_state, Render.FAILED)
        self.assertIn("down", att.processing_error)

    def test_failure_before_extraction_marks_failed(self):
        att = _make_attachment(self.thread, self.user, "d.pdf", _PDF_MIME, b"%PDF")
        self._on_failure(att, RuntimeError("boom"))
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.FAILED)
        self.assertIn("boom", att.processing_error)

    def test_dispatch_success_queues_the_task(self):
        att = _make_attachment(self.thread, self.user, "d.pdf", _PDF_MIME, b"%PDF")
        with patch("chat.tasks.process_chat_attachment.delay") as delay:
            self.assertTrue(dispatch_processing(att.id))
        delay.assert_called_once_with(str(att.id))

    def test_dispatch_failure_marks_failed_so_the_turn_never_waits(self):
        att = _make_attachment(self.thread, self.user, "d.pdf", _PDF_MIME, b"%PDF")
        with patch("chat.tasks.process_chat_attachment.delay", side_effect=RuntimeError("broker down")):
            self.assertFalse(dispatch_processing(att.id))
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.FAILED)
        self.assertEqual(att.processing_error, "Could not queue processing.")

    def test_mark_for_reprocessing_resets_and_dispatches_after_commit(self):
        att = _make_attachment(self.thread, self.user, "d.pptx", _PPTX_MIME, b"PK", state=State.READY)
        ChatAttachment.objects.filter(pk=att.pk).update(page_render_state=Render.READY, extracted_content="## Slide 1")
        with patch("chat.tasks.process_chat_attachment.delay") as delay, self.captureOnCommitCallbacks(execute=True):
            mark_for_reprocessing(att)
        att.refresh_from_db()
        self.assertEqual(att.processing_state, State.PENDING)
        self.assertEqual(att.page_render_state, Render.NONE)
        self.assertEqual(att.extracted_content, "## Slide 1")
        delay.assert_called_once_with(str(att.id))


@_IN_MEMORY_STORAGE
class AttachmentStatusViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="status@example.com", password="pw")
        self.other = User.objects.create_user(email="nosy@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.att = _make_attachment(self.thread, self.user, "d.pdf", _PDF_MIME, b"%PDF")
        ChatAttachment.objects.filter(pk=self.att.pk).update(page_count=4, processing_error="")
        self.url = reverse("chat_attachment_status", args=[self.att.id])

    def test_owner_sees_the_state(self):
        self.client.force_login(self.user)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {
            "id": str(self.att.id), "processing_state": "pending", "page_render_state": "none",
            "page_count": 4, "error": "",
        })

    def test_other_user_gets_404(self):
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(self.url).status_code, 404)

    def test_anonymous_is_redirected(self):
        self.assertEqual(self.client.get(self.url).status_code, 302)
