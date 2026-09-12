"""_enrich_with_attachments metering: native tagging, budget degrade, and the
oversized-PDF → extracted-text fallback (Anthropic 32 MB request guard)."""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase, override_settings

from channels.db import database_sync_to_async

from chat.consumers import ChatConsumer
from chat.models import ChatAttachment, ChatThread
from llm.types.context import RunContext
from llm.types.messages import Message

User = get_user_model()

_PDF_MIME = "application/pdf"


@override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
class EnrichAttachmentBudgetTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="budget@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _consumer(self):
        c = ChatConsumer()
        c.scope = {"user": self.user}
        c.user = self.user
        return c

    async def _make_att(self, data, content_type, name="f"):
        return await database_sync_to_async(ChatAttachment.objects.create)(
            thread=self.thread,
            uploaded_by=self.user,
            file=SimpleUploadedFile(name, data, content_type=content_type),
            original_filename=name,
            content_type=content_type,
            size_bytes=len(data),
        )

    def _msgs_and_history(self, att):
        history = [{"role": "user", "content": "see attached",
                    "attachment_ids": [str(att.id)]}]
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="see attached"),
        ]
        return messages, history

    async def test_image_attachment_metered_and_tagged(self):
        att = await self._make_att(b"x" * 500, "image/png", "pic.png")
        messages, history = self._msgs_and_history(att)
        ctx = RunContext.create()
        c = self._consumer()
        with patch("llm.display.supports_modality", return_value=True), \
             patch("chat.services.detect_provider", return_value="anthropic"):
            await c._enrich_with_attachments(messages, history, "test-model", ctx)

        blocks = messages[1].content
        native = [b for b in blocks if isinstance(b, dict) and b.get("type") == "image"]
        self.assertEqual(len(native), 1)
        self.assertEqual(native[0]["_wf_pathway"], "attachment")
        self.assertGreater(native[0]["_wf_b64len"], 0)
        # Budget was consumed on the attachment pathway.
        self.assertGreater(ctx._b64_used_by_pathway.get("attachment", 0), 0)

    @override_settings(NATIVE_ASSET_BUDGET_B64_BYTES=10)
    async def test_image_over_budget_degrades_to_note(self):
        att = await self._make_att(b"x" * 5000, "image/png", "big.png")
        messages, history = self._msgs_and_history(att)
        ctx = RunContext.create()  # pool is only 10 bytes → reserve fails
        c = self._consumer()
        with patch("llm.display.supports_modality", return_value=True), \
             patch("chat.services.detect_provider", return_value="anthropic"):
            await c._enrich_with_attachments(messages, history, "test-model", ctx)

        blocks = messages[1].content
        self.assertFalse(any(b.get("type") == "image" for b in blocks if isinstance(b, dict)))
        note = [b for b in blocks if isinstance(b, dict) and b.get("type") == "text"
                and "budget" in b.get("text", "").lower()]
        self.assertEqual(len(note), 1)

    @override_settings(NATIVE_REQUEST_MAX_B64_BYTES_ANTHROPIC=100)
    async def test_oversized_pdf_degrades_to_extracted_text(self):
        """A PDF whose base64 exceeds the provider request ceiling falls back to
        extracted text instead of a native document block (the 32 MB 400 fix)."""
        att = await self._make_att(b"%PDF-1.4\n" + b"x" * 4000, _PDF_MIME, "big.pdf")
        messages, history = self._msgs_and_history(att)
        ctx = RunContext.create()
        c = self._consumer()
        with patch("llm.display.supports_modality", return_value=True), \
             patch("chat.services.detect_provider", return_value="anthropic"), \
             patch("chat.services.get_or_extract_attachment_text", return_value="EXTRACTED-BODY"):
            await c._enrich_with_attachments(messages, history, "test-model", ctx)

        blocks = messages[1].content
        self.assertFalse(any(b.get("type") in ("document", "file")
                             for b in blocks if isinstance(b, dict)))
        text_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        self.assertTrue(any("EXTRACTED-BODY" in b.get("text", "") for b in text_blocks))
        # Nothing reserved on the attachment pathway (degraded before reserve).
        self.assertEqual(ctx._b64_used_by_pathway.get("attachment", 0), 0)

    async def test_pdf_within_limits_is_native_and_tagged(self):
        att = await self._make_att(b"%PDF-1.4\n" + b"x" * 2000, _PDF_MIME, "ok.pdf")
        messages, history = self._msgs_and_history(att)
        ctx = RunContext.create()
        c = self._consumer()
        with patch("llm.display.supports_modality", return_value=True), \
             patch("chat.services.detect_provider", return_value="anthropic"), \
             patch("chat.services.get_or_extract_attachment_text", return_value="body"):
            await c._enrich_with_attachments(messages, history, "test-model", ctx)

        blocks = messages[1].content
        native = [b for b in blocks if isinstance(b, dict) and b.get("type") == "document"]
        self.assertEqual(len(native), 1)
        self.assertEqual(native[0]["_wf_pathway"], "attachment")
