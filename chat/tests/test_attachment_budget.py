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
            await c._enrich_with_attachments(
                messages, history, "test-model", ctx, thread_id=str(self.thread.id)
            )

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
            await c._enrich_with_attachments(
                messages, history, "test-model", ctx, thread_id=str(self.thread.id)
            )

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
            await c._enrich_with_attachments(
                messages, history, "test-model", ctx, thread_id=str(self.thread.id)
            )

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
            await c._enrich_with_attachments(
                messages, history, "test-model", ctx, thread_id=str(self.thread.id)
            )

        blocks = messages[1].content
        native = [b for b in blocks if isinstance(b, dict) and b.get("type") == "document"]
        self.assertEqual(len(native), 1)
        self.assertEqual(native[0]["_wf_pathway"], "attachment")


@override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
class EnrichCurrentTurnOnlyTests(TransactionTestCase):
    """Attachments are sent natively only on their own turn; every message that
    carried files gets ``[Attached: #N …]`` markers."""

    def setUp(self):
        self.user = User.objects.create_user(email="turn@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _consumer(self):
        c = ChatConsumer()
        c.scope = {"user": self.user}
        c.user = self.user
        return c

    async def _make_att(self, data, content_type, name, *, message=None, user=None, **extra):
        return await database_sync_to_async(ChatAttachment.objects.create)(
            thread=self.thread,
            message=message,
            uploaded_by=user or self.user,
            file=SimpleUploadedFile(name, data, content_type=content_type),
            original_filename=name,
            content_type=content_type,
            size_bytes=len(data),
            **extra,
        )

    async def _enrich(self, messages, history, ctx=None):
        ctx = ctx or RunContext.create()
        with patch("llm.display.supports_modality", return_value=True), \
             patch("chat.services.detect_provider", return_value="anthropic"), \
             patch("chat.services.get_or_extract_attachment_text", return_value="body"):
            await self._consumer()._enrich_with_attachments(
                messages, history, "test-model", ctx, thread_id=str(self.thread.id)
            )
        return ctx

    @staticmethod
    def _as_messages(history):
        return [Message(role="system", content="sys")] + [
            Message(role=h["role"], content=h["content"]) for h in history
        ]

    async def test_older_message_gets_marker_only(self):
        att = await self._make_att(b"x" * 500, "image/png", "pic.png")
        history = [
            {"role": "user", "content": "look", "attachment_ids": [str(att.id)]},
            {"role": "assistant", "content": "nice"},
            {"role": "user", "content": "and now?"},
        ]
        messages = self._as_messages(history)
        ctx = await self._enrich(messages, history)
        self.assertIsInstance(messages[1].content, str)
        self.assertIn("look", messages[1].content)
        self.assertIn("[Attached: #1 pic.png (image)]", messages[1].content)
        self.assertEqual(messages[3].content, "and now?")
        self.assertEqual(ctx._b64_used_by_pathway.get("attachment", 0), 0)

    async def test_current_message_has_marker_and_native_block(self):
        att = await self._make_att(b"x" * 500, "image/png", "pic.png")
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "look", "attachment_ids": [str(att.id)]},
        ]
        messages = self._as_messages(history)
        await self._enrich(messages, history)
        blocks = messages[3].content
        self.assertEqual(blocks[0]["type"], "text")
        self.assertIn("look", blocks[0]["text"])
        self.assertIn("[Attached: #1 pic.png (image)]", blocks[0]["text"])
        self.assertTrue(any(b.get("type") == "image" for b in blocks))

    async def test_all_user_entries_after_last_assistant_are_current(self):
        """Minutes seed turn: hidden seed + disclaimer(att), no assistant yet."""
        att = await self._make_att(b"x" * 500, "image/png", "slide.png")
        history = [
            {"role": "user", "content": "seed"},
            {"role": "user", "content": "disclaimer", "attachment_ids": [str(att.id)]},
        ]
        messages = self._as_messages(history)
        await self._enrich(messages, history)
        self.assertEqual(messages[1].content, "seed")
        self.assertTrue(any(b.get("type") == "image" for b in messages[2].content))

    async def test_branch_fallback_resolves_by_message_linkage(self):
        from chat.models import ChatMessage

        msg = await database_sync_to_async(ChatMessage.objects.create)(
            thread=self.thread, role="user", content="look",
        )
        await self._make_att(b"x" * 500, "image/png", "copy.png", message=msg)
        # Metadata ids point at the SOURCE thread's rows (not on this thread).
        history = [{
            "role": "user", "content": "look", "message_id": str(msg.id),
            "attachment_ids": ["00000000-0000-0000-0000-000000000000"],
        }]
        messages = self._as_messages(history)
        await self._enrich(messages, history)
        blocks = messages[1].content
        self.assertIn("[Attached: #1 copy.png (image)]", blocks[0]["text"])
        self.assertTrue(any(b.get("type") == "image" for b in blocks))

    async def test_pdf_marker_includes_known_page_count(self):
        att = await self._make_att(b"%PDF-1.4\n" + b"x" * 100, _PDF_MIME, "deck.pdf", page_count=12)
        history = [
            {"role": "user", "content": "read", "attachment_ids": [str(att.id)]},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "more"},
        ]
        messages = self._as_messages(history)
        await self._enrich(messages, history)
        self.assertIn("[Attached: #1 deck.pdf (PDF, 12 pages)]", messages[1].content)

    async def test_other_users_upload_is_not_sent(self):
        other = await database_sync_to_async(User.objects.create_user)(
            email="other-turn@example.com", password="pw"
        )
        att = await self._make_att(b"x" * 500, "image/png", "theirs.png", user=other)
        history = [{"role": "user", "content": "look", "attachment_ids": [str(att.id)]}]
        messages = self._as_messages(history)
        ctx = await self._enrich(messages, history)
        self.assertEqual(messages[1].content, "look")
        self.assertEqual(ctx._b64_used_by_pathway.get("attachment", 0), 0)
