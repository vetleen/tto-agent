"""_enrich_with_attachments metering: native tagging, budget degrade, and the
oversized-PDF → extracted-text fallback (Anthropic 32 MB request guard)."""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase, override_settings

from channels.db import database_sync_to_async

from django.core.files.base import ContentFile

from chat.consumers import ChatConsumer
from chat.models import Asset, ChatAttachment, ChatThread
from chat.tests.test_image_view import _slide_jpeg
from llm.types.context import RunContext
from llm.types.messages import Message

User = get_user_model()

_PDF_MIME = "application/pdf"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


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


@override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
    CHAT_ATTACHMENT_INITIAL_SLIDES=20,
)
class EnrichPptxSlidesTests(TransactionTestCase):
    """A deck on the upload turn: its slide-by-slide text, then the worker's
    slide renders as image blocks (per-message allowance, shared across decks),
    then a note that says how to see the rest."""

    def setUp(self):
        self.user = User.objects.create_user(email="slides@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _consumer(self):
        c = ChatConsumer()
        c.scope = {"user": self.user}
        c.user = self.user
        return c

    def _deck_sync(self, name, n_renders, *, total=None, state=None, processing=None, extracted=None):
        Render = ChatAttachment.PageRenderState
        att = ChatAttachment.objects.create(
            thread=self.thread,
            uploaded_by=self.user,
            file=SimpleUploadedFile(name, b"PK deck", content_type=_PPTX_MIME),
            original_filename=name,
            content_type=_PPTX_MIME,
            size_bytes=7,
            page_count=total if total is not None else n_renders,
            page_render_state=state or Render.READY,
            processing_state=processing or ChatAttachment.ProcessingState.READY,
            extracted_content=extracted if extracted is not None else "## Slide 1\n\nHello",
        )
        for n in range(1, n_renders + 1):
            asset = Asset(
                attachment=att, kind=Asset.KIND_IMAGE, role=Asset.ROLE_PAGE_RENDER,
                page_number=n, content_type="image/jpeg", description=f"Slide {n}",
            )
            asset.blob.save(f"{asset.id}.jpg", ContentFile(_slide_jpeg(n)), save=True)
        return att

    async def _deck(self, *args, **kwargs):
        return await database_sync_to_async(self._deck_sync)(*args, **kwargs)

    @staticmethod
    def _msgs_and_history(*atts):
        history = [{"role": "user", "content": "see the deck",
                    "attachment_ids": [str(a.id) for a in atts]}]
        messages = [Message(role="system", content="sys"), Message(role="user", content="see the deck")]
        return messages, history

    async def _enrich(self, messages, history, *, ctx=None, vision=True, not_ready_ids=None):
        ctx = ctx or RunContext.create()
        with patch("llm.display.supports_modality", return_value=vision), \
             patch("chat.services.detect_provider", return_value="anthropic"):
            await self._consumer()._enrich_with_attachments(
                messages, history, "test-model", ctx,
                thread_id=str(self.thread.id), not_ready_ids=not_ready_ids,
            )
        return ctx

    @staticmethod
    def _images(blocks):
        return [b for b in blocks if isinstance(b, dict) and b.get("type") == "image"]

    @staticmethod
    def _texts(blocks):
        return [b["text"] for b in blocks if isinstance(b, dict) and b.get("type") == "text"]

    async def test_first_slides_go_in_as_image_blocks_after_the_text(self):
        att = await self._deck("deck.pptx", 25)
        messages, history = self._msgs_and_history(att)
        ctx = await self._enrich(messages, history)

        blocks = messages[1].content
        self.assertIn("[Attached: #1 deck.pptx (presentation, 25 slides)]", blocks[0]["text"])
        self.assertTrue(blocks[1]["text"].startswith("[Attached file: deck.pptx]"))
        self.assertIn("## Slide 1", blocks[1]["text"])
        images = self._images(blocks)
        self.assertEqual(len(images), 20)
        self.assertEqual(images[0]["_wf_label"], "deck.pptx slide 1")
        self.assertEqual(images[-1]["_wf_label"], "deck.pptx slide 20")
        self.assertTrue(all(b["_wf_pathway"] == "attachment" for b in images))
        self.assertTrue(all(b["_wf_est_tokens"] > 0 for b in images))
        note = self._texts(blocks)[-1]
        self.assertIn("Slides 1–20 of 25", note)
        self.assertIn("pages='21-25'", note)
        self.assertGreater(ctx._b64_used_by_pathway.get("attachment", 0), 0)

    async def test_allowance_is_shared_across_decks_on_one_message(self):
        first = await self._deck("a.pptx", 15)
        second = await self._deck("b.pptx", 15)
        messages, history = self._msgs_and_history(first, second)
        await self._enrich(messages, history)

        blocks = messages[1].content
        labels = [b["_wf_label"] for b in self._images(blocks)]
        self.assertEqual(len(labels), 20)
        self.assertEqual(labels[:15], [f"a.pptx slide {n}" for n in range(1, 16)])
        self.assertEqual(labels[15:], [f"b.pptx slide {n}" for n in range(1, 6)])
        notes = self._texts(blocks)
        self.assertTrue(any("Slides 1–15 of 15 of 'a.pptx'" in t for t in notes))
        self.assertTrue(any("Slides 1–5 of 15 of 'b.pptx'" in t and "pages='6-15'" in t for t in notes))

    async def test_deck_past_the_allowance_gets_text_and_a_pointer(self):
        first = await self._deck("a.pptx", 20)
        second = await self._deck("b.pptx", 3)
        messages, history = self._msgs_and_history(first, second)
        await self._enrich(messages, history)
        blocks = messages[1].content
        self.assertEqual(len(self._images(blocks)), 20)
        self.assertTrue(any("allowance for this message is used up" in t and "'b.pptx'" in t
                            for t in self._texts(blocks)))

    async def test_pending_deck_is_text_with_a_note_and_a_still_processing_marker(self):
        att = await self._deck(
            "deck.pptx", 0, total=9,
            state=ChatAttachment.PageRenderState.PENDING,
            processing=ChatAttachment.ProcessingState.PENDING,
        )
        messages, history = self._msgs_and_history(att)
        ctx = await self._enrich(messages, history, not_ready_ids=[str(att.id)])
        blocks = messages[1].content
        self.assertIn("(presentation, 9 slides; still processing)", blocks[0]["text"])
        self.assertEqual(self._images(blocks), [])
        self.assertIn("## Slide 1", blocks[1]["text"])
        self.assertIn("still being prepared", self._texts(blocks)[-1])
        self.assertEqual(ctx._b64_used_by_pathway.get("attachment", 0), 0)

    async def test_skipped_deck_is_text_only(self):
        att = await self._deck("big.pptx", 0, total=80, state=ChatAttachment.PageRenderState.SKIPPED)
        messages, history = self._msgs_and_history(att)
        await self._enrich(messages, history)
        blocks = messages[1].content
        self.assertEqual(self._images(blocks), [])
        self.assertIn("No slide images are available for 'big.pptx'", self._texts(blocks)[-1])

    async def test_non_vision_model_gets_text_only(self):
        att = await self._deck("deck.pptx", 5)
        messages, history = self._msgs_and_history(att)
        ctx = await self._enrich(messages, history, vision=False)
        blocks = messages[1].content
        self.assertEqual(self._images(blocks), [])
        self.assertIn("cannot view images", self._texts(blocks)[-1])
        self.assertEqual(ctx._b64_used_by_pathway.get("attachment", 0), 0)

    async def test_budget_exhaustion_stops_the_slides_and_says_so(self):
        att = await self._deck("deck.pptx", 6)
        messages, history = self._msgs_and_history(att)
        ctx = RunContext.create()
        with patch.object(RunContext, "reserve_native_asset", side_effect=[True, True, False]):
            await self._enrich(messages, history, ctx=ctx)
        blocks = messages[1].content
        self.assertEqual([b["_wf_label"] for b in self._images(blocks)],
                         ["deck.pptx slide 1", "deck.pptx slide 2"])
        note = self._texts(blocks)[-1]
        self.assertIn("Slides 1–2 of 6", note)
        self.assertIn("budget", note)
        self.assertNotIn("pages=", note)

    async def test_older_deck_message_keeps_only_the_marker(self):
        att = await self._deck("deck.pptx", 5)
        history = [
            {"role": "user", "content": "look", "attachment_ids": [str(att.id)]},
            {"role": "assistant", "content": "nice"},
            {"role": "user", "content": "and slide 3?"},
        ]
        messages = [Message(role="system", content="sys")] + [
            Message(role=h["role"], content=h["content"]) for h in history
        ]
        ctx = await self._enrich(messages, history)
        self.assertIsInstance(messages[1].content, str)
        self.assertIn("[Attached: #1 deck.pptx (presentation, 5 slides)]", messages[1].content)
        self.assertEqual(ctx._b64_used_by_pathway.get("attachment", 0), 0)
