"""Tests for message and canvas redaction on guardrail block."""

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase

from chat.consumers import ChatConsumer
from chat.models import CanvasCheckpoint, ChatCanvas, ChatMessage, ChatThread
from chat.services import create_canvas_checkpoint

User = get_user_model()

REDACTED_TEXT = "[This message was removed by the content safety system.]"


class RedactMessagesTests(TransactionTestCase):
    """Test _redact_messages overwrites content and sets is_redacted flag."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", password="pass"
        )
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()
        self.consumer.user = self.user

    @database_sync_to_async
    def _make_msg(self, content, role="user", **kwargs):
        return ChatMessage.objects.create(
            thread=self.thread,
            role=role,
            content=content,
            **kwargs,
        )

    @database_sync_to_async
    def _refresh(self, msg):
        msg.refresh_from_db()
        return msg

    async def test_redacts_user_and_assistant_messages(self):
        user_msg = await self._make_msg("attack prompt", role="user")
        asst_msg = await self._make_msg("dangerous response", role="assistant")

        await self.consumer._redact_messages(self.thread)

        user_msg = await self._refresh(user_msg)
        asst_msg = await self._refresh(asst_msg)
        self.assertEqual(user_msg.content, REDACTED_TEXT)
        self.assertTrue(user_msg.is_redacted)
        self.assertEqual(user_msg.token_count, 0)
        self.assertEqual(user_msg.metadata, {})
        self.assertEqual(asst_msg.content, REDACTED_TEXT)
        self.assertTrue(asst_msg.is_redacted)

    async def test_redacts_tool_messages_after_user(self):
        user_msg = await self._make_msg("attack", role="user")
        tool_msg = await self._make_msg("tool output", role="tool", tool_call_id="call_1")
        asst_msg = await self._make_msg("response", role="assistant")

        await self.consumer._redact_messages(self.thread)

        tool_msg = await self._refresh(tool_msg)
        self.assertTrue(tool_msg.is_redacted)
        self.assertEqual(tool_msg.content, REDACTED_TEXT)

    async def test_redact_user_only(self):
        user_msg = await self._make_msg("attack", role="user")
        asst_msg = await self._make_msg("response", role="assistant")

        await self.consumer._redact_messages(
            self.thread, redact_assistant=False,
        )

        user_msg = await self._refresh(user_msg)
        asst_msg = await self._refresh(asst_msg)
        self.assertTrue(user_msg.is_redacted)
        self.assertFalse(asst_msg.is_redacted)
        self.assertEqual(asst_msg.content, "response")

    async def test_redact_assistant_only(self):
        user_msg = await self._make_msg("question", role="user")
        asst_msg = await self._make_msg("bad response", role="assistant")

        await self.consumer._redact_messages(
            self.thread, redact_user=False,
        )

        user_msg = await self._refresh(user_msg)
        asst_msg = await self._refresh(asst_msg)
        self.assertFalse(user_msg.is_redacted)
        self.assertEqual(user_msg.content, "question")
        self.assertTrue(asst_msg.is_redacted)

    async def test_does_not_redact_earlier_messages(self):
        old_user = await self._make_msg("old question", role="user")
        old_asst = await self._make_msg("old answer", role="assistant")
        new_user = await self._make_msg("attack", role="user")
        new_asst = await self._make_msg("leaked", role="assistant")

        await self.consumer._redact_messages(self.thread)

        old_user = await self._refresh(old_user)
        old_asst = await self._refresh(old_asst)
        self.assertFalse(old_user.is_redacted)
        self.assertFalse(old_asst.is_redacted)
        self.assertEqual(old_user.content, "old question")
        self.assertEqual(old_asst.content, "old answer")

    async def test_clears_metadata(self):
        user_msg = await self._make_msg(
            "attack", role="user",
            metadata={"attachment_ids": ["abc"]},
        )
        await self.consumer._redact_messages(self.thread)
        user_msg = await self._refresh(user_msg)
        self.assertEqual(user_msg.metadata, {})

    async def test_no_user_message_is_noop(self):
        """If there's no user message, redaction does nothing."""
        asst_msg = await self._make_msg("orphan", role="assistant")
        await self.consumer._redact_messages(self.thread)
        asst_msg = await self._refresh(asst_msg)
        self.assertFalse(asst_msg.is_redacted)


class LoadHistoryExcludesRedactedTests(TransactionTestCase):
    """Test that _load_history excludes redacted messages."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="hist@example.com", password="pass"
        )
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()

    @database_sync_to_async
    def _make_msg(self, content, role="user", is_redacted=False):
        return ChatMessage.objects.create(
            thread=self.thread,
            role=role,
            content=content,
            is_redacted=is_redacted,
        )

    async def test_excludes_redacted_from_history(self):
        await self._make_msg("good question", role="user")
        await self._make_msg("good answer", role="assistant")
        await self._make_msg("attack", role="user", is_redacted=True)
        await self._make_msg("leaked", role="assistant", is_redacted=True)

        result = await self.consumer._load_history(self.thread)
        self.assertEqual(result["meta"]["total_messages"], 2)
        contents = [m["content"] for m in result["messages"]]
        self.assertIn("good question", contents)
        self.assertIn("good answer", contents)
        self.assertNotIn("attack", contents)
        self.assertNotIn("leaked", contents)

    async def test_redacted_still_in_page_load_queryset(self):
        """Redacted messages should still appear in the view's queryset for rendering."""
        msg = await self._make_msg("attack", role="user", is_redacted=True)

        @database_sync_to_async
        def get_all():
            return list(self.thread.messages.order_by("created_at")[:100])

        all_msgs = await get_all()
        self.assertEqual(len(all_msgs), 1)
        self.assertTrue(all_msgs[0].is_redacted)


class RedactCanvasTests(TransactionTestCase):
    """Test _redact_canvases rolls back or replaces canvas content on guardrail block."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="canvas@example.com", password="pass"
        )
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()
        self.consumer.user = self.user

    @database_sync_to_async
    def _make_msg(self, content, role="user", **kwargs):
        return ChatMessage.objects.create(
            thread=self.thread, role=role, content=content, **kwargs
        )

    @database_sync_to_async
    def _make_canvas(self, title="Test doc", content="canvas content"):
        return ChatCanvas.objects.create(
            thread=self.thread, title=title, content=content
        )

    @database_sync_to_async
    def _make_checkpoint(self, canvas, source="ai_edit", content=None, description=""):
        canvas.content = content if content is not None else canvas.content
        canvas.save(update_fields=["content", "updated_at"])
        return create_canvas_checkpoint(canvas, source=source, description=description)

    @database_sync_to_async
    def _refresh_canvas(self, canvas):
        canvas.refresh_from_db()
        return canvas

    @database_sync_to_async
    def _get_checkpoint(self, pk):
        return CanvasCheckpoint.objects.get(pk=pk)

    @database_sync_to_async
    def _get_checkpoints(self, canvas):
        return list(CanvasCheckpoint.objects.filter(canvas=canvas).order_by("order"))

    async def test_canvas_created_during_blocked_turn_is_redacted(self):
        """Canvas created via write_canvas during a blocked turn gets content replaced."""
        user_msg = await self._make_msg("attack prompt", role="user")
        canvas = await self._make_canvas(content="fabricated legal letter")
        cp = await self._make_checkpoint(canvas, source="original", content="fabricated legal letter")

        # Track it as modified during this turn
        self.consumer._modified_canvas_ids = {str(canvas.pk)}

        result = await self.consumer._redact_canvases(self.thread)

        self.assertEqual(result, [str(canvas.pk)])
        canvas = await self._refresh_canvas(canvas)
        self.assertEqual(canvas.content, REDACTED_TEXT)
        self.assertIsNone(canvas.accepted_checkpoint)

        cp = await self._get_checkpoint(cp.pk)
        self.assertEqual(cp.source, "redacted")
        self.assertEqual(cp.content, REDACTED_TEXT)

    async def test_canvas_rolls_back_to_pre_turn_checkpoint(self):
        """Canvas with existing history rolls back to the accepted checkpoint."""
        # Pre-turn: user saved a version
        old_user_msg = await self._make_msg("old question", role="user")
        canvas = await self._make_canvas(title="Contract", content="safe content v1")
        pre_cp = await self._make_checkpoint(
            canvas, source="user_save", content="safe content v1", description="Saved"
        )

        @database_sync_to_async
        def accept_checkpoint():
            canvas.accepted_checkpoint = pre_cp
            canvas.save(update_fields=["accepted_checkpoint"])
        await accept_checkpoint()

        # New turn: attack prompt triggers AI canvas edit
        new_user_msg = await self._make_msg("attack", role="user")

        @database_sync_to_async
        def simulate_ai_edit():
            canvas.content = "harmful content"
            canvas.save(update_fields=["content", "updated_at"])
            return create_canvas_checkpoint(canvas, source="ai_edit", description="AI edit")
        turn_cp = await simulate_ai_edit()

        self.consumer._modified_canvas_ids = {str(canvas.pk)}

        result = await self.consumer._redact_canvases(self.thread)

        self.assertEqual(result, [str(canvas.pk)])
        canvas = await self._refresh_canvas(canvas)
        self.assertEqual(canvas.content, "safe content v1")
        self.assertEqual(canvas.title, "Contract")

        # Pre-turn checkpoint should be the accepted one now
        self.assertEqual(canvas.accepted_checkpoint_id, pre_cp.pk)

        # Turn checkpoint should be marked redacted
        turn_cp = await self._get_checkpoint(turn_cp.pk)
        self.assertEqual(turn_cp.source, "redacted")

    async def test_unrelated_canvas_not_affected(self):
        """A canvas from an earlier turn should not be redacted."""
        old_user = await self._make_msg("old question", role="user")
        old_canvas = await self._make_canvas(title="Old doc", content="old content")
        await self._make_checkpoint(old_canvas, source="original", content="old content")

        # New turn with no canvas modifications
        new_user = await self._make_msg("attack", role="user")
        self.consumer._modified_canvas_ids = set()

        result = await self.consumer._redact_canvases(self.thread)

        self.assertEqual(result, [])
        old_canvas = await self._refresh_canvas(old_canvas)
        self.assertEqual(old_canvas.content, "old content")

    async def test_redacted_checkpoint_cannot_be_restored(self):
        """_canvas_restore_version returns None for redacted checkpoints."""
        canvas = await self._make_canvas(content="bad content")
        cp = await self._make_checkpoint(canvas, source="original", content="bad content")

        @database_sync_to_async
        def mark_redacted():
            CanvasCheckpoint.objects.filter(pk=cp.pk).update(
                source="redacted", content=REDACTED_TEXT
            )
        await mark_redacted()

        result = await self.consumer._canvas_restore_version(
            str(self.thread.pk), cp.pk, canvas_id=str(canvas.pk)
        )
        self.assertIsNone(result)

    async def test_no_canvas_modifications_is_noop(self):
        """When no canvases were modified during the turn, returns empty list."""
        await self._make_msg("attack", role="user")
        self.consumer._modified_canvas_ids = set()

        result = await self.consumer._redact_canvases(self.thread)

        self.assertEqual(result, [])
