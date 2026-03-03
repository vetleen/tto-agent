"""Tests for the token-aware _load_history in ProjectChatConsumer."""

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase

from chat.consumers import MAX_HISTORY_TOKENS, OVERLAP_TOKENS, ProjectChatConsumer
from chat.models import ChatMessage, ChatThread
from core.tokens import count_tokens
from documents.models import Project

User = get_user_model()


class LoadHistoryTests(TransactionTestCase):
    """Test _load_history returns correct structure and respects token budget."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", password="pass"
        )
        self.project = Project.objects.create(
            name="Test", slug="test-proj", created_by=self.user
        )
        self.thread = ChatThread.objects.create(
            project=self.project, created_by=self.user
        )
        self.consumer = ProjectChatConsumer()

    @database_sync_to_async
    def _make_msg(self, content, role="user"):
        return ChatMessage.objects.create(
            thread=self.thread,
            role=role,
            content=content,
            token_count=count_tokens(content),
        )

    @database_sync_to_async
    def _update_thread(self, **kwargs):
        ChatThread.objects.filter(pk=self.thread.pk).update(**kwargs)

    async def test_returns_dict_with_messages_and_meta(self):
        await self._make_msg("Hello")
        result = await self.consumer._load_history(self.thread)
        self.assertIn("messages", result)
        self.assertIn("meta", result)
        self.assertIsInstance(result["messages"], list)
        self.assertIsInstance(result["meta"], dict)

    async def test_meta_keys(self):
        await self._make_msg("Hello")
        meta = (await self.consumer._load_history(self.thread))["meta"]
        self.assertIn("total_messages", meta)
        self.assertIn("included_messages", meta)
        self.assertIn("has_summary", meta)
        self.assertIn("needs_summary", meta)

    async def test_all_messages_included_when_under_budget(self):
        for i in range(5):
            await self._make_msg(f"Message {i}")
        result = await self.consumer._load_history(self.thread)
        self.assertEqual(result["meta"]["total_messages"], 5)
        self.assertEqual(result["meta"]["included_messages"], 5)
        self.assertFalse(result["meta"]["needs_summary"])

    async def test_respects_token_budget(self):
        # Create messages that will exceed MAX_HISTORY_TOKENS
        big_content = "word " * 5000  # ~5000 tokens each
        for _ in range(6):  # ~30k tokens > 20k budget
            await self._make_msg(big_content)

        result = await self.consumer._load_history(self.thread)
        meta = result["meta"]
        self.assertEqual(meta["total_messages"], 6)
        self.assertLess(meta["included_messages"], 6)
        self.assertTrue(meta["needs_summary"])

    async def test_includes_summary_as_system_message(self):
        summary_text = "Previous conversation about cats."
        await self._update_thread(
            summary=summary_text,
            summary_token_count=count_tokens(summary_text),
        )

        await self._make_msg("New message")
        result = await self.consumer._load_history(self.thread)
        messages = result["messages"]

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Previous conversation about cats", messages[0]["content"])
        self.assertTrue(result["meta"]["has_summary"])

    async def test_no_summary_when_empty(self):
        await self._make_msg("Hello")
        result = await self.consumer._load_history(self.thread)
        system_msgs = [m for m in result["messages"] if m["role"] == "system"]
        self.assertEqual(len(system_msgs), 0)
        self.assertFalse(result["meta"]["has_summary"])

    async def test_messages_in_chronological_order(self):
        await self._make_msg("First")
        await self._make_msg("Second")
        await self._make_msg("Third")

        result = await self.consumer._load_history(self.thread)
        contents = [m["content"] for m in result["messages"]]
        self.assertEqual(contents, ["First", "Second", "Third"])

    async def test_overlap_includes_summarised_msgs_within_2k_tokens(self):
        """Small summarised messages still appear raw because they fit in the overlap window."""
        m1 = await self._make_msg("Short old message")
        await self._make_msg("Short new message")

        summary_text = "Summary of old stuff."
        await self._update_thread(
            summary=summary_text,
            summary_token_count=count_tokens(summary_text),
            summary_up_to_message_id=m1.id,
            summary_message_count=1,
        )

        result = await self.consumer._load_history(self.thread)
        # Both messages are tiny (<<2k tokens), so both appear in the overlap window
        non_system = [m for m in result["messages"] if m["role"] != "system"]
        self.assertEqual(len(non_system), 2)
        contents = [m["content"] for m in non_system]
        self.assertIn("Short old message", contents)
        self.assertIn("Short new message", contents)

    async def test_large_summarised_msgs_excluded_from_raw_history(self):
        """A summarised message that is outside the overlap window is not shown raw."""
        # Each message alone exceeds OVERLAP_TOKENS so it pushes the other out of the window
        big_content = "word " * 2500  # well over 2000 tokens
        m1 = await self._make_msg(big_content)
        await self._make_msg(big_content)

        summary_text = "Summary of old stuff."
        await self._update_thread(
            summary=summary_text,
            summary_token_count=count_tokens(summary_text),
            summary_up_to_message_id=m1.id,
            summary_message_count=1,
        )

        result = await self.consumer._load_history(self.thread)
        # Overlap = [m2] alone already >= OVERLAP_TOKENS, so m1 is outside the overlap
        non_system = [m for m in result["messages"] if m["role"] != "system"]
        self.assertEqual(len(non_system), 1)
        self.assertEqual(non_system[0]["content"], big_content)
