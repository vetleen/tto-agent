"""Tests for chat.services (summarisation)."""

from unittest.mock import AsyncMock, MagicMock, patch

from django.test import TransactionTestCase


class GenerateSummaryTests(TransactionTestCase):
    @patch("llm.get_llm_service")
    async def test_generates_summary(self, mock_get_service):
        from chat.services import generate_summary

        mock_response = MagicMock()
        mock_response.message.content = "This is a summary."
        mock_service = MagicMock()
        mock_service.arun = AsyncMock(return_value=mock_response)
        mock_get_service.return_value = mock_service

        msg1 = MagicMock(role="user", content="Hello")
        msg2 = MagicMock(role="assistant", content="Hi there!")

        result = await generate_summary(
            [msg1, msg2],
            user_id=1,
            conversation_id="proj-1",
        )

        self.assertEqual(result, "This is a summary.")
        mock_service.arun.assert_called_once()

        # Verify it used the medium-tier model from settings
        from django.conf import settings
        call_args = mock_service.arun.call_args
        request = call_args[0][1]
        self.assertEqual(request.model, settings.LLM_DEFAULT_MEDIUM_MODEL)

    @patch("llm.get_llm_service")
    async def test_includes_existing_summary(self, mock_get_service):
        from chat.services import generate_summary

        mock_response = MagicMock()
        mock_response.message.content = "Updated summary."
        mock_service = MagicMock()
        mock_service.arun = AsyncMock(return_value=mock_response)
        mock_get_service.return_value = mock_service

        msg = MagicMock(role="user", content="New message")

        result = await generate_summary(
            [msg],
            existing_summary="Old summary text.",
            user_id=1,
            conversation_id="proj-1",
        )

        self.assertEqual(result, "Updated summary.")
        # The prompt should contain the old summary
        call_args = mock_service.arun.call_args
        request = call_args[0][1]
        user_msg = [m for m in request.messages if m.role == "user"][0]
        self.assertIn("Old summary text.", user_msg.content)

    @patch("llm.get_llm_service")
    async def test_target_tokens_reach_the_prompt(self, mock_get_service):
        from chat.services import generate_summary

        mock_response = MagicMock()
        mock_response.message.content = "S"
        mock_service = MagicMock()
        mock_service.arun = AsyncMock(return_value=mock_response)
        mock_get_service.return_value = mock_service

        await generate_summary(
            [MagicMock(role="user", content="x")], user_id=1, conversation_id="c",
            target_tokens=12_000,
        )
        request = mock_service.arun.call_args[0][1]
        system = [m for m in request.messages if m.role == "system"][0].content
        self.assertIn("12,000 tokens", system)
        self.assertIn("9,000 words", system)
