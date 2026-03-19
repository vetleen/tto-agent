"""Tests for the shared to_langchain_messages utility."""

from django.test import TestCase

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llm.core.langchain_utils import to_langchain_messages
from llm.types.messages import Message


class ToLangchainMessagesTests(TestCase):
    """Test role mapping and edge cases for to_langchain_messages."""

    def test_system_role_maps_to_system_message(self):
        result = to_langchain_messages([Message(role="system", content="Be helpful")])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], SystemMessage)
        self.assertEqual(result[0].content, "Be helpful")

    def test_user_role_maps_to_human_message(self):
        result = to_langchain_messages([Message(role="user", content="Hi")])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HumanMessage)
        self.assertEqual(result[0].content, "Hi")

    def test_assistant_role_maps_to_ai_message(self):
        result = to_langchain_messages([Message(role="assistant", content="Hello")])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AIMessage)
        self.assertEqual(result[0].content, "Hello")

    def test_tool_role_maps_to_human_message(self):
        """role='tool' without tool_call_id maps to HumanMessage (backward compat)."""
        result = to_langchain_messages([Message(role="tool", content="result")])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], HumanMessage)
        self.assertEqual(result[0].content, "result")

    def test_assistant_with_tool_calls_maps_to_ai_message_with_tool_calls(self):
        from llm.types.messages import ToolCall
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="id1", name="search_documents", arguments={"a": 1, "b": 2}),
            ],
        )
        result = to_langchain_messages([msg])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AIMessage)
        self.assertEqual(result[0].content, "")
        self.assertEqual(len(result[0].tool_calls), 1)
        self.assertEqual(result[0].tool_calls[0]["id"], "id1")
        self.assertEqual(result[0].tool_calls[0]["name"], "search_documents")
        self.assertEqual(result[0].tool_calls[0]["args"], {"a": 1, "b": 2})

    def test_tool_role_with_tool_call_id_maps_to_tool_message(self):
        result = to_langchain_messages([
            Message(role="tool", content='{"result": 5}', tool_call_id="call_1"),
        ])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ToolMessage)
        self.assertEqual(result[0].content, '{"result": 5}')
        self.assertEqual(result[0].tool_call_id, "call_1")

    def test_mixed_conversation_preserves_order(self):
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Question"),
            Message(role="assistant", content="Answer"),
            Message(role="user", content="Follow-up"),
        ]
        result = to_langchain_messages(messages)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], SystemMessage)
        self.assertIsInstance(result[1], HumanMessage)
        self.assertIsInstance(result[2], AIMessage)
        self.assertIsInstance(result[3], HumanMessage)

    def test_empty_list_returns_empty(self):
        result = to_langchain_messages([])
        self.assertEqual(result, [])


class AnthropicCacheControlTests(TestCase):
    """Test Anthropic prompt caching via cache_control breakpoints."""

    def test_system_message_gets_cache_control_for_anthropic(self):
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hi"),
        ]
        result = to_langchain_messages(messages, provider="anthropic")
        self.assertIsInstance(result[0], SystemMessage)
        # System message should have content-block format with cache_control
        self.assertIsInstance(result[0].content, list)
        self.assertEqual(len(result[0].content), 1)
        block = result[0].content[0]
        self.assertEqual(block["type"], "text")
        self.assertEqual(block["text"], "You are helpful")
        self.assertEqual(block["cache_control"], {"type": "ephemeral"})

    def test_second_to_last_gets_cache_control_with_3_plus_messages(self):
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Second question"),
        ]
        result = to_langchain_messages(messages, provider="anthropic")
        # Second-to-last (index 2) should have cache_control in additional_kwargs
        self.assertIn("cache_control", result[2].additional_kwargs)
        self.assertEqual(result[2].additional_kwargs["cache_control"], {"type": "ephemeral"})
        # Last message should NOT have cache_control
        self.assertNotIn("cache_control", result[3].additional_kwargs)

    def test_no_cache_control_when_provider_is_none(self):
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Second question"),
        ]
        result = to_langchain_messages(messages)
        # System message should be plain string content
        self.assertIsInstance(result[0].content, str)
        # No cache_control on any message
        for msg in result:
            self.assertNotIn("cache_control", getattr(msg, "additional_kwargs", {}))

    def test_no_cache_control_when_provider_is_openai(self):
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="Question"),
            Message(role="assistant", content="Answer"),
            Message(role="user", content="Follow-up"),
        ]
        result = to_langchain_messages(messages, provider="openai")
        self.assertIsInstance(result[0].content, str)
        for msg in result:
            self.assertNotIn("cache_control", getattr(msg, "additional_kwargs", {}))

    def test_short_conversation_only_caches_system(self):
        """With < 3 messages, only system gets cache_control (no second breakpoint)."""
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="Hello"),
        ]
        result = to_langchain_messages(messages, provider="anthropic")
        # System gets cache_control
        self.assertIsInstance(result[0].content, list)
        self.assertEqual(result[0].content[0]["cache_control"], {"type": "ephemeral"})
        # User message should NOT have cache_control
        self.assertNotIn("cache_control", result[1].additional_kwargs)

    def test_static_system_always_single_block(self):
        """System message is always a single cached block (static-only content)."""
        messages = [
            Message(role="system", content="Static instructions only"),
            Message(role="user", content="Hello"),
        ]
        result = to_langchain_messages(messages, provider="anthropic")
        self.assertIsInstance(result[0].content, list)
        self.assertEqual(len(result[0].content), 1)
        self.assertEqual(result[0].content[0]["text"], "Static instructions only")
        self.assertEqual(result[0].content[0]["cache_control"], {"type": "ephemeral"})
