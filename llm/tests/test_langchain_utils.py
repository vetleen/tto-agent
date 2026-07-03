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
                ToolCall(id="id1", name="document_search", arguments={"a": 1, "b": 2}),
            ],
        )
        result = to_langchain_messages([msg])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AIMessage)
        self.assertEqual(result[0].content, "")
        self.assertEqual(len(result[0].tool_calls), 1)
        self.assertEqual(result[0].tool_calls[0]["id"], "id1")
        self.assertEqual(result[0].tool_calls[0]["name"], "document_search")
        self.assertEqual(result[0].tool_calls[0]["args"], {"a": 1, "b": 2})

    def test_assistant_content_blocks_round_trip_as_list_content(self):
        """Captured thinking/text blocks are echoed as list content (with the
        signature) instead of dropped to a plain-text rebuild."""
        from llm.types.messages import ToolCall
        blocks = [
            {"type": "thinking", "thinking": "reasoning...", "signature": "sig123"},
            {"type": "text", "text": "Let me search."},
        ]
        msg = Message(
            role="assistant",
            content="Let me search.",
            tool_calls=[ToolCall(id="tu1", name="document_search", arguments={"q": "x"})],
            metadata={"content_blocks": blocks},
        )
        result = to_langchain_messages([msg])
        self.assertIsInstance(result[0].content, list)
        self.assertEqual(result[0].content[0]["type"], "thinking")
        self.assertEqual(result[0].content[0]["signature"], "sig123")
        self.assertEqual(len(result[0].tool_calls), 1)

    def test_assistant_without_content_blocks_stays_plain_string(self):
        msg = Message(role="assistant", content="plain", metadata={})
        self.assertEqual(to_langchain_messages([msg])[0].content, "plain")

    def test_assistant_additional_kwargs_restored_for_gemini_signatures(self):
        """Gemini function-call thought signatures stashed in metadata are
        restored onto the AIMessage's additional_kwargs (keyed by tool_call id)
        so langchain-google-genai resends them."""
        from llm.types.messages import ToolCall
        key = "__gemini_function_call_thought_signatures__"
        sig_map = {"tu1": "c2lnMTIz"}
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="tu1", name="search", arguments={})],
            metadata={"additional_kwargs": {key: sig_map}},
        )
        result = to_langchain_messages([msg])
        self.assertEqual(result[0].additional_kwargs[key], sig_map)
        self.assertEqual(result[0].tool_calls[0]["id"], "tu1")

    def test_anthropic_serializes_thinking_first_then_tool_use(self):
        """Integration guard against langchain-anthropic: the reconstructed
        message must serialize to [thinking, ..., tool_use] with thinking FIRST —
        the exact shape Anthropic requires for a thinking+tool-use turn. The
        tool_use block is re-appended from tool_calls by langchain-anthropic."""
        from langchain_anthropic.chat_models import _format_messages
        from llm.types.messages import ToolCall
        blocks = [
            {"type": "thinking", "thinking": "reasoning...", "signature": "sig123"},
            {"type": "text", "text": "Searching."},
        ]
        msg = Message(
            role="assistant",
            content="Searching.",
            tool_calls=[ToolCall(id="tu1", name="document_search", arguments={"q": "x"})],
            metadata={"content_blocks": blocks},
        )
        _system, formatted = _format_messages(to_langchain_messages([msg]))
        content = formatted[0]["content"]
        types = [b["type"] for b in content]
        self.assertEqual(types[0], "thinking")
        self.assertIn("tool_use", types)
        self.assertLess(types.index("thinking"), types.index("tool_use"))

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
        self.assertEqual(block["cache_control"], {"type": "ephemeral", "ttl": "1h"})

    def test_no_conversation_prefix_breakpoint(self):
        """Conversation history should NOT get explicit cache_control breakpoints.

        Prefix caching for conversation history is handled automatically via
        the cache_control kwarg bound in AnthropicChatModel._get_streaming_client.
        """
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Second question"),
        ]
        result = to_langchain_messages(messages, provider="anthropic")
        for msg in result[1:]:
            self.assertNotIn("cache_control", getattr(msg, "additional_kwargs", {}))

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

    def test_short_conversation_caches_system_only(self):
        """Even short conversations get system message caching."""
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="Hello"),
        ]
        result = to_langchain_messages(messages, provider="anthropic")
        self.assertIsInstance(result[0].content, list)
        self.assertEqual(result[0].content[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
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
        self.assertEqual(result[0].content[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
