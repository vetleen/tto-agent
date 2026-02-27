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
                ToolCall(id="id1", name="add_number", arguments={"a": 1, "b": 2}),
            ],
        )
        result = to_langchain_messages([msg])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AIMessage)
        self.assertEqual(result[0].content, "")
        self.assertEqual(len(result[0].tool_calls), 1)
        self.assertEqual(result[0].tool_calls[0]["id"], "id1")
        self.assertEqual(result[0].tool_calls[0]["name"], "add_number")
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
