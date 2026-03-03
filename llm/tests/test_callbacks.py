"""Unit tests for llm.core.callbacks.PromptCaptureCallback."""

import uuid
from unittest.mock import MagicMock
from unittest import TestCase

from llm.core.callbacks import PromptCaptureCallback, _serialize_lc_message


def _make_system(content="You are helpful."):
    from langchain_core.messages import SystemMessage
    return SystemMessage(content=content)


def _make_human(content="Hello!"):
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=content)


def _make_ai(content="Sure!", tool_calls=None):
    from langchain_core.messages import AIMessage
    return AIMessage(content=content, tool_calls=tool_calls or [])


def _make_tool(content="42", tool_call_id="call_abc"):
    from langchain_core.messages import ToolMessage
    return ToolMessage(content=content, tool_call_id=tool_call_id)


class SerializeLcMessageTests(TestCase):
    """Tests for _serialize_lc_message helper."""

    def test_system_message(self):
        msg = _make_system("Be concise.")
        result = _serialize_lc_message(msg)
        self.assertEqual(result, {"role": "system", "content": "Be concise."})

    def test_human_message(self):
        msg = _make_human("What is 2+2?")
        result = _serialize_lc_message(msg)
        self.assertEqual(result, {"role": "user", "content": "What is 2+2?"})

    def test_ai_message_no_tool_calls(self):
        msg = _make_ai("The answer is 4.")
        result = _serialize_lc_message(msg)
        self.assertEqual(result, {"role": "assistant", "content": "The answer is 4."})
        self.assertNotIn("tool_calls", result)

    def test_ai_message_with_tool_calls(self):
        tool_calls = [{"id": "call_1", "name": "calculator", "args": {"x": 1}}]
        msg = _make_ai("", tool_calls=tool_calls)
        result = _serialize_lc_message(msg)
        self.assertEqual(result["role"], "assistant")
        self.assertIn("tool_calls", result)
        self.assertEqual(len(result["tool_calls"]), 1)
        tc = result["tool_calls"][0]
        self.assertEqual(tc["id"], "call_1")
        self.assertEqual(tc["name"], "calculator")
        self.assertEqual(tc["args"], {"x": 1})

    def test_tool_message(self):
        msg = _make_tool("result_value", tool_call_id="call_xyz")
        result = _serialize_lc_message(msg)
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["content"], "result_value")
        self.assertEqual(result["tool_call_id"], "call_xyz")

    def test_unknown_message_falls_back(self):
        unknown = MagicMock()
        unknown.content = "some content"
        # Unknown type — not an instance of any known LangChain message class
        result = _serialize_lc_message(unknown)
        self.assertEqual(result["role"], "unknown")
        self.assertEqual(result["content"], "some content")


class PromptCaptureCallbackTests(TestCase):
    """Tests for PromptCaptureCallback.on_chat_model_start."""

    def test_captured_messages_none_before_firing(self):
        cb = PromptCaptureCallback()
        self.assertIsNone(cb.captured_messages)

    def test_captures_single_message(self):
        cb = PromptCaptureCallback()
        msg = _make_human("Hi there")
        cb.on_chat_model_start({}, [[msg]], run_id=uuid.uuid4())
        self.assertIsNotNone(cb.captured_messages)
        self.assertEqual(len(cb.captured_messages), 1)
        self.assertEqual(cb.captured_messages[0], {"role": "user", "content": "Hi there"})

    def test_captures_full_conversation(self):
        cb = PromptCaptureCallback()
        messages = [
            _make_system("You are a helper."),
            _make_human("What is Python?"),
            _make_ai("A programming language."),
        ]
        cb.on_chat_model_start({}, [messages], run_id=uuid.uuid4())
        self.assertEqual(len(cb.captured_messages), 3)
        self.assertEqual(cb.captured_messages[0]["role"], "system")
        self.assertEqual(cb.captured_messages[1]["role"], "user")
        self.assertEqual(cb.captured_messages[2]["role"], "assistant")

    def test_captures_tool_call_in_ai_message(self):
        cb = PromptCaptureCallback()
        tool_calls = [{"id": "call_99", "name": "search", "args": {"query": "foo"}}]
        messages = [_make_ai("", tool_calls=tool_calls)]
        cb.on_chat_model_start({}, [messages], run_id=uuid.uuid4())
        ai_dict = cb.captured_messages[0]
        self.assertIn("tool_calls", ai_dict)
        self.assertEqual(ai_dict["tool_calls"][0]["name"], "search")

    def test_empty_messages_batch_does_not_set(self):
        cb = PromptCaptureCallback()
        cb.on_chat_model_start({}, [], run_id=uuid.uuid4())
        self.assertIsNone(cb.captured_messages)

    def test_uses_first_batch_element(self):
        """on_chat_model_start receives list[list[...]] — we use messages[0]."""
        cb = PromptCaptureCallback()
        batch_1 = [_make_human("First batch")]
        batch_2 = [_make_human("Second batch")]
        cb.on_chat_model_start({}, [batch_1, batch_2], run_id=uuid.uuid4())
        self.assertEqual(cb.captured_messages[0]["content"], "First batch")
