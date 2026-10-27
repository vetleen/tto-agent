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

    def test_signed_text_blocks(self):
        from llm.core.langchain_utils import signed_text_blocks

        signed = [
            {"type": "thinking", "thinking": "plan"},
            {"type": "text", "text": "Answer"},
            {"type": "text", "text": "", "extras": {"signature": "c2ln"}},
        ]
        self.assertEqual(signed_text_blocks(signed), signed[1:])
        self.assertIsNone(signed_text_blocks([{"type": "text", "text": "Answer"}]))
        self.assertIsNone(signed_text_blocks(None))

    def test_gemini_signed_text_part_round_trips_to_a_signed_part(self):
        """langchain-google-genai turns a text block's extras.signature back into
        Part.thought_signature on the request."""
        from langchain_google_genai.chat_models import _parse_chat_history

        blocks = [
            {"type": "text", "text": "Answer"},
            {"type": "text", "text": "", "extras": {"signature": "c2ln"}},
        ]
        msg = Message(role="assistant", content="Answer", metadata={"content_blocks": blocks})
        lc = to_langchain_messages([Message(role="user", content="hi"), msg])
        _, contents = _parse_chat_history(lc)
        parts = contents[-1].parts
        self.assertTrue(any(p.thought_signature == b"sig" for p in parts))

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

    def test_summary_system_message_does_not_replace_static_prompt(self):
        """A second system message (the conversation summary) is sent AFTER the
        static prompt; the 1h breakpoint stays on the static prompt only."""
        from langchain_anthropic.chat_models import _format_messages

        messages = [
            Message(role="system", content="Static instructions"),
            Message(role="system", content="Summary of earlier conversation:\nS"),
            Message(role="user", content="Hello"),
        ]
        result = to_langchain_messages(messages, provider="anthropic")
        system, _formatted = _format_messages(result)
        self.assertEqual(
            [b["text"] for b in system],
            ["Static instructions", "Summary of earlier conversation:\nS"],
        )
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertNotIn("cache_control", system[1])


class AnthropicHistoryBreakpointTests(TestCase):
    """Turn N writes a cache entry just before its anchor (the user message
    carrying the per-turn preamble); turn N+1 reads exactly there."""

    @staticmethod
    def _breakpoints(lc):
        out = []
        for i, m in enumerate(lc):
            if isinstance(m.content, list):
                for b in m.content:
                    if isinstance(b, dict) and "cache_control" in b:
                        out.append((i, b.get("text", "")[:20]))
        return out

    def _turn(self, history, user_text):
        from llm.core.langchain_utils import TURN_ANCHOR_KEY

        anchor = Message(role="user", content="PREAMBLE\n" + user_text,
                         metadata={TURN_ANCHOR_KEY: True})
        return [Message(role="system", content="STATIC")] + history + [anchor]

    def test_no_anchor_no_history_breakpoints(self):
        msgs = [Message(role="system", content="S"), Message(role="user", content="q"),
                Message(role="assistant", content="a"), Message(role="user", content="q2")]
        lc = to_langchain_messages(msgs, provider="anthropic")
        self.assertEqual([i for i, _ in self._breakpoints(lc)], [0])

    def test_write_then_read_at_the_same_block(self):
        from llm.types.messages import ToolCall

        turn1 = [Message(role="user", content="Q1"), Message(role="assistant", content="A1")]
        # Turn 2's request: write before its anchor (A1); no previous turn to read.
        lc2 = to_langchain_messages(self._turn(turn1, "Q2"), provider="anthropic")
        self.assertEqual(self._breakpoints(lc2), [(0, "STATIC"), (2, "A1")])

        # Turn 2 ran tools (more than 20 blocks would defeat the lookback).
        loop = []
        for n in range(15):
            loop += [Message(role="assistant", content="", tool_calls=[ToolCall(id=f"t{n}", name="x", arguments={})]),
                     Message(role="tool", content=f"R{n}", tool_call_id=f"t{n}")]
        turn2 = turn1 + [Message(role="user", content="Q2")] + loop + [Message(role="assistant", content="A2")]
        lc3 = to_langchain_messages(self._turn(turn2, "Q3"), provider="anthropic")
        # Read where turn 2 wrote (A1, index 2), write before the new anchor (A2).
        self.assertEqual(self._breakpoints(lc3), [(0, "STATIC"), (2, "A1"), (len(turn2), "A2")])
        # The prefix up to and including the read breakpoint is identical.
        self.assertEqual(lc3[:3], lc2[:3])

    def test_history_breakpoints_use_the_1h_ttl(self):
        history = [Message(role="user", content="Q1"), Message(role="assistant", content="A1"),
                   Message(role="user", content="Q2"), Message(role="assistant", content="A2")]
        lc = to_langchain_messages(self._turn(history, "Q3"), provider="anthropic")
        controls = [b["cache_control"] for m in lc if isinstance(m.content, list)
                    for b in m.content if isinstance(b, dict) and "cache_control" in b]
        self.assertEqual(len(controls), 3)  # system + read (A1) + write (A2)
        self.assertTrue(all(c == {"type": "ephemeral", "ttl": "1h"} for c in controls))

    def test_tool_results_and_call_only_turns_are_skipped(self):
        from llm.types.messages import ToolCall

        history = [Message(role="user", content="Q1"),
                   Message(role="assistant", content="", tool_calls=[ToolCall(id="t", name="x", arguments={})]),
                   Message(role="tool", content="R", tool_call_id="t")]
        lc = to_langchain_messages(self._turn(history, "Q2"), provider="anthropic")
        # Nothing between Q1 and the anchor can carry a breakpoint → Q1 does.
        self.assertEqual(self._breakpoints(lc), [(0, "STATIC"), (1, "Q1")])

    def test_never_more_than_three_explicit_breakpoints(self):
        """With langchain-anthropic's automatic end breakpoint that is four,
        Anthropic's maximum."""
        history = []
        for n in range(6):
            history += [Message(role="user", content=f"Q{n}"), Message(role="assistant", content=f"A{n}")]
        history.insert(1, Message(role="system", content="Summary of earlier conversation:\nS"))
        lc = to_langchain_messages(self._turn(history, "Q9"), provider="anthropic")
        self.assertLessEqual(len(self._breakpoints(lc)), 3)

    def test_other_providers_untouched(self):
        lc = to_langchain_messages(self._turn([Message(role="user", content="Q1"),
                                               Message(role="assistant", content="A1")], "Q2"),
                                   provider="openai")
        self.assertEqual(self._breakpoints(lc), [])
