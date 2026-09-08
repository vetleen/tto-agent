"""Tests for the provider-level trailing-assistant-message repair.

Anthropic rejects a request whose conversation ends with an assistant message
("assistant message prefill"); no caller builds that shape on purpose, but a
few edge paths can (WILFRED-7C). The provider base appends a continuation user
message instead of failing the turn, and logs the producing context.
"""

import logging

from django.test import SimpleTestCase

from llm.core.providers.base import BaseLangChainChatModel
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest


def _request(roles, context=None):
    msgs = [Message(role=r, content=f"{r} content") for r in roles]
    return ChatRequest(messages=msgs, context=context)


class TrailingRoleRepairTests(SimpleTestCase):
    def setUp(self):
        self.provider = BaseLangChainChatModel("test-model", client=object())

    def test_trailing_assistant_gets_continuation_user_message(self):
        req = _request(["system", "user", "assistant"])
        with self.assertLogs("llm.core.providers.base", level="WARNING") as cm:
            messages = self.provider._messages_with_trailing_role_repair(req)
        self.assertEqual([m.role for m in messages], ["system", "user", "assistant", "user"])
        self.assertIn("ended with an assistant message", cm.output[0])
        # The original request is not mutated — the repair is call-local.
        self.assertEqual(len(req.messages), 3)

    def test_repair_logs_run_context(self):
        ctx = RunContext.create(user_id="7", conversation_id="conv-1")
        ctx.agent_kind = "subagent"
        req = _request(["user", "assistant"], context=ctx)
        with self.assertLogs("llm.core.providers.base", level="WARNING") as cm:
            self.provider._messages_with_trailing_role_repair(req)
        msg = cm.records[0].getMessage()
        self.assertIn("subagent", msg)
        self.assertIn("conv-1", msg)

    def test_trailing_user_untouched(self):
        req = _request(["system", "user", "assistant", "user"])
        with self.assertNoLogs("llm.core.providers.base", level="WARNING"):
            messages = self.provider._messages_with_trailing_role_repair(req)
        self.assertIs(messages, req.messages)

    def test_trailing_tool_untouched(self):
        """Tool results become user turns in conversion — no repair needed."""
        req = _request(["system", "user", "assistant", "tool"])
        with self.assertNoLogs("llm.core.providers.base", level="WARNING"):
            messages = self.provider._messages_with_trailing_role_repair(req)
        self.assertIs(messages, req.messages)

    def test_empty_messages_untouched(self):
        req = _request([])
        messages = self.provider._messages_with_trailing_role_repair(req)
        self.assertEqual(messages, [])
