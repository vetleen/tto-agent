"""Tests for guardrails.web_content — the enqueue adapter for web-content scanning.

The heuristic scan was removed; this module now only resolves attribution and
enqueues the worker task (guardrails.tasks.scan_web_content_task). The scan logic
itself is tested in test_web_tasks.py.
"""

from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase, TransactionTestCase

from accounts.models import Membership, Organization, User
from guardrails.tasks import _MAX_WEB_SCAN_CHARS
from guardrails.web_content import _resolve_org_id, scan_web_content_from_tool


class ScanWebContentFromToolTests(TestCase):
    """The single fire-and-forget adapter for tool call sites."""

    def _ctx(self, user_id="7", conversation_id="thread-1"):
        return SimpleNamespace(user_id=user_id, conversation_id=conversation_id)

    @patch("guardrails.tasks.scan_web_content_task.delay")
    def test_enqueues_with_text_and_attribution(self, mock_delay):
        scan_web_content_from_tool("some page text", self._ctx(), source_label="web_fetch")
        mock_delay.assert_called_once()
        text, user_id, thread_id, source_label = mock_delay.call_args[0]
        self.assertEqual(text, "some page text")
        self.assertEqual(user_id, "7")
        self.assertEqual(thread_id, "thread-1")
        self.assertEqual(source_label, "web_fetch")

    @patch("guardrails.tasks.scan_web_content_task.delay")
    def test_text_capped_before_enqueue(self, mock_delay):
        long_text = "x" * (_MAX_WEB_SCAN_CHARS + 5000)
        scan_web_content_from_tool(long_text, self._ctx(), source_label="web_fetch")
        (text, *_rest) = mock_delay.call_args[0]
        self.assertEqual(len(text), _MAX_WEB_SCAN_CHARS)

    @patch("guardrails.tasks.scan_web_content_task.delay")
    def test_none_context_does_not_enqueue(self, mock_delay):
        scan_web_content_from_tool("text", None, source_label="web_fetch")
        mock_delay.assert_not_called()

    @patch("guardrails.tasks.scan_web_content_task.delay")
    def test_missing_user_id_does_not_enqueue(self, mock_delay):
        ctx = SimpleNamespace(user_id=None, conversation_id="thread-1")
        scan_web_content_from_tool("text", ctx, source_label="web_fetch")
        mock_delay.assert_not_called()

    @patch("guardrails.tasks.scan_web_content_task.delay")
    def test_empty_text_does_not_enqueue(self, mock_delay):
        scan_web_content_from_tool("", self._ctx(), source_label="web_fetch")
        scan_web_content_from_tool("   ", self._ctx(), source_label="web_fetch")
        mock_delay.assert_not_called()

    @patch(
        "guardrails.tasks.scan_web_content_task.delay",
        side_effect=RuntimeError("broker down"),
    )
    def test_broker_error_is_swallowed(self, mock_delay):
        # Never raises — a missed observation must not break the calling tool.
        scan_web_content_from_tool("text", self._ctx(), source_label="web_fetch")


class ResolveOrgIdTests(TransactionTestCase):
    """_resolve_org_id maps a user to their membership org (used by the task)."""

    def test_returns_membership_org(self):
        user = User.objects.create_user(email="orgtest@example.com", password="test1234")
        org = Organization.objects.create(name="Org", slug="org")
        Membership.objects.create(user=user, org=org)
        self.assertEqual(_resolve_org_id(user.pk), org.pk)

    def test_returns_none_without_membership(self):
        user = User.objects.create_user(email="noorg@example.com", password="test1234")
        self.assertIsNone(_resolve_org_id(user.pk))
