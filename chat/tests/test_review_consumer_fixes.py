"""Tests for the Group C consumer fixes from the chat-app review.

Focused on the cleanly unit-testable piece: attachment-id validation (C4). The
stream/seed/guardrail races (C1/C2) are exercised by test_stop / test_redaction /
test_subagent; the sync sub-agent reported_at fix (C3) is asserted in
test_subagent.CreateSubagentToolTimeoutTests.
"""
from __future__ import annotations

import uuid

from django.test import SimpleTestCase

from chat.consumers import _valid_uuids


class ValidUuidsTests(SimpleTestCase):
    def test_keeps_valid_drops_invalid(self):
        good = str(uuid.uuid4())
        self.assertEqual(_valid_uuids([good, "not-a-uuid", "", None]), [good])

    def test_empty_and_none(self):
        self.assertEqual(_valid_uuids([]), [])
        self.assertEqual(_valid_uuids(None), [])

    def test_coerces_uuid_objects_to_str(self):
        u = uuid.uuid4()
        self.assertEqual(_valid_uuids([u]), [str(u)])

    def test_all_invalid_yields_empty(self):
        # The brick scenario: a non-UUID must be dropped, never persisted/loaded.
        self.assertEqual(_valid_uuids(["abc", 123, {"x": 1}]), [])
