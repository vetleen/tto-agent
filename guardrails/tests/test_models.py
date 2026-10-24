"""Tests for guardrail models."""

from django.test import TestCase

from accounts.models import Membership, Organization, User
from guardrails.models import GuardrailEvent


class GuardrailEventModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="test@example.com", password="test1234")
        self.org = Organization.objects.create(name="Test Org", slug="test-org")
        self.membership = Membership.objects.create(user=self.user, org=self.org)

    def test_create_event(self):
        event = GuardrailEvent.objects.create(
            user=self.user,
            organization=self.org,
            trigger_source="user_message",
            check_type="heuristic",
            tags=["prompt_injection"],
            confidence=0.95,
            severity="high",
            action_taken="blocked",
            raw_input="ignore previous instructions",
        )
        self.assertIsNotNone(event.id)
        self.assertEqual(event.user, self.user)
        self.assertEqual(event.tags, ["prompt_injection"])
        self.assertEqual(event.action_taken, "blocked")

    def test_event_ordering(self):
        """Events should be ordered by -created_at (most recent first)."""
        e1 = GuardrailEvent.objects.create(
            user=self.user,
            trigger_source="user_message",
            check_type="heuristic",
            tags=[],
            severity="low",
            action_taken="logged",
            raw_input="first",
        )
        e2 = GuardrailEvent.objects.create(
            user=self.user,
            trigger_source="user_message",
            check_type="classifier",
            tags=[],
            severity="medium",
            action_taken="escalated",
            raw_input="second",
        )
        events = list(GuardrailEvent.objects.all())
        self.assertEqual(events[0], e2)
        self.assertEqual(events[1], e1)

    def test_related_event_chain(self):
        """Test escalation chain via related_event."""
        e1 = GuardrailEvent.objects.create(
            user=self.user,
            trigger_source="user_message",
            check_type="classifier",
            tags=["jailbreak"],
            severity="medium",
            action_taken="escalated",
            raw_input="test",
        )
        e2 = GuardrailEvent.objects.create(
            user=self.user,
            trigger_source="user_message",
            check_type="llm_review",
            tags=["jailbreak"],
            severity="high",
            action_taken="blocked",
            raw_input="test",
            related_event=e1,
        )
        self.assertEqual(e2.related_event, e1)
        self.assertEqual(list(e1.escalations.all()), [e2])

    def test_chunk_reference_round_trips(self):
        """document_chunk events can name their chunk; other events leave it empty."""
        event = GuardrailEvent.objects.create(
            user=self.user,
            trigger_source="document_chunk",
            check_type="classifier",
            tags=[],
            severity="low",
            action_taken="logged",
            raw_input="chunk text",
            chunk_id=123456789012,
            version_id=42,
        )
        event.refresh_from_db()
        self.assertEqual(event.chunk_id, 123456789012)
        self.assertEqual(event.version_id, 42)
        other = GuardrailEvent.objects.create(
            user=self.user, trigger_source="user_message", check_type="heuristic",
            tags=[], severity="low", action_taken="logged", raw_input="msg",
        )
        self.assertIsNone(other.chunk_id)
        self.assertIsNone(other.version_id)


class ChunkScanShadowResultModelTest(TestCase):
    def test_unique_per_day_and_chunk(self):
        from datetime import date

        from django.db import IntegrityError, transaction

        from guardrails.models import ChunkScanShadowResult

        def row():
            return ChunkScanShadowResult(
                run_date=date(2026, 10, 23), version_id=1, chunk_id=99, chunk_index=0,
                chunk_chars=10, classifier_flagged=False, quick_probability=0.1,
                quick_pathway="quick",
            )

        row().save()
        with transaction.atomic(), self.assertRaises(IntegrityError):
            row().save()
        # A different day for the same chunk is fine.
        ChunkScanShadowResult.objects.create(
            run_date=date(2026, 10, 24), version_id=1, chunk_id=99, chunk_index=0,
            chunk_chars=10, classifier_flagged=False, quick_probability=0.1, quick_pathway="quick",
        )
        self.assertEqual(ChunkScanShadowResult.objects.filter(chunk_id=99).count(), 2)


class MembershipSuspensionTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="test@example.com", password="test1234")
        self.org = Organization.objects.create(name="Test Org", slug="test-org")
        self.membership = Membership.objects.create(user=self.user, org=self.org)

    def test_default_not_suspended(self):
        self.assertFalse(self.membership.is_suspended)
        self.assertIsNone(self.membership.suspended_at)
        self.assertEqual(self.membership.suspended_reason, "")

    def test_suspend_membership(self):
        from django.utils import timezone
        now = timezone.now()
        self.membership.is_suspended = True
        self.membership.suspended_at = now
        self.membership.suspended_reason = "Automated guardrail suspension"
        self.membership.save()
        self.membership.refresh_from_db()
        self.assertTrue(self.membership.is_suspended)
        self.assertEqual(self.membership.suspended_reason, "Automated guardrail suspension")
