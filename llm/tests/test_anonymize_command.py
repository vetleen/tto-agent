"""Tests for the LLMCallLogIdentity target of enforce_retention: rows older than
LLM_LOG_IDENTITY_RETENTION_DAYS lose their user and object references but keep
model, tokens, cost and the organization."""

from datetime import timedelta
from decimal import Decimal
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.db.models import Sum
from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.models import Membership, Organization
from llm.models import LLMCallLog

User = get_user_model()


def _make_log(age_days, user=None, **fields):
    defaults = dict(
        model="gpt-5-mini",
        prompt=[{"role": "user", "content": "hi"}],
        raw_output="body",
        run_id="run-1",
        trace_id="trace-1",
        conversation_id="conv-1",
        input_tokens=5,
        output_tokens=7,
        total_tokens=12,
        cost_usd=Decimal("0.25"),
    )
    defaults.update(fields)
    log = LLMCallLog.objects.create(user=user, **defaults)
    LLMCallLog.objects.filter(pk=log.pk).update(
        created_at=timezone.now() - timedelta(days=age_days)
    )
    log.refresh_from_db()
    return log


def _run(*args):
    out = StringIO()
    call_command("enforce_retention", "--target", "LLMCallLogIdentity", *args, stdout=out)
    return out.getvalue()


class AnonymizeLLMLogsTests(TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Org A", slug="org-a")
        self.user = User.objects.create_user(email="anon@example.com", password="pw12345!")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)

    def test_old_row_loses_user_and_references(self):
        log = _make_log(366, user=self.user, document_id=11, data_room_id=22)

        _run()

        log.refresh_from_db()
        self.assertIsNone(log.user_id)
        self.assertEqual(log.run_id, "")
        self.assertEqual(log.trace_id, "")
        self.assertEqual(log.conversation_id, "")
        self.assertIsNone(log.document_id)
        self.assertIsNone(log.data_room_id)

    def test_org_stamped_and_metadata_kept(self):
        log = _make_log(366, user=self.user)
        created_at = log.created_at

        _run()

        log.refresh_from_db()
        self.assertEqual(log.org_id, self.org.pk)
        self.assertEqual(log.model, "gpt-5-mini")
        self.assertEqual(log.input_tokens, 5)
        self.assertEqual(log.output_tokens, 7)
        self.assertEqual(log.total_tokens, 12)
        self.assertEqual(log.cost_usd, Decimal("0.25"))
        self.assertEqual(log.created_at, created_at)

    def test_cost_still_aggregates_per_org_and_model(self):
        other_org = Organization.objects.create(name="Org B", slug="org-b")
        other = User.objects.create_user(email="other@example.com", password="pw12345!")
        Membership.objects.create(user=other, org=other_org, role=Membership.Role.MEMBER)
        _make_log(400, user=self.user, cost_usd=Decimal("1.00"))
        _make_log(400, user=self.user, cost_usd=Decimal("2.00"))
        _make_log(400, user=self.user, cost_usd=Decimal("4.00"), model="other-model")
        _make_log(400, user=other, cost_usd=Decimal("8.00"))

        _run()

        rows = {
            (r["org_id"], r["model"]): r["cost"]
            for r in LLMCallLog.objects.values("org_id", "model").annotate(cost=Sum("cost_usd"))
        }
        self.assertEqual(rows, {
            (self.org.pk, "gpt-5-mini"): Decimal("3.00"),
            (self.org.pk, "other-model"): Decimal("4.00"),
            (other_org.pk, "gpt-5-mini"): Decimal("8.00"),
        })

    def test_recent_row_untouched(self):
        log = _make_log(364, user=self.user)

        _run()

        log.refresh_from_db()
        self.assertEqual(log.user_id, self.user.pk)
        self.assertEqual(log.conversation_id, "conv-1")
        self.assertEqual(log.run_id, "run-1")
        self.assertEqual(log.trace_id, "trace-1")
        self.assertIsNone(log.org_id)

    def test_user_without_membership_gets_null_org(self):
        loner = User.objects.create_user(email="loner@example.com", password="pw12345!")
        log = _make_log(366, user=loner)

        _run()

        log.refresh_from_db()
        self.assertIsNone(log.user_id)
        self.assertIsNone(log.org_id)

    def test_existing_org_id_is_kept(self):
        # Stamped at account deletion: no user left to derive it from.
        log = _make_log(366, user=None, org_id=self.org.pk)

        _run()

        log.refresh_from_db()
        self.assertEqual(log.org_id, self.org.pk)
        self.assertEqual(log.conversation_id, "")

    def test_userless_row_with_ids_is_anonymized(self):
        log = _make_log(366, user=None, run_id="", conversation_id="", trace_id="shadow-x")

        _run()

        log.refresh_from_db()
        self.assertEqual(log.trace_id, "")

    def test_content_is_not_touched_by_this_target(self):
        log = _make_log(366, user=self.user)

        _run()

        log.refresh_from_db()
        self.assertEqual(log.raw_output, "body")

    def test_idempotent(self):
        _make_log(366, user=self.user)

        _run()
        out = _run()

        self.assertIn("LLMCallLogIdentity: 0 rows to anonymize", out)

    def test_dry_run_does_not_modify(self):
        log = _make_log(366, user=self.user)

        out = _run("--dry-run")

        log.refresh_from_db()
        self.assertEqual(log.user_id, self.user.pk)
        self.assertEqual(log.conversation_id, "conv-1")
        self.assertIn("1 row(s) to anonymize", out)

    def test_batch_size_processes_all_rows(self):
        ids = [_make_log(366, user=self.user).pk for _ in range(5)]

        _run("--batch-size", "2")

        self.assertEqual(
            LLMCallLog.objects.filter(pk__in=ids, user__isnull=True, org_id=self.org.pk).count(), 5,
        )

    @override_settings(LLM_LOG_IDENTITY_RETENTION_DAYS=10)
    def test_period_is_configurable(self):
        old = _make_log(11, user=self.user)
        recent = _make_log(9, user=self.user)

        _run()

        old.refresh_from_db()
        recent.refresh_from_db()
        self.assertIsNone(old.user_id)
        self.assertEqual(recent.user_id, self.user.pk)

    def test_runs_as_part_of_the_default_command(self):
        log = _make_log(366, user=self.user)

        call_command("enforce_retention", stdout=StringIO())

        log.refresh_from_db()
        self.assertIsNone(log.user_id)
        self.assertEqual(log.org_id, self.org.pk)
        self.assertEqual(log.prompt, {"redacted": True})
