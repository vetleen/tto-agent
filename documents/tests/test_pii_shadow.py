"""Tests for the nightly PII shadow comparison (documents.services.pii_shadow)."""

import os
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.db.utils import OperationalError
from django.test import SimpleTestCase, TestCase, override_settings

from accounts.models import Membership, Organization, User
from core.preferences import DecisionsSettings
from documents.models import (
    DataRoom,
    DataRoomDocument,
    DataRoomDocumentTag,
    DataRoomDocumentVersion,
    PIIReviewEvent,
    PIIScanShadowResult,
    PIIScanShadowRun,
)
from documents.services.pii_scan import GATED_CATEGORIES, PII_CATEGORIES, _PII_SYSTEM_PROMPT
from documents.services.pii_shadow import (
    SHADOW_FEATURE_KEY,
    build_input,
    build_questions,
    run_pii_shadow_for_day,
    select_version_ids,
)
from documents.tests._helpers import make_version
from guardrails.shadow import day_bounds, shadow_availability
from llm.models import LLMCallLog
from llm.types.decisions import CallRecord, DecisionsResult, PredicateAnswer, RefusalAnswer
from llm.types.responses import Usage

DAY = date(2026, 10, 25)
READY = DataRoomDocument.Status.READY


def _decision(article_9=False, article_10=False, confidence=0.9):
    from llm.types.structured import PIIReviewDecision

    return PIIReviewDecision(article_9=article_9, article_10=article_10, confidence=confidence, reasoning="r", findings="f")


class ShapeTests(SimpleTestCase):
    def test_ten_predicates_named_after_the_categories(self):
        questions = build_questions()
        self.assertEqual([q.name for q in questions], list(PII_CATEGORIES))
        self.assertTrue(all(q.type == "predicate" for q in questions))
        self.assertTrue(questions[0].instructions.startswith("The text contains personal identity data"))
        self.assertTrue(questions[0].instructions.endswith("."))
        gated = {q.name: q.instructions for q in questions}
        self.assertTrue(gated["pii_special_category"].startswith("The text contains GDPR Article 9"))
        self.assertTrue(gated["pii_criminal_offence"].startswith("The text contains GDPR Article 10"))

    def test_input_is_the_pii_prompt_plus_the_window(self):
        messages = build_input("Alice Smith, born 1980.")
        self.assertEqual(messages[0].role, "developer")
        self.assertEqual(messages[0].content[0].text, _PII_SYSTEM_PROMPT)
        self.assertEqual(messages[1].role, "user")
        self.assertEqual(messages[1].content[0].text, "Alice Smith, born 1980.")

    def test_input_applies_the_production_truncation(self):
        with patch("documents.services.description._prepare_document_text", return_value="truncated") as prep:
            messages = build_input("x" * 100)
        prep.assert_called_once_with("x" * 100)
        self.assertEqual(messages[1].content[0].text, "truncated")


@override_settings(PII_SCAN_WINDOW_TOKENS=10, PII_SHADOW_MAX_WINDOWS=500, PII_SHADOW_MAX_REVIEWS=20)
class _Base(TestCase):
    """A READY version created on DAY with one chunk per window; the decisions
    service and the PII reviewer mocked; the org eligible for both."""

    def setUp(self):
        self.user = User.objects.create_user(email="pii-shadow@example.com", password="test1234")
        self.org = Organization.objects.create(name="PII Org", slug="pii-shadow-org")
        Membership.objects.create(user=self.user, org=self.org)
        self.room = DataRoom.objects.create(name="PII Room", slug="pii-shadow-room", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user, original_filename="pii.txt", status=READY, name="Register",
        )
        start, _ = day_bounds(DAY)
        self.in_day = start + timedelta(hours=11)
        self.version = self._version(3)

        patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}).start()
        self.resolve = patch("core.preferences.resolve_org_feature_model", return_value="test/model").start()
        self.decisions_settings = patch(
            "core.preferences.resolve_decisions_settings",
            return_value=DecisionsSettings(True, "openai/gpt-6-luna", "", None),
        ).start()
        self.service = MagicMock()
        self.service.decide.side_effect = self._decide
        patch("guardrails.shadow.get_shadow_decisions_service", return_value=self.service).start()
        self.reviewer_llm = patch("documents.services.pii_review._get_llm_service").start().return_value
        self.reviewer_llm.run_structured.side_effect = self._review
        self.review_requests = []
        self.review_decision = _decision(article_9=True)
        self.addCleanup(patch.stopall)
        shadow_availability().reset()
        self.addCleanup(shadow_availability().reset)

        # Per decide call (windows in order): {category: probability | None}; default all 0.05.
        self.window_probs: list[dict] = []
        self.pathway = "quick"
        self.log_cost = None  # when set, the fake decide writes an LLMCallLog row with this cost

    def _version(self, n_windows, *, status=READY, created_at=None, doc=None, chunks=True, version_index=0):
        version = make_version(
            doc or self.doc, status=status, version_index=version_index,
            chunks=[{"text": f"window text {i}", "token_count": 10} for i in range(n_windows)] if chunks else None,
        )
        DataRoomDocumentVersion.objects.filter(pk=version.pk).update(created_at=created_at or self.in_day)
        version.refresh_from_db()
        return version

    def _decide(self, questions, input, **kwargs):
        probs = self.window_probs.pop(0) if self.window_probs else {}
        answers = []
        for q in questions:
            p = probs.get(q.name, 0.05)
            answers.append(RefusalAnswer(name=q.name) if p is None else PredicateAnswer(name=q.name, probability=p))
        if self.log_cost is not None:
            LLMCallLog.objects.create(model="openai/gpt-6-luna", prompt=[], trace_id=kwargs["context"].trace_id, cost_usd=self.log_cost)
        return DecisionsResult(
            answers=answers, usage=Usage(prompt_tokens=100),
            calls=[CallRecord(pathway=self.pathway, model="openai/gpt-6-luna", questions=len(questions), duration_ms=4)],
        )

    def _review(self, request, schema):
        self.review_requests.append(request)
        return self.review_decision, None

    def _event(self, window_index, candidates, action="dismissed", article_9=False, article_10=False, version=None):
        return PIIReviewEvent.objects.create(
            document=self.doc, version_id=(version or self.version).pk, user_id=self.user.pk, org_id=self.org.pk,
            window_index=window_index, candidate_categories=list(candidates), action=action,
            article_9=article_9, article_10=article_10, confidence=0.9, excerpt="x",
        )

    def _tag(self, category, version=None):
        return DataRoomDocumentTag.objects.create(version=version or self.version, key=category, value="true")

    def _rows(self):
        return list(PIIScanShadowResult.objects.filter(run_date=DAY).order_by("version_id", "window_index"))


class SelectionTests(_Base):
    def test_ready_in_day_versions_with_chunks(self):
        other_doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user, original_filename="o.txt", status=READY,
        )
        yesterday = self._version(1, doc=other_doc, created_at=self.in_day - timedelta(days=1), version_index=0)
        processing = self._version(1, doc=other_doc, status=DataRoomDocument.Status.PROCESSING, version_index=1)
        empty = self._version(0, doc=other_doc, chunks=False, version_index=2)
        also = self._version(2, doc=other_doc, version_index=3)
        ids = select_version_ids(DAY)
        self.assertIn(self.version.pk, ids)
        self.assertIn(also.pk, ids)
        for excluded in (yesterday, processing, empty):
            self.assertNotIn(excluded.pk, ids)


class RunTests(_Base):
    def test_rows_per_window_with_production_verdicts(self):
        # Production: identity tagged; window 1 had a special-category candidate the reviewer dismissed.
        self._tag("pii_ordinary_identity")
        self._event(1, ["pii_special_category"], action="dismissed")
        self.window_probs = [
            {"pii_ordinary_identity": 0.9},
            {"pii_special_category": 0.8},                # prod candidate here → no second review
            {"pii_special_category": 0.7, "pii_ordinary_identity": None},  # quick-only → blind review; one refusal
        ]
        summary = run_pii_shadow_for_day(DAY)

        rows = self._rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual([r.window_index for r in rows], [0, 1, 2])
        self.assertEqual([(r.chunk_index_from, r.chunk_index_to) for r in rows], [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(rows[0].window_tokens, 10)
        self.assertEqual(rows[0].window_chars, len("window text 0"))
        self.assertEqual(rows[0].org_id, self.org.pk)
        self.assertEqual(rows[0].document_id, self.doc.pk)
        self.assertAlmostEqual(rows[0].quick_probabilities["pii_ordinary_identity"], 0.9)
        self.assertAlmostEqual(rows[0].quick_probabilities["pii_special_category"], 0.05)
        self.assertEqual(rows[0].prod_version_categories, ["pii_ordinary_identity"])
        self.assertEqual(rows[0].prod_candidates, [])
        self.assertEqual(rows[0].blind_review, "")
        # Window 1: production's own (blind) review is recorded; no shadow review.
        self.assertEqual(rows[1].prod_candidates, ["pii_special_category"])
        self.assertEqual(rows[1].prod_action, "dismissed")
        self.assertIs(rows[1].prod_article_9, False)
        self.assertEqual(rows[1].blind_review, "")
        # Window 2: quick-only gated positive → blind review with the quick candidates.
        self.assertIsNone(rows[2].quick_probabilities["pii_ordinary_identity"])
        self.assertEqual(rows[2].blind_review, "reviewed")
        self.assertEqual(rows[2].blind_candidates, ["pii_special_category"])
        self.assertTrue(rows[2].blind_article_9)
        self.assertIs(rows[2].blind_article_10, False)
        self.assertAlmostEqual(rows[2].blind_confidence, 0.9)
        self.assertEqual(self.reviewer_llm.run_structured.call_count, 1)
        prompt = self.review_requests[0].messages[1].content
        self.assertIn("Article 9 (special category)", prompt)
        self.assertNotIn("Article 10", prompt)
        self.assertIn("window text 2", prompt)
        self.assertIn("Register", prompt)
        self.assertEqual(self.review_requests[0].context.trace_id, summary["trace_id"])

        self.assertEqual(summary["windows"], 3)
        self.assertEqual(summary["versions"], 1)
        self.assertEqual(summary["orgs"], 1)
        self.assertEqual(summary["refusals"], 1)
        self.assertEqual(summary["reviews"], 1)
        self.assertEqual(summary["blind_confirmed"], 1)
        self.assertEqual(summary["quick_gated_windows"], 2)
        self.assertEqual(summary["prod_gated_windows"], 1)
        run = PIIScanShadowRun.objects.get()
        self.assertEqual(run.summary["windows"], 3)
        self.assertIsNotNone(run.finished_at)
        # The decide call used the shadow feature key, the PII prompt and the run's trace id.
        kwargs = self.service.decide.call_args.kwargs
        self.assertEqual(kwargs["feature_key"], SHADOW_FEATURE_KEY)
        self.assertEqual(kwargs["org_id"], self.org.pk)
        self.assertEqual(kwargs["context"].trace_id, summary["trace_id"])
        self.assertEqual(self.service.decide.call_args.args[1][0].content[0].text, _PII_SYSTEM_PROMPT)

    def test_gated_positive_already_tagged_on_the_version_is_not_reviewed(self):
        self._tag("pii_special_category")
        self.window_probs = [{"pii_special_category": 0.9}]
        run_pii_shadow_for_day(DAY)
        self.assertEqual(self._rows()[0].blind_review, "")
        self.reviewer_llm.run_structured.assert_not_called()

    def test_both_gated_categories_reviewed_together(self):
        self.review_decision = _decision(article_9=False, article_10=True)
        self.window_probs = [{"pii_special_category": 0.6, "pii_criminal_offence": 0.7}]
        summary = run_pii_shadow_for_day(DAY)
        row = self._rows()[0]
        self.assertEqual(row.blind_candidates, list(GATED_CATEGORIES))
        self.assertIs(row.blind_article_9, False)
        self.assertTrue(row.blind_article_10)
        self.assertEqual(summary["blind_confirmed"], 1)
        prompt = self.review_requests[0].messages[1].content
        self.assertIn("Article 9", prompt)
        self.assertIn("Article 10", prompt)

    def test_blind_dismissal_counted(self):
        self.review_decision = _decision(article_9=False)
        self.window_probs = [{"pii_special_category": 0.9}]
        summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["blind_dismissed"], 1)
        self.assertEqual(summary.get("blind_confirmed", 0), 0)

    def test_review_cap(self):
        self.window_probs = [{"pii_special_category": 0.9}, {"pii_special_category": 0.9}]
        run_pii_shadow_for_day(DAY, max_reviews=1)
        rows = self._rows()
        self.assertEqual(rows[0].blind_review, "reviewed")
        self.assertEqual(rows[1].blind_review, "not_reviewed")
        self.assertEqual(self.reviewer_llm.run_structured.call_count, 1)

    def test_window_cap(self):
        summary = run_pii_shadow_for_day(DAY, max_windows=2)
        self.assertEqual(len(self._rows()), 2)
        self.assertEqual(summary["skipped"], {"cap_windows": 1})
        self.assertEqual(self.service.decide.call_count, 2)

    def test_rerun_is_idempotent_per_window(self):
        run_pii_shadow_for_day(DAY, max_windows=2)
        self.assertEqual(self.service.decide.call_count, 2)
        run_pii_shadow_for_day(DAY)
        self.assertEqual(self.service.decide.call_count, 3)  # only the missing window
        self.assertEqual(len(self._rows()), 3)

    def test_reviewer_error_and_no_model(self):
        self.window_probs = [{"pii_special_category": 0.9}, {"pii_special_category": 0.9}]
        self.reviewer_llm.run_structured.side_effect = RuntimeError("reviewer down")
        run_pii_shadow_for_day(DAY, max_windows=1)
        self.assertEqual(self._rows()[0].blind_review, "error")
        PIIScanShadowResult.objects.all().delete()
        self.resolve.return_value = ""  # no pii_reviewer model (and no pii_scan model → org skipped)
        summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["skipped"], {"pii_off": 1})

    def test_no_reviewer_model_records_no_model(self):
        def resolve(org_id, feature_key):
            return "" if feature_key == "pii_reviewer" else "test/model"

        self.resolve.side_effect = resolve
        self.window_probs = [{"pii_special_category": 0.9}]
        run_pii_shadow_for_day(DAY, max_windows=1)
        self.assertEqual(self._rows()[0].blind_review, "no_model")

    def test_decide_errors_isolated_then_abort(self):
        calls = {"n": 0}

        def flaky(questions, input, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("endpoint down")
            return self._decide(questions, input, **kwargs)

        self.service.decide.side_effect = flaky
        summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["decide_errors"], 1)
        self.assertEqual(len(self._rows()), 2)  # the failed window wrote no row
        self.assertEqual([r.window_index for r in self._rows()], [1, 2])

        PIIScanShadowResult.objects.all().delete()
        self.service.decide.side_effect = RuntimeError("down")
        summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["decide_errors"], 3)
        self.assertEqual(summary["skipped"], {"aborted": 1})
        self.assertEqual(len(self._rows()), 0)

    def test_non_quick_pathway_recorded(self):
        self.pathway = "structured"
        summary = run_pii_shadow_for_day(DAY, max_windows=1)
        self.assertEqual(self._rows()[0].quick_pathway, "structured")
        self.assertEqual(summary["non_quick"], 1)

    def test_org_gates(self):
        with patch("core.preferences.resolve_decisions_settings",
                   return_value=DecisionsSettings(False, "openai/gpt-6-luna", "", None)):
            summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["skipped"], {"quick_off": 1})
        self.org.preferences = {"pii_scan_enabled": False}
        self.org.save(update_fields=["preferences"])
        summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["skipped"], {"pii_off": 1})
        self.service.decide.assert_not_called()

    def test_suspended_window_skips(self):
        with patch.object(shadow_availability(), "is_suspended", return_value=True):
            summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["skipped"], {"quick_suspended": 3})
        self.service.decide.assert_not_called()

    def test_no_openai_key_skips(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["skipped"], {"no_openai_key": 1})
        self.assertEqual(PIIScanShadowRun.objects.count(), 1)

    def test_db_unavailable_returns_without_a_run(self):
        with patch("documents.services.pii_shadow.select_version_ids", side_effect=OperationalError("down")):
            summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["skipped"], {"db_unavailable": 1})

    def test_side_effect_free(self):
        self._tag("pii_ordinary_identity")
        self._event(0, ["pii_special_category"], action="dismissed")
        before = (
            PIIReviewEvent.objects.count(), DataRoomDocumentTag.objects.count(),
            DataRoomDocumentVersion.objects.filter(pk=self.version.pk).values("status", "is_quarantined", "quarantine_reason")[0],
        )
        self.window_probs = [{"pii_special_category": 0.9, "pii_criminal_offence": 0.9}] * 3
        run_pii_shadow_for_day(DAY)
        after = (
            PIIReviewEvent.objects.count(), DataRoomDocumentTag.objects.count(),
            DataRoomDocumentVersion.objects.filter(pk=self.version.pk).values("status", "is_quarantined", "quarantine_reason")[0],
        )
        self.assertEqual(before, after)
        self.assertEqual(self.reviewer_llm.run_structured.call_count, 3)

    def test_cost_summed_from_llm_rows_by_trace_id(self):
        self.log_cost = "0.0020"
        summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["llm_calls"], 3)
        self.assertAlmostEqual(summary["cost_usd"], 0.006)
        self.assertEqual(LLMCallLog.objects.filter(trace_id=summary["trace_id"]).count(), 3)

    def test_no_text_stored(self):
        run_pii_shadow_for_day(DAY, max_windows=1)
        row = self._rows()[0]
        for field in row._meta.fields:
            value = getattr(row, field.name)
            if isinstance(value, str):
                self.assertNotIn("window text", value)
            else:
                self.assertNotIn("window text", str(value))


class NightlyTaskTests(TestCase):
    @override_settings(PII_SHADOW_ENABLED=False)
    def test_disabled(self):
        from documents.tasks import shadow_pii_scan_nightly

        self.assertEqual(shadow_pii_scan_nightly.apply().get(), "disabled")
        self.assertEqual(PIIScanShadowRun.objects.count(), 0)

    @override_settings(PII_SHADOW_ENABLED=True)
    def test_runs_the_previous_local_day(self):
        from django.utils import timezone

        from documents.tasks import shadow_pii_scan_nightly

        with patch("documents.services.pii_shadow.run_pii_shadow_for_day", return_value={"windows": 0}) as run:
            self.assertEqual(shadow_pii_scan_nightly.apply().get(), {"windows": 0})
        run.assert_called_once_with(timezone.localdate() - timedelta(days=1))

    def test_beat_schedule_entry_present(self):
        entry = settings.CELERY_BEAT_SCHEDULE["shadow-pii-scan-nightly"]
        self.assertEqual(entry["task"], "documents.tasks.shadow_pii_scan_nightly")
        self.assertTrue(hasattr(entry["schedule"], "hour"))
