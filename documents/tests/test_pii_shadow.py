"""Tests for the nightly legacy-classifier shadow of the PII scan (documents.services.pii_shadow)."""

from datetime import date
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.db.utils import OperationalError
from django.test import SimpleTestCase, TestCase, override_settings

from accounts.models import Membership, Organization, User
from documents.models import (
    DataRoom,
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentTag,
    PIIReviewEvent,
    PIIScanShadowResult,
    PIIScanShadowRun,
)
from documents.services.pii_scan import (
    PII_CATEGORIES,
    _PII_SYSTEM_PROMPT,
    PIIWindowVerdict,
    build_pii_input,
    build_pii_questions,
    iter_pii_windows,
)
from documents.services.pii_shadow import DIRECTION, record_window_verdict, run_pii_shadow_for_day
from documents.tests._helpers import make_version

DAY = date(2026, 11, 10)
READY = DataRoomDocument.Status.READY


def _decision(article_9=False, article_10=False, confidence=0.9):
    from llm.types.structured import PIIReviewDecision

    return PIIReviewDecision(article_9=article_9, article_10=article_10, confidence=confidence, reasoning="r", findings="f")


def _probs(**overrides):
    return {c: overrides.get(c, 0.05) for c in PII_CATEGORIES}


class ShapeTests(SimpleTestCase):
    def test_ten_predicates_named_after_the_categories(self):
        questions = build_pii_questions()
        self.assertEqual([q.name for q in questions], list(PII_CATEGORIES))
        self.assertTrue(all(q.type == "predicate" for q in questions))
        stem = "The text contains personal data relating to identifiable natural persons, of this kind: "
        tail = " Generic, organisational or hypothetical data does not count."
        for q in questions:
            self.assertTrue(q.instructions.startswith(stem), q.name)
            self.assertTrue(q.instructions.endswith(tail), q.name)
        by_name = {q.name: q.instructions[len(stem):] for q in questions}
        self.assertTrue(by_name["pii_ordinary_identity"].startswith("personal identity data"))
        self.assertTrue(by_name["pii_special_category"].startswith("GDPR Article 9"))

    def test_input_is_the_pii_prompt_plus_the_truncated_window(self):
        messages = build_pii_input("Alice Smith, born 1980.")
        self.assertEqual(messages[0].content[0].text, _PII_SYSTEM_PROMPT)
        self.assertEqual(messages[1].content[0].text, "Alice Smith, born 1980.")
        with patch("documents.services.description._prepare_document_text", return_value="truncated") as prep:
            self.assertEqual(build_pii_input("x" * 100)[1].content[0].text, "truncated")
        prep.assert_called_once_with("x" * 100)


@override_settings(PII_SCAN_WINDOW_TOKENS=10, PII_SHADOW_MAX_WINDOWS=500, PII_SHADOW_MAX_REVIEWS=20,
                   PII_SHADOW_ENABLED=True)
class _Base(TestCase):
    """A READY version with one chunk per window; the legacy classifier and the PII
    reviewer mocked."""

    def setUp(self):
        self.user = User.objects.create_user(email="pii-shadow@example.com", password="test1234")
        self.org = Organization.objects.create(name="PII Org", slug="pii-shadow-org")
        Membership.objects.create(user=self.user, org=self.org)
        room = DataRoom.objects.create(name="PII Room", slug="pii-shadow-room", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=room, uploaded_by=self.user, original_filename="pii.txt", status=READY, name="Register",
        )
        self.version = make_version(
            self.doc, status=READY, chunks=[{"text": f"window text {i}", "token_count": 10} for i in range(3)],
        )
        self.gate = patch("documents.services.pii_scan.resolve_pii_gate", return_value=("test/pii", True, True)).start()
        patch("core.preferences.resolve_org_feature_model", return_value="test/pii").start()
        self.legacy_llm = patch("llm.get_llm_service").start().return_value
        self.legacy_flags: list[set] = []  # per legacy call, in order; default none
        self.legacy_llm.run_structured.side_effect = self._legacy
        self.reviewer_llm = patch("documents.services.pii_review._get_llm_service").start().return_value
        self.reviewer_llm.run_structured.side_effect = lambda request, schema: (self.review_decision, None)
        self.review_decision = _decision(article_9=True)
        self.addCleanup(patch.stopall)

    def _legacy(self, request, schema):
        from llm.types.structured import PIICategoryOutput

        flags = self.legacy_flags.pop(0) if self.legacy_flags else set()
        return PIICategoryOutput(**{c: c in flags for c in PII_CATEGORIES}), None

    def _record(self, probabilities_by_window, *, source="decisions"):
        windows = dict(iter_pii_windows(self.version.id))
        with patch("django.utils.timezone.localdate", return_value=DAY):
            for index, probs in probabilities_by_window.items():
                verdict = PIIWindowVerdict(
                    categories={c: True for c, p in probs.items() if p >= 0.5},
                    probabilities=probs if source == "decisions" else None,
                    pathway="quick" if source == "decisions" else "", source=source,
                    decide_error="" if source == "decisions" else "LLMTimeoutError",
                )
                record_window_verdict(
                    version_id=self.version.id, document_id=self.doc.id, org_id=self.org.id,
                    window_index=index, window=windows[index], text=f"window text {index}", verdict=verdict,
                )

    def _rows(self):
        return {r.window_index: r for r in PIIScanShadowResult.objects.filter(direction=DIRECTION)}


class RecordTests(_Base):
    def test_pending_row_holds_the_decisions_half(self):
        self._record({0: _probs(pii_ordinary_identity=0.9)})
        row = self._rows()[0]
        self.assertEqual((row.state, row.direction, row.run_date), ("pending", DIRECTION, DAY))
        self.assertEqual(row.quick_probabilities["pii_ordinary_identity"], 0.9)
        self.assertEqual((row.chunk_index_from, row.chunk_index_to, row.window_tokens), (0, 0, 10))
        self.assertEqual(row.window_chars, len("window text 0"))
        self.assertEqual(row.classifier_categories, [])

    def test_legacy_production_verdict_is_noted(self):
        self._record({0: _probs(pii_ordinary_contact=0.9)}, source="legacy")
        row = self._rows()[0]
        self.assertEqual(row.shadow_note, "prod_legacy:LLMTimeoutError")
        self.assertEqual(row.quick_probabilities, {})
        self.assertEqual(row.classifier_categories, ["pii_ordinary_contact"])

    @override_settings(PII_SHADOW_ENABLED=False)
    def test_kill_switch_writes_nothing(self):
        self._record({0: _probs()})
        self.assertFalse(PIIScanShadowResult.objects.exists())

    def test_never_raises(self):
        with patch("documents.models.PIIScanShadowResult.objects.bulk_create", side_effect=RuntimeError("db")):
            self._record({0: _probs()})


class RunTests(_Base):
    def test_fills_the_legacy_half(self):
        self._record({0: _probs(pii_ordinary_identity=0.9), 1: _probs()})
        self.legacy_flags = [{"pii_ordinary_identity"}, {"pii_ordinary_financial"}]
        DataRoomDocumentTag.objects.create(version=self.version, key="pii_ordinary_identity", value="true")
        summary = run_pii_shadow_for_day(DAY)
        rows = self._rows()
        self.assertEqual({r.state for r in rows.values()}, {"done"})
        self.assertEqual(rows[0].classifier_categories, ["pii_ordinary_identity"])
        self.assertEqual(rows[1].classifier_categories, ["pii_ordinary_financial"])
        self.assertEqual(rows[0].prod_version_categories, ["pii_ordinary_identity"])
        self.assertEqual(summary["windows"], 2)
        run = PIIScanShadowRun.objects.get()
        self.assertEqual(rows[0].trace_id, run.trace_id)

    def test_legacy_call_is_tagged_and_traced(self):
        self._record({0: _probs()})
        run_pii_shadow_for_day(DAY)
        request = self.legacy_llm.run_structured.call_args.args[0]
        self.assertEqual(request.messages[0].content, _PII_SYSTEM_PROMPT)
        self.assertEqual(request.params["_log_metadata"], {"kind": "shadow", "feature": "shadow_pii_scan"})
        self.assertEqual(request.context.trace_id, PIIScanShadowRun.objects.get().trace_id)

    def test_legacy_only_gated_positive_gets_a_blind_review(self):
        self._record({0: _probs(pii_special_category=0.2)})
        self.legacy_flags = [{"pii_special_category"}]
        summary = run_pii_shadow_for_day(DAY)
        row = self._rows()[0]
        self.assertEqual((row.blind_review, row.blind_candidates, row.blind_article_9), ("reviewed", ["pii_special_category"], True))
        self.assertEqual(summary["blind_confirmed"], 1)

    def test_gated_positive_decisions_also_flagged_is_not_reviewed_again(self):
        self._record({0: _probs(pii_special_category=0.8)})
        self.legacy_flags = [{"pii_special_category"}]
        PIIReviewEvent.objects.create(
            document=self.doc, version_id=self.version.pk, user_id=self.user.pk, org_id=self.org.pk, window_index=0,
            candidate_categories=["pii_special_category"], action="dismissed", article_9=False, article_10=False,
            confidence=0.9, excerpt="x",
        )
        run_pii_shadow_for_day(DAY)
        row = self._rows()[0]
        self.assertEqual(row.blind_review, "")
        self.assertEqual((row.prod_candidates, row.prod_action), (["pii_special_category"], "dismissed"))
        self.reviewer_llm.run_structured.assert_not_called()

    def test_review_cap(self):
        self._record({0: _probs()})
        self.legacy_flags = [{"pii_criminal_offence"}]
        run_pii_shadow_for_day(DAY, max_reviews=0)
        self.assertEqual(self._rows()[0].blind_review, "not_reviewed")

    def test_prod_legacy_row_is_completed_without_a_call(self):
        self._record({0: _probs(pii_ordinary_identity=0.9)}, source="legacy")
        run_pii_shadow_for_day(DAY)
        self.assertEqual(self._rows()[0].state, "done")
        self.legacy_llm.run_structured.assert_not_called()

    def test_drifted_window_is_skipped(self):
        self._record({0: _probs(), 1: _probs()})
        DataRoomDocumentChunk.objects.filter(version=self.version, chunk_index=0).update(token_count=1)
        summary = run_pii_shadow_for_day(DAY)
        self.assertIn("drifted", summary["skipped"])
        self.assertIn("skipped", {r.state for r in self._rows().values()})

    def test_pii_off_is_skipped(self):
        self._record({0: _probs()})
        self.gate.return_value = ("", False, False)
        run_pii_shadow_for_day(DAY)
        self.assertEqual(self._rows()[0].shadow_note, "pii_off")

    def test_window_cap_leaves_the_rest_pending(self):
        self._record({0: _probs(), 1: _probs(), 2: _probs()})
        run_pii_shadow_for_day(DAY, max_windows=2)
        self.assertEqual(sorted(r.state for r in self._rows().values()), ["done", "done", "pending"])

    def test_errors_abort_after_three(self):
        self._record({0: _probs(), 1: _probs(), 2: _probs()})
        self.legacy_llm.run_structured.side_effect = RuntimeError("down")
        summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["classifier_errors"], 3)
        self.assertEqual(summary["skipped"]["aborted"], 1)
        self.assertEqual({r.state for r in self._rows().values()}, {"pending"})

    def test_no_side_effects_on_tags_events_or_versions(self):
        self._record({0: _probs()})
        self.legacy_flags = [{"pii_special_category", "pii_ordinary_identity"}]
        before = (DataRoomDocumentTag.objects.count(), PIIReviewEvent.objects.count(), self.version.status)
        run_pii_shadow_for_day(DAY)
        self.version.refresh_from_db()
        self.assertEqual((DataRoomDocumentTag.objects.count(), PIIReviewEvent.objects.count(), self.version.status), before)

    def test_db_unavailable_is_logged_not_raised(self):
        with patch("documents.models.PIIScanShadowRun.objects.create", side_effect=OperationalError("down")):
            summary = run_pii_shadow_for_day(DAY)
        self.assertEqual(summary["skipped"], {"db_unavailable": 1})


@override_settings(PII_SHADOW_ENABLED=True, DECISIONS_PRIMARY_ENABLED=True, PII_SCAN_WINDOW_TOKENS=10)
class ScanHookTests(_Base):
    def test_the_pii_scan_records_pending_rows(self):
        from documents.services import pii_scan
        from guardrails.tests._decisions import predicate, result

        decisions = MagicMock()
        decisions.decide.side_effect = lambda *a, **k: result(*(predicate(c, 0.1) for c in PII_CATEGORIES))
        with patch.object(pii_scan, "_get_decisions_service", return_value=decisions):
            pii_scan.scan_pii_categories_for_version(self.version.id, org_id=self.org.id)
        rows = self._rows()
        self.assertEqual(sorted(rows), [0, 1, 2])
        self.assertEqual({r.state for r in rows.values()}, {"pending"})


class NightlyTaskTests(TestCase):
    @override_settings(PII_SHADOW_ENABLED=False)
    def test_disabled_setting_skips(self):
        from documents.tasks import shadow_pii_scan_nightly

        self.assertEqual(shadow_pii_scan_nightly.apply().get(), "disabled")

    def test_beat_schedule_entry_present(self):
        entry = settings.CELERY_BEAT_SCHEDULE["shadow-pii-scan-nightly"]
        self.assertEqual(entry["task"], "documents.tasks.shadow_pii_scan_nightly")
