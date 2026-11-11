"""The nightly legacy-classifier shadow of the chunk scan (guardrails/shadow.py)."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.db.utils import OperationalError
from django.test import TestCase, override_settings

from accounts.models import Membership, Organization, User
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk
from documents.tests._helpers import make_version
from guardrails.decisions import PrimaryChunkClassification
from guardrails.models import ChunkScanShadowResult, ChunkScanShadowRun, GuardrailEvent
from guardrails.schemas import BatchClassifierResult, ChunkClassification, ChunkReviewDecision
from guardrails.shadow import (
    DIRECTION,
    SHADOW_REVIEW_PLACEHOLDER,
    production_verdicts,
    record_chunk_verdicts,
    run_shadow_for_day,
)
from guardrails.tasks import CHUNK_SCAN_SYSTEM_PROMPT

DAY = date(2026, 11, 10)


def _verdict(index, p, *, concern="none", source="decisions", decide_error=""):
    flagged = p >= 0.5
    return PrimaryChunkClassification(
        chunk_index=index, is_suspicious=flagged, concern_tags=["prompt_injection"] if flagged else [],
        confidence=p, reasoning="r", probability=p if source == "decisions" else None,
        concern=concern, pathway="quick" if source == "decisions" else "", source=source,
        decide_error=decide_error,
    )


@override_settings(GUARDRAIL_SHADOW_ENABLED=True)
class _ShadowBase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="shadow@example.com", password="test1234")
        self.org = Organization.objects.create(name="Shadow Org", slug="shadow-org")
        Membership.objects.create(user=self.user, org=self.org)
        room = DataRoom.objects.create(name="Shadow Room", slug="shadow-room", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=room, uploaded_by=self.user, original_filename="shadow.txt", status="ready", name="Shadow Doc",
        )
        self.version = make_version(self.doc)
        self.resolve = patch("core.preferences.resolve_org_feature_model", return_value="test/model").start()
        self.legacy_llm = patch("llm.get_llm_service").start().return_value
        self.legacy = {}  # chunk_index -> (flagged, confidence, tags); missing = omitted
        self.legacy_llm.run_structured.side_effect = self._legacy_answer
        self.reviewer_llm = patch("guardrails.reviewer._get_llm_service").start().return_value
        self.reviewer_llm.run_structured.return_value = (
            ChunkReviewDecision(action="allow", confidence=0.9, severity="low", reasoning="benign"), None,
        )
        self.addCleanup(patch.stopall)

    def _legacy_answer(self, request, schema):
        import json
        import re

        payload = json.loads(re.search(r"```json\n(.*)\n```", request.messages[1].content, re.S).group(1))
        results = []
        for item in payload:
            if item["chunk_index"] not in self.legacy:
                continue
            flagged, confidence, tags = self.legacy[item["chunk_index"]]
            results.append(ChunkClassification(
                chunk_index=item["chunk_index"], is_suspicious=flagged, concern_tags=tags,
                confidence=confidence, reasoning="legacy",
            ))
        return BatchClassifierResult(results=results), None

    def _chunk(self, index, text="Normal patent text about semiconductors."):
        return DataRoomDocumentChunk.objects.create(version=self.version, chunk_index=index, text=text, token_count=5)

    def _record(self, chunks, verdicts):
        with patch("django.utils.timezone.localdate", return_value=DAY):
            record_chunk_verdicts(
                version=self.version,
                chunks=[{"id": c.id, "chunk_index": c.chunk_index, "text": c.text} for c in chunks],
                results={v.chunk_index: v for v in verdicts}, org_id=self.org.id,
            )

    def _rows(self):
        return {r.chunk_index: r for r in ChunkScanShadowResult.objects.filter(direction=DIRECTION)}


class RecordTests(_ShadowBase):
    def test_pending_rows_hold_the_decisions_half_only(self):
        a, b = self._chunk(0), self._chunk(1)
        self._record([a, b], [_verdict(0, 0.1), _verdict(1, 0.8, concern="jailbreak")])
        rows = self._rows()
        self.assertEqual({r.state for r in rows.values()}, {"pending"})
        self.assertEqual(rows[1].quick_probability, 0.8)
        self.assertTrue(rows[1].quick_flagged)
        self.assertEqual(rows[1].quick_concern, "jailbreak")
        self.assertEqual(rows[1].quick_pathway, "quick")
        self.assertEqual(rows[1].run_date, DAY)
        self.assertEqual(rows[1].chunk_chars, len(b.text))
        self.assertFalse(rows[1].classifier_flagged)

    def test_legacy_production_verdict_is_noted(self):
        c = self._chunk(0)
        self._record([c], [_verdict(0, 0.9, source="legacy", decide_error="LLMTimeoutError")])
        row = self._rows()[0]
        self.assertIsNone(row.quick_probability)
        self.assertTrue(row.quick_flagged)
        self.assertEqual(row.shadow_note, "prod_legacy:LLMTimeoutError")

    @override_settings(GUARDRAIL_SHADOW_ENABLED=False)
    def test_kill_switch_writes_nothing(self):
        c = self._chunk(0)
        self._record([c], [_verdict(0, 0.9)])
        self.assertFalse(ChunkScanShadowResult.objects.exists())

    def test_never_raises(self):
        c = self._chunk(0)
        with patch("guardrails.models.ChunkScanShadowResult.objects.bulk_create", side_effect=RuntimeError("db")):
            self._record([c], [_verdict(0, 0.9)])  # no exception

    def test_rerecording_the_same_batch_is_ignored(self):
        c = self._chunk(0)
        self._record([c], [_verdict(0, 0.9)])
        self._record([c], [_verdict(0, 0.1)])
        self.assertEqual(ChunkScanShadowResult.objects.count(), 1)


class ProductionVerdictTests(_ShadowBase):
    def test_review_outcomes(self):
        chunks = [self._chunk(i) for i in range(4)]

        def event(chunk, check_type, action):
            return GuardrailEvent.objects.create(
                user=self.user, trigger_source="document_chunk", check_type=check_type, action_taken=action,
                severity="medium", tags=["prompt_injection"], confidence=0.8, raw_input="x",
                chunk_id=chunk.id, version_id=chunk.version_id,
            )

        event(chunks[0], "classifier", "escalated")
        event(chunks[0], "llm_review", "dismissed")
        event(chunks[1], "classifier", "escalated")
        event(chunks[1], "llm_review", "blocked")
        event(chunks[2], "classifier", "blocked")
        verdicts = production_verdicts([c.id for c in chunks])
        self.assertEqual([verdicts[c.id].review for c in chunks], ["allow", "quarantine", "threshold_quarantine", ""])


class RunShadowTests(_ShadowBase):
    def test_fills_the_legacy_half_and_counts(self):
        chunks = [self._chunk(i) for i in range(4)]
        self._record(chunks, [_verdict(0, 0.1), _verdict(1, 0.9), _verdict(2, 0.1), _verdict(3, 0.9)])
        self.legacy = {0: (False, 0.1, []), 1: (True, 0.8, ["jailbreak"]), 2: (True, 0.7, ["prompt_injection"]),
                       3: (False, 0.2, [])}
        summary = run_shadow_for_day(DAY)
        rows = self._rows()
        self.assertEqual({r.state for r in rows.values()}, {"done"})
        self.assertEqual((summary["both_negative"], summary["both_positive"], summary["classifier_only"],
                          summary["quick_only"]), (1, 1, 1, 1))
        self.assertTrue(rows[2].classifier_flagged)
        self.assertEqual(rows[2].classifier_tags, ["prompt_injection"])
        self.assertEqual(rows[2].classifier_confidence, 0.7)
        run = ChunkScanShadowRun.objects.get()
        self.assertEqual(rows[2].trace_id, run.trace_id)
        self.assertEqual(run.summary["direction"], DIRECTION)

    def test_legacy_call_uses_the_production_prompt_and_its_own_trace(self):
        c = self._chunk(0)
        self._record([c], [_verdict(0, 0.1)])
        self.legacy = {0: (False, 0.1, [])}
        run_shadow_for_day(DAY)
        request = self.legacy_llm.run_structured.call_args.args[0]
        self.assertEqual(request.messages[0].content, CHUNK_SCAN_SYSTEM_PROMPT)
        self.assertEqual(request.context.trace_id, ChunkScanShadowRun.objects.get().trace_id)
        self.assertEqual(request.params["_log_metadata"], {"kind": "shadow", "feature": "shadow_chunk_scan"})

    def test_only_legacy_only_positives_get_a_blind_review(self):
        chunks = [self._chunk(i) for i in range(2)]
        self._record(chunks, [_verdict(0, 0.9), _verdict(1, 0.1)])
        self.legacy = {0: (True, 0.9, ["jailbreak"]), 1: (True, 0.75, ["jailbreak"])}
        summary = run_shadow_for_day(DAY)
        rows = self._rows()
        self.assertEqual(rows[0].blind_review, "")  # production's reviewer already judged it
        self.assertEqual((rows[1].blind_review, rows[1].blind_review_source), ("allow", "classifier"))
        self.assertEqual(summary["reviews"], 1)
        prompt = self.reviewer_llm.run_structured.call_args.args[0].messages[1].content
        self.assertIn(SHADOW_REVIEW_PLACEHOLDER, prompt)
        self.assertIn("jailbreak", prompt)

    def test_review_cap_marks_not_reviewed(self):
        c = self._chunk(0)
        self._record([c], [_verdict(0, 0.1)])
        self.legacy = {0: (True, 0.9, ["jailbreak"])}
        run_shadow_for_day(DAY, max_reviews=0)
        self.assertEqual(self._rows()[0].blind_review, "not_reviewed")
        self.reviewer_llm.run_structured.assert_not_called()

    def test_omitted_chunk_is_skipped_and_missing_chunk_is_skipped(self):
        a, b = self._chunk(0), self._chunk(1)
        self._record([a, b], [_verdict(0, 0.1), _verdict(1, 0.1)])
        DataRoomDocumentChunk.objects.filter(pk=b.pk).delete()
        self.legacy = {}
        summary = run_shadow_for_day(DAY)
        rows = self._rows()
        self.assertEqual((rows[0].state, rows[0].shadow_note), ("skipped", "legacy_omitted"))
        self.assertEqual((rows[1].state, rows[1].shadow_note), ("skipped", "chunk_gone"))
        self.assertEqual(summary["skipped"], {"legacy_omitted": 1, "chunk_gone": 1})

    def test_classifier_off_is_skipped(self):
        c = self._chunk(0)
        self._record([c], [_verdict(0, 0.1)])
        self.resolve.return_value = ""
        run_shadow_for_day(DAY)
        self.assertEqual(self._rows()[0].shadow_note, "classifier_off")
        self.legacy_llm.run_structured.assert_not_called()

    def test_chunk_cap_leaves_the_rest_pending(self):
        chunks = [self._chunk(i) for i in range(3)]
        self._record(chunks, [_verdict(i, 0.1) for i in range(3)])
        self.legacy = {i: (False, 0.1, []) for i in range(3)}
        summary = run_shadow_for_day(DAY, max_chunks=2)
        self.assertEqual(sorted(r.state for r in self._rows().values()), ["done", "done", "pending"])
        self.assertEqual(summary["skipped"]["cap_chunks"], 1)
        run_shadow_for_day(DAY)
        self.assertEqual({r.state for r in self._rows().values()}, {"done"})

    def test_rows_from_later_days_wait(self):
        c = self._chunk(0)
        self._record([c], [_verdict(0, 0.1)])
        run_shadow_for_day(DAY - timedelta(days=1))
        self.assertEqual(self._rows()[0].state, "pending")

    def test_errors_abort_after_three(self):
        chunks = [self._chunk(i, "x" * 30_000) for i in range(5)]  # one chunk per batch
        self._record(chunks, [_verdict(i, 0.1) for i in range(5)])
        self.legacy_llm.run_structured.side_effect = RuntimeError("down")
        summary = run_shadow_for_day(DAY)
        self.assertEqual(summary["classifier_errors"], 3)
        self.assertEqual(summary["skipped"]["aborted"], 1)
        rows = self._rows()
        self.assertEqual({r.state for r in rows.values()}, {"pending"})
        self.assertEqual([rows[i].attempts for i in range(5)], [1, 1, 1, 0, 0])

    def test_rows_that_always_fail_are_given_up_and_stop_blocking(self):
        """Three chunks whose batches always fail come first every night; without an
        attempt bound the abort would keep the chunk behind them pending forever."""
        chunks = [self._chunk(i, "x" * 30_000) for i in range(4)]  # one chunk per batch
        self._record(chunks, [_verdict(i, 0.1) for i in range(4)])
        self.legacy = {3: (False, 0.1, [])}

        def answer(request, schema):
            if '"chunk_index": 3' not in request.messages[1].content:
                raise RuntimeError("refused")
            return self._legacy_answer(request, schema)

        self.legacy_llm.run_structured.side_effect = answer
        for _night in range(2):
            run_shadow_for_day(DAY)
            self.assertEqual({r.state for r in self._rows().values()}, {"pending"})
        summary = run_shadow_for_day(DAY)
        rows = self._rows()
        self.assertEqual(
            [(rows[i].state, rows[i].shadow_note, rows[i].attempts) for i in range(3)],
            [("skipped", "classifier_error", 3)] * 3,
        )
        self.assertEqual(summary["skipped"]["classifier_error"], 3)
        run_shadow_for_day(DAY)
        self.assertEqual(self._rows()[3].state, "done")

    def test_a_chunk_pending_on_two_days_is_classified_once_and_both_rows_finish(self):
        c = self._chunk(0)
        self._record([c], [_verdict(0, 0.1)])
        with patch("django.utils.timezone.localdate", return_value=DAY + timedelta(days=1)):
            record_chunk_verdicts(
                version=self.version, chunks=[{"id": c.id, "chunk_index": 0, "text": c.text}],
                results={0: _verdict(0, 0.9)}, org_id=self.org.id,
            )
        self.legacy = {0: (True, 0.8, ["jailbreak"])}
        summary = run_shadow_for_day(DAY + timedelta(days=1))
        rows = list(ChunkScanShadowResult.objects.order_by("run_date"))
        self.assertEqual([(r.state, r.classifier_flagged) for r in rows], [("done", True), ("done", True)])
        self.assertEqual(self.legacy_llm.run_structured.call_count, 1)
        self.assertEqual(self.legacy_llm.run_structured.call_args.args[0].messages[1].content.count('"chunk_index"'), 1)
        self.assertEqual((summary["chunks"], summary["classifier_only"], summary["both_positive"]), (1, 1, 1))
        self.assertEqual(summary["reviews"], 1)

    def test_no_side_effects_on_chunks_events_or_versions(self):
        c = self._chunk(0)
        self._record([c], [_verdict(0, 0.1)])
        self.legacy = {0: (True, 0.95, ["jailbreak"])}
        before = (GuardrailEvent.objects.count(), self.version.status)
        run_shadow_for_day(DAY)
        c.refresh_from_db()
        self.version.refresh_from_db()
        self.assertEqual((GuardrailEvent.objects.count(), self.version.status), before)
        self.assertFalse(c.is_quarantined)

    def test_db_unavailable_is_logged_not_raised(self):
        with patch("guardrails.models.ChunkScanShadowRun.objects.create", side_effect=OperationalError("db down")):
            with self.assertLogs("guardrails.shadow", level="INFO"):
                summary = run_shadow_for_day(DAY)
        self.assertEqual(summary["skipped"], {"db_unavailable": 1})


@override_settings(GUARDRAIL_SHADOW_ENABLED=True, DECISIONS_PRIMARY_ENABLED=True)
class ScanHookTests(_ShadowBase):
    """The chunk scan writes the pending rows itself."""

    def test_scan_records_pending_rows(self):
        from guardrails.tasks import scan_document_version
        from guardrails.tests._decisions import concern, predicate, result

        version = make_version(self.doc, version_index=1, status=DataRoomDocument.Status.SCANNING, make_active=False)
        DataRoomDocumentChunk.objects.create(version=version, chunk_index=0, text="Plain text.", token_count=2)
        decisions = MagicMock()
        decisions.decide.return_value = result(predicate("c0_adversarial", 0.2), concern("c0_concern"))
        with patch("guardrails.tasks._get_decisions_service", return_value=decisions), \
             patch("documents.tasks.finalize_document_metadata.delay"):
            scan_document_version(version.id)
        row = ChunkScanShadowResult.objects.get(version_id=version.id)
        self.assertEqual((row.state, row.direction, row.quick_probability), ("pending", DIRECTION, 0.2))


class NightlyTaskTests(TestCase):
    @override_settings(GUARDRAIL_SHADOW_ENABLED=False)
    def test_disabled_setting_skips(self):
        from guardrails.tasks import shadow_chunk_scan_nightly

        with patch("guardrails.shadow.run_shadow_for_day") as run:
            self.assertEqual(shadow_chunk_scan_nightly.apply().get(), "disabled")
        run.assert_not_called()

    @override_settings(GUARDRAIL_SHADOW_ENABLED=True)
    def test_runs_the_previous_local_day(self):
        from django.utils import timezone

        from guardrails.tasks import shadow_chunk_scan_nightly

        with patch("guardrails.shadow.run_shadow_for_day", return_value={"chunks": 0}) as run:
            result = shadow_chunk_scan_nightly.apply().get()
        self.assertEqual(result, {"chunks": 0})
        run.assert_called_once_with(timezone.localdate() - timedelta(days=1))

    def test_beat_schedule_entry_present(self):
        entry = settings.CELERY_BEAT_SCHEDULE["shadow-chunk-scan-nightly"]
        self.assertEqual(entry["task"], "guardrails.tasks.shadow_chunk_scan_nightly")
        self.assertTrue(hasattr(entry["schedule"], "hour"))
