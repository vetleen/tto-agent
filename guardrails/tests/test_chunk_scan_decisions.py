"""The chunk scan on the Decisions service: thresholds, reviewer, fallback, model gate."""

from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from accounts.models import User
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk
from guardrails.models import GuardrailEvent
from guardrails.tests._decisions import concern, predicate, refusal, result
from llm.service.decisions_fallback import FALLBACK_KIND
from llm.service.errors import LLMTimeoutError


def _models(chunk="test/model", reviewer=""):
    def resolve(org_id, feature):
        return {"guardrail_chunk_scan": chunk, "guardrails_reviewer": reviewer}.get(feature, "test/model")
    return resolve


@override_settings(DECISIONS_PRIMARY_ENABLED=True)
class ChunkScanDecisionsTest(TestCase):
    def setUp(self):
        from documents.tests._helpers import make_version

        self.user = User.objects.create_user(email="scan@example.com", password="test1234")
        room = DataRoom.objects.create(name="Room", slug="room", created_by=self.user)
        self.document = DataRoomDocument.objects.create(
            data_room=room, uploaded_by=self.user, original_filename="a.txt", status="ready",
        )
        self.version = make_version(self.document, status=DataRoomDocument.Status.SCANNING, make_active=False)
        for patcher in (
            patch("documents.tasks.finalize_document_metadata.delay"),
            patch("guardrails.tasks._get_decisions_service", return_value=MagicMock()),
        ):
            mock = patcher.start()
            self.addCleanup(patcher.stop)
        self.decisions = mock.return_value
        self.chunk = DataRoomDocumentChunk.objects.create(
            version=self.version, chunk_index=0, text="Some ordinary patent text.", token_count=4,
        )

    def _answer(self, p, choice="prompt_injection"):
        self.decisions.decide.return_value = result(predicate("c0_adversarial", p), concern("c0_concern", choice))

    def _scan(self, resolve=None):
        from guardrails.tasks import scan_document_version

        with patch("core.preferences.resolve_org_feature_model", side_effect=resolve or _models()):
            scan_document_version(self.version.id)
        self.chunk.refresh_from_db()

    def test_probability_below_half_is_clean(self):
        self._answer(0.49)
        with patch("llm.get_llm_service") as legacy:
            self._scan()
        legacy.assert_not_called()
        self.assertFalse(self.chunk.is_quarantined)
        self.assertFalse(GuardrailEvent.objects.exists())
        kwargs = self.decisions.decide.call_args.kwargs
        self.assertEqual(kwargs["feature_key"], "guardrail_chunk_scan")
        self.assertEqual(kwargs["prompt_cache_key"], "decisions:guardrail_chunk_scan")

    def test_without_reviewer_0_70_quarantines(self):
        self._answer(0.70)
        self._scan()
        self.assertTrue(self.chunk.is_quarantined)
        self.assertIn("prompt_injection", self.chunk.quarantine_reason)
        event = GuardrailEvent.objects.get(check_type="classifier")
        self.assertEqual(event.action_taken, "blocked")
        self.assertAlmostEqual(event.confidence, 0.70)

    def test_without_reviewer_0_69_is_logged_only(self):
        self._answer(0.69)
        self._scan()
        self.assertFalse(self.chunk.is_quarantined)
        self.assertEqual(GuardrailEvent.objects.get(check_type="classifier").action_taken, "logged")

    def test_reviewer_decides_a_flag(self):
        self._answer(0.95, "none")
        decision = MagicMock(action="allow", confidence=0.9, severity="low", reasoning="benign")
        with patch("guardrails.reviewer.review_flagged_chunk", return_value=decision) as review:
            self._scan(_models(reviewer="test/reviewer"))
        self.assertFalse(self.chunk.is_quarantined)
        sent = review.call_args.kwargs["classifier_result"]
        self.assertEqual(sent.concern_tags, ["prompt_injection"])  # argmax of the concern choice
        self.assertIn("probability only", sent.reasoning)
        self.assertEqual(GuardrailEvent.objects.get(check_type="llm_review").action_taken, "dismissed")

    def test_decisions_failure_uses_the_tagged_legacy_classifier(self):
        from guardrails.schemas import BatchClassifierResult, ChunkClassification

        self.decisions.decide.side_effect = LLMTimeoutError("down")
        legacy = MagicMock()
        legacy.run_structured.return_value = (BatchClassifierResult(results=[ChunkClassification(
            chunk_index=0, is_suspicious=True, concern_tags=["jailbreak"], confidence=0.8, reasoning="legacy",
        )]), None)
        with patch("llm.get_llm_service", return_value=legacy):
            self._scan()
        self.assertTrue(self.chunk.is_quarantined)
        request = legacy.run_structured.call_args.args[0]
        self.assertEqual(request.params["_log_metadata"]["kind"], FALLBACK_KIND)
        self.assertEqual(request.params["_log_metadata"]["feature"], "guardrail_chunk_scan")

    @override_settings(GUARDRAIL_SHADOW_ENABLED=True)
    def test_a_chunk_the_fallback_omits_is_retried_tagged_and_recorded_as_legacy(self):
        from guardrails.models import ChunkScanShadowResult
        from guardrails.schemas import BatchClassifierResult, ChunkClassification

        DataRoomDocumentChunk.objects.create(
            version=self.version, chunk_index=1, text="More ordinary patent text.", token_count=4,
        )
        self.decisions.decide.side_effect = LLMTimeoutError("down")

        def clean(index):
            return ChunkClassification(
                chunk_index=index, is_suspicious=False, concern_tags=[], confidence=0.1, reasoning="ok",
            )

        legacy = MagicMock()
        legacy.run_structured.side_effect = [
            (BatchClassifierResult(results=[clean(0)]), None),  # the batch call omits chunk 1
            (BatchClassifierResult(results=[clean(1)]), None),  # its individual retry
        ]
        with patch("llm.get_llm_service", return_value=legacy):
            self._scan()
        batch, retry = (call.args[0] for call in legacy.run_structured.call_args_list)
        self.assertEqual(retry.params["_log_metadata"]["kind"], FALLBACK_KIND)
        self.assertEqual(retry.context.run_id, batch.context.run_id)
        notes = dict(ChunkScanShadowResult.objects.values_list("chunk_index", "shadow_note"))
        self.assertEqual(notes, {0: "prod_legacy:LLMTimeoutError", 1: "prod_legacy:LLMTimeoutError"})

    def test_refusal_uses_the_legacy_classifier(self):
        from guardrails.schemas import BatchClassifierResult, ChunkClassification

        self.decisions.decide.return_value = result(refusal("c0_adversarial"), concern("c0_concern"))
        legacy = MagicMock()
        legacy.run_structured.return_value = (BatchClassifierResult(results=[ChunkClassification(
            chunk_index=0, is_suspicious=False, concern_tags=[], confidence=0.1, reasoning="ok",
        )]), None)
        with patch("llm.get_llm_service", return_value=legacy):
            self._scan()
        legacy.run_structured.assert_called_once()
        self.assertFalse(self.chunk.is_quarantined)

    def test_no_chunk_model_still_scans_through_decisions(self):
        self._answer(0.9)
        with patch("core.preferences.resolve_decisions_settings") as prefs:
            prefs.return_value = MagicMock(quick_model="openai/gpt-6-luna", structured_model="")
            self._scan(_models(chunk=""))
        self.decisions.decide.assert_called_once()
        self.assertTrue(self.chunk.is_quarantined)

    def test_no_model_anywhere_releases_heuristics_only(self):
        with patch("core.preferences.resolve_decisions_settings") as prefs:
            prefs.return_value = MagicMock(quick_model="", structured_model="")
            self._scan(_models(chunk=""))
        self.decisions.decide.assert_not_called()
        self.assertEqual(self.chunk.guardrail_scan_state, DataRoomDocumentChunk.GuardrailScanState.HEURISTIC_DONE)

    def test_decisions_and_legacy_both_failing_fails_the_scan(self):
        self.decisions.decide.side_effect = LLMTimeoutError("down")
        with patch("core.preferences.resolve_decisions_settings") as prefs:
            prefs.return_value = MagicMock(quick_model="openai/gpt-6-luna", structured_model="")
            with self.assertRaises(RuntimeError):  # the task's retry -> SCAN_FAILED machinery
                self._scan(_models(chunk=""))
        self.chunk.refresh_from_db()
        self.assertNotEqual(self.chunk.guardrail_scan_state, DataRoomDocumentChunk.GuardrailScanState.DONE)
