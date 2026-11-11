"""The PII scan on the Decisions service, with the legacy classifier as fallback."""

from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from documents.services import pii_scan
from documents.services.pii_scan import PII_CATEGORIES, _PII_SYSTEM_PROMPT, scan_pii_categories, scan_pii_window
from guardrails.tests._decisions import predicate, refusal, result
from llm.service.decisions_fallback import FALLBACK_KIND
from llm.service.errors import LLMTimeoutError


def _answers(**probabilities):
    return result(*(predicate(c, probabilities.get(c, 0.01)) for c in PII_CATEGORIES))


@override_settings(DECISIONS_PRIMARY_ENABLED=True)
class ScanPIIWindowTests(TestCase):
    def setUp(self):
        self.decisions = MagicMock()
        patcher = patch.object(pii_scan, "_get_decisions_service", return_value=self.decisions)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_categories_at_or_above_half(self):
        self.decisions.decide.return_value = _answers(
            pii_ordinary_identity=0.5, pii_ordinary_financial=0.49, pii_special_category=0.9,
        )
        with patch("llm.get_llm_service") as legacy:
            verdict = scan_pii_window("Alice Smith, diagnosed with asthma.", user_id=3, org_id=4)
        legacy.assert_not_called()
        self.assertEqual(verdict.categories, {"pii_ordinary_identity": True, "pii_special_category": True})
        self.assertEqual(verdict.source, "decisions")
        self.assertEqual(verdict.probabilities["pii_ordinary_financial"], 0.49)
        args, kwargs = self.decisions.decide.call_args
        self.assertEqual([q.name for q in args[0]], list(PII_CATEGORIES))
        self.assertEqual(args[1][0].content[0].text, _PII_SYSTEM_PROMPT)
        self.assertEqual(kwargs["feature_key"], "pii_scan")
        self.assertEqual(kwargs["prompt_cache_key"], "decisions:pii_scan")

    def test_wrapper_returns_the_categories(self):
        self.decisions.decide.return_value = _answers(pii_ordinary_contact=0.8)
        self.assertEqual(scan_pii_categories("Call 555-0100", org_id=None), {"pii_ordinary_contact": True})

    def test_empty_text_makes_no_call(self):
        self.assertEqual(scan_pii_window("   ").categories, {})
        self.decisions.decide.assert_not_called()

    def _legacy(self, **flags):
        from llm.types.structured import PIICategoryOutput

        service = MagicMock()
        service.run_structured.return_value = (
            PIICategoryOutput(**{c: flags.get(c, False) for c in PII_CATEGORIES}), MagicMock(),
        )
        return service

    def test_failure_falls_back_to_the_tagged_legacy_classifier(self):
        self.decisions.decide.side_effect = LLMTimeoutError("down")
        legacy = self._legacy(pii_criminal_offence=True)
        with patch("llm.get_llm_service", return_value=legacy), \
             patch("core.preferences.resolve_org_feature_model", return_value="test/pii-model"):
            verdict = scan_pii_window("text", user_id=1, org_id=2)
        self.assertEqual(verdict.categories, {"pii_criminal_offence": True})
        self.assertEqual((verdict.source, verdict.decide_error), ("legacy", "LLMTimeoutError"))
        self.assertIsNone(verdict.probabilities)
        request = legacy.run_structured.call_args.args[0]
        self.assertEqual(request.params["_log_metadata"]["kind"], FALLBACK_KIND)
        self.assertEqual(request.params["_log_metadata"]["feature"], "pii_scan")
        self.assertEqual(request.messages[0].content, _PII_SYSTEM_PROMPT)

    def test_any_refused_category_falls_back(self):
        answers = [predicate(c, 0.1) for c in PII_CATEGORIES[:-1]] + [refusal(PII_CATEGORIES[-1])]
        self.decisions.decide.return_value = result(*answers)
        legacy = self._legacy()
        with patch("llm.get_llm_service", return_value=legacy), \
             patch("core.preferences.resolve_org_feature_model", return_value="test/pii-model"):
            verdict = scan_pii_window("text")
        self.assertEqual((verdict.source, verdict.decide_error), ("legacy", "DecisionsRefused"))


@override_settings(DECISIONS_PRIMARY_ENABLED=True)
class ScanWindowReviewTests(TestCase):
    """Gated Decisions positives still go to the unchanged Art. 9/10 reviewer."""

    def test_gated_positive_is_reviewed_and_ordinary_is_tagged(self):
        from documents.services.pii_scan import _scan_window, _ScanState

        decisions = MagicMock()
        decisions.decide.return_value = _answers(pii_ordinary_identity=0.9, pii_special_category=0.6)
        decision = MagicMock(article_9=False, article_10=False, confidence=0.9, reasoning="r", findings="")
        state = _ScanState()
        with patch.object(pii_scan, "_get_decisions_service", return_value=decisions), \
             patch("documents.services.pii_review.review_flagged_pii_window", return_value=decision) as review:
            _scan_window(
                [{"text": "Alice has asthma", "heading": "", "token_count": 4, "chunk_index": 0}], state,
                document=None, document_title="doc", version_id=None, user_id=None, data_room_id=None, org_id=None,
            )
        self.assertEqual(review.call_args.args[1], ["pii_special_category"])
        self.assertEqual(state.detected, {"pii_ordinary_identity": True})
