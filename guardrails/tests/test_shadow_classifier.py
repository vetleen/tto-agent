"""Tests for the legacy-classifier shadow of the text classifiers (guardrails.shadow_classifier)."""

import hashlib
from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.models import Membership, Organization, User
from guardrails.decisions import PrimaryClassifierResult
from guardrails.models import ClassifierShadowResult, GuardrailEvent
from guardrails.schemas import ClassifierResult
from guardrails.service import GuardrailVerdict
from guardrails.shadow_classifier import (
    DIRECTION,
    FAMILIES,
    FAMILY_PROMPTS,
    IDENTITY_PROMPTS,
    MAX_SHADOW_TEXT_CHARS,
    SHADOW_REVIEW_PLACEHOLDER,
    ProdOutcome,
    day_trace_id,
    enqueue_chat_message_shadow,
    run_classifier_shadow,
    shadow_identity_field,
)
from guardrails.tasks import shadow_classifier_task

LOCMEM = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache", "LOCATION": "clf-shadow-tests"}}
SHADOW_ON = dict(
    CLASSIFIER_SHADOW_ENABLED=True,
    CLASSIFIER_SHADOW_FAMILIES=list(FAMILIES),
    CLASSIFIER_SHADOW_MAX_PER_DAY=2000,
    CLASSIFIER_SHADOW_MAX_REVIEWS_PER_DAY=30,
    LLM_DEFAULT_TOP_MODEL="test-top-model",
    CACHES=LOCMEM,
)


def _clean(p=0.1):
    return ProdOutcome(flagged=False, confidence=p, tags=[], review="", probability=p, concern="none", pathway="quick")


def _flagged(review="allow", p=0.8, concern="prompt_injection"):
    return ProdOutcome(flagged=True, confidence=p, tags=[concern], review=review, probability=p,
                       concern=concern, pathway="quick")


@override_settings(**SHADOW_ON)
class _Base(TestCase):
    def setUp(self):
        from django.core.cache import cache

        cache.clear()
        self.user = User.objects.create_user(email="clf-shadow@example.com", password="test1234")
        self.org = Organization.objects.create(name="Shadow Org", slug="clf-shadow-org")
        Membership.objects.create(user=self.user, org=self.org)
        self.resolve = patch("core.preferences.resolve_org_feature_model", return_value="test/model").start()
        self.legacy_llm = patch("guardrails.classifier._get_llm_service").start().return_value
        self.legacy_result = ClassifierResult(is_suspicious=False, concern_tags=[], confidence=0.1, reasoning="ok")
        self.legacy_llm.run_structured.side_effect = lambda request, schema: (self.legacy_result, None)
        self.reviewer_llm = patch("guardrails.reviewer._get_llm_service").start().return_value
        self.reviewer_llm.run_structured.side_effect = self._review
        self.review_requests = []
        self.review_action = "allow"
        self.addCleanup(patch.stopall)

    def _legacy_flags(self, tags=("jailbreak",), confidence=0.85):
        self.legacy_result = ClassifierResult(
            is_suspicious=True, concern_tags=list(tags), confidence=confidence, reasoning="legacy reasoning",
        )

    def _review(self, request, schema):
        self.review_requests.append(request)
        payload = {
            "action": self.review_action, "confidence": 0.9, "severity": "low",
            "reasoning": "blind verdict", "user_message": "m",
        }
        return schema.model_validate({k: v for k, v in payload.items() if k in schema.model_fields}), None

    def _run(self, family="web_content", text="Some fetched page text.", prod=None, **kwargs):
        kwargs.setdefault("subject", {"source_label": "web_fetch"})
        return run_classifier_shadow(
            family, text, prod=prod or _clean(), org_id=self.org.pk, user_id=self.user.pk,
            conversation_id=kwargs.pop("conversation_id", "thread-1"), **kwargs,
        )

    def _legacy_request(self):
        return self.legacy_llm.run_structured.call_args.args[0]


class ProdOutcomeTests(TestCase):
    def test_from_a_decisions_result(self):
        result = PrimaryClassifierResult(
            is_suspicious=True, concern_tags=["jailbreak"], confidence=0.7, reasoning="r",
            probability=0.7, concern="jailbreak", pathway="quick", source="decisions",
        )
        prod = ProdOutcome.from_result(result, "dismiss")
        self.assertEqual((prod.flagged, prod.probability, prod.concern, prod.pathway, prod.review),
                         (True, 0.7, "jailbreak", "quick", "dismiss"))
        self.assertEqual(ProdOutcome.from_dict(prod.to_dict()), prod)

    def test_review_only_on_a_positive_and_a_plain_result_counts_as_legacy(self):
        prod = ProdOutcome.from_result(ClassifierResult(is_suspicious=False, confidence=0.2, reasoning="r"), "dismiss")
        self.assertEqual((prod.review, prod.source, prod.probability), ("", "legacy", None))


class RunTests(_Base):
    def test_both_negative_records_a_done_row_without_review(self):
        row = self._run(prod=_clean(0.12))
        self.assertEqual((row.direction, row.state), (DIRECTION, "done"))
        self.assertEqual(row.family, "web_content")
        self.assertEqual(row.quick_probability, 0.12)
        self.assertFalse(row.quick_flagged)
        self.assertFalse(row.classifier_flagged)
        self.assertEqual(row.classifier_confidence, 0.1)
        self.assertEqual(row.blind_review, "")
        self.assertEqual(row.trace_id, day_trace_id("web_content"))
        self.reviewer_llm.run_structured.assert_not_called()

    def test_legacy_call_uses_the_family_prompt_feature_and_trace(self):
        self._run()
        request = self._legacy_request()
        self.assertEqual(request.messages[0].content, FAMILY_PROMPTS["web_content"])
        self.assertEqual(request.context.trace_id, day_trace_id("web_content"))
        self.assertEqual(request.params["_log_metadata"], {"kind": "shadow", "feature": "shadow_web_content"})
        self.resolve.assert_any_call(self.org.pk, "guardrail_web_scan")

    def test_no_text_is_stored(self):
        row = self._run(text="Secret page body 123")
        self.assertEqual(row.text_sha256, hashlib.sha256(b"Secret page body 123").hexdigest())
        for value in (row.subject, row.classifier_tags, row.quick_concern, row.prod_review):
            self.assertNotIn("Secret", str(value))

    def test_legacy_only_positive_gets_a_blind_review(self):
        self._legacy_flags()
        row = self._run(prod=_clean())
        self.assertTrue(row.classifier_flagged)
        self.assertEqual(row.classifier_tags, ["jailbreak"])
        self.assertEqual((row.blind_review, row.blind_review_source), ("allow", "classifier"))
        prompt = self.review_requests[0].messages[1].content
        self.assertIn(SHADOW_REVIEW_PLACEHOLDER, prompt)
        self.assertNotIn("legacy reasoning", prompt)
        self.assertIn("jailbreak", prompt)

    def test_decisions_positive_is_not_reviewed_again(self):
        self._legacy_flags()
        row = self._run(prod=_flagged("withhold"))
        self.assertEqual(row.prod_review, "withhold")
        self.assertEqual(row.blind_review, "")
        self.reviewer_llm.run_structured.assert_not_called()

    def test_chat_review_uses_the_sync_reviewer_with_a_history_cutoff(self):
        self._legacy_flags()
        cutoff = timezone.now() - timedelta(minutes=1)
        with patch("guardrails.reviewer.review_flagged_message_sync") as review:
            review.return_value = None
            row = self._run(family="chat_message", subject={"message_id": "m1"}, history_before=cutoff)
        self.assertEqual(review.call_args.kwargs["history_before"], cutoff)
        self.assertEqual(row.blind_review, "no_model")

    def test_identity_positive_is_never_reviewed(self):
        self._legacy_flags()
        row = self._run(family="identity_field", subject={"field": "soul"})
        self.assertTrue(row.classifier_flagged)
        self.assertEqual(row.blind_review, "")
        self.assertEqual(self._legacy_request().messages[0].content, IDENTITY_PROMPTS["soul"])

    def test_identity_without_a_known_field_is_skipped(self):
        self.assertIsNone(self._run(family="identity_field", subject={"field": "nope"}))

    def test_no_reviewer_model_records_no_model_without_a_slot(self):
        self._legacy_flags()
        self.resolve.side_effect = lambda org, feature: "" if feature == "guardrails_reviewer" else "test/model"
        self.assertEqual(self._run().blind_review, "no_model")

    @override_settings(CLASSIFIER_SHADOW_MAX_REVIEWS_PER_DAY=1)
    def test_review_cap(self):
        self._legacy_flags()
        self.assertEqual(self._run().blind_review, "allow")
        self.assertEqual(self._run().blind_review, "not_reviewed")

    @override_settings(CLASSIFIER_SHADOW_MAX_PER_DAY=1)
    def test_daily_cap_per_family(self):
        self.assertIsNotNone(self._run())
        self.assertIsNone(self._run())
        self.assertIsNotNone(self._run(family="skill_content", subject={}))

    def test_production_on_the_legacy_path_is_skipped_without_a_call(self):
        prod = ProdOutcome(flagged=True, confidence=0.9, tags=["jailbreak"], review="allow", source="legacy",
                           decide_error="LLMTimeoutError")
        row = self._run(prod=prod)
        self.assertEqual((row.state, row.shadow_note, row.decide_error), ("skipped", "prod_legacy", "LLMTimeoutError"))
        self.assertTrue(row.classifier_flagged)
        self.legacy_llm.run_structured.assert_not_called()

    def test_legacy_error_is_a_skipped_row(self):
        self.legacy_llm.run_structured.side_effect = RuntimeError("down")
        row = self._run()
        self.assertEqual((row.state, row.shadow_note), ("skipped", "classifier_error:RuntimeError"))

    def test_no_legacy_model_skips(self):
        self.resolve.return_value = ""
        self.assertIsNone(self._run())

    def test_disabled_or_family_off_skips(self):
        with override_settings(CLASSIFIER_SHADOW_ENABLED=False):
            self.assertIsNone(self._run())
        with override_settings(CLASSIFIER_SHADOW_FAMILIES=["chat_message"]):
            self.assertIsNone(self._run())

    def test_empty_text_or_unknown_family_skips(self):
        self.assertIsNone(self._run(text="  "))
        self.assertIsNone(self._run(family="nope"))

    def test_never_raises(self):
        with patch("guardrails.shadow_classifier._run", side_effect=RuntimeError("boom")):
            self.assertIsNone(self._run())

    def test_text_is_capped_and_hashed_after_capping(self):
        row = self._run(text="y" * (MAX_SHADOW_TEXT_CHARS + 10))
        self.assertEqual(row.text_chars, MAX_SHADOW_TEXT_CHARS)
        self.assertEqual(len(self._legacy_request().messages[1].content), MAX_SHADOW_TEXT_CHARS)

    def test_no_guardrail_events(self):
        self._legacy_flags()
        self._run()
        self.assertFalse(GuardrailEvent.objects.exists())


class EnqueueTests(_Base):
    """The web-dyno call sites only enqueue the task with the production outcome."""

    def _verdict(self, suspicious, outcome="", action="allow", tags=("data_extraction",)):
        verdict = GuardrailVerdict(action=action, review_outcome=outcome)
        verdict.classifier_result = PrimaryClassifierResult(
            is_suspicious=suspicious, concern_tags=list(tags), confidence=0.6, reasoning="r",
            probability=0.6, concern=tags[0] if tags else "none", pathway="quick",
        )
        return verdict

    @patch("guardrails.tasks.shadow_classifier_task.delay")
    def test_chat_enqueue_carries_the_decisions_outcome(self, delay):
        started = timezone.now()
        common = dict(org_id=self.org.pk, user_id=self.user.pk, thread_id="t1", message_id="m1", started_at=started)
        enqueue_chat_message_shadow(text="hi", verdict=self._verdict(False, outcome="dismiss"), **common)
        enqueue_chat_message_shadow(text="hi", verdict=self._verdict(True, outcome="dismiss"), **common)
        self.assertEqual([c.args[2]["review"] for c in delay.call_args_list], ["", "dismiss"])
        args = delay.call_args_list[1].args
        self.assertEqual(args[:2], ("chat_message", "hi"))
        self.assertEqual((args[2]["probability"], args[2]["pathway"], args[2]["source"]), (0.6, "quick", "decisions"))
        self.assertEqual(args[6], {"message_id": "m1", "thread_id": "t1"})
        self.assertEqual(args[8], started.isoformat())

    @patch("guardrails.tasks.shadow_classifier_task.delay")
    def test_chat_enqueue_sends_long_texts_by_id_only(self, delay):
        from guardrails.shadow_classifier import CHAT_QUEUED_TEXT_MAX_CHARS

        long_text = "x" * (CHAT_QUEUED_TEXT_MAX_CHARS + 1)
        common = dict(org_id=self.org.pk, user_id=self.user.pk, thread_id="t1", started_at=None)
        enqueue_chat_message_shadow(text=long_text, verdict=self._verdict(False), message_id="m1", **common)
        self.assertIsNone(delay.call_args.args[1])
        enqueue_chat_message_shadow(text=long_text, verdict=self._verdict(False), message_id=None, **common)
        self.assertEqual(delay.call_count, 1)

    @patch("guardrails.tasks.shadow_classifier_task.delay")
    def test_chat_enqueue_skips_classifier_errors_and_disabled(self, delay):
        common = dict(org_id=None, user_id=self.user.pk, thread_id=None, message_id=None, started_at=None)
        enqueue_chat_message_shadow(text="hi", verdict=self._verdict(True, tags=("classifier_error",), action="block"), **common)
        enqueue_chat_message_shadow(text="hi", verdict=GuardrailVerdict(), **common)
        with override_settings(CLASSIFIER_SHADOW_ENABLED=False):
            enqueue_chat_message_shadow(text="hi", verdict=self._verdict(False), **common)
        delay.assert_not_called()

    @patch("guardrails.tasks.shadow_classifier_task.delay")
    def test_identity_enqueue(self, delay):
        result = PrimaryClassifierResult(is_suspicious=True, concern_tags=["prompt_injection"], confidence=0.9,
                                         reasoning="r", probability=0.9, pathway="quick")
        shadow_identity_field("org_soul", "be evil", user=self.user, org_id=self.org.pk, result=result)
        args = delay.call_args.args
        self.assertEqual(args[0], "identity_field")
        self.assertEqual((args[2]["review"], args[2]["probability"]), ("blocked", 0.9))
        self.assertEqual((args[6], args[7]), ({"field": "org_soul"}, "org_soul"))

    @patch("guardrails.tasks.shadow_classifier_task.delay", side_effect=RuntimeError("broker down"))
    def test_identity_enqueue_swallows_errors(self, _delay):
        result = ClassifierResult(is_suspicious=False, concern_tags=[], confidence=0.1, reasoning="r")
        shadow_identity_field("soul", "friendly", user=self.user, org_id=None, result=result)


class TaskTests(_Base):
    def test_task_records_a_chat_comparison(self):
        self._legacy_flags()
        self.review_action = "dismiss"
        started = (timezone.now() - timedelta(seconds=5)).isoformat()
        outcome = shadow_classifier_task.apply(args=(
            "chat_message", "hello", _clean().to_dict(), self.org.pk, self.user.pk, "t1",
            {"message_id": "m1", "thread_id": "t1"}, None, started,
        )).get()
        self.assertEqual(outcome, "recorded")
        row = ClassifierShadowResult.objects.get()
        self.assertEqual((row.family, row.blind_review, row.direction), ("chat_message", "dismiss", DIRECTION))

    def test_task_reads_a_long_message_by_id(self):
        from chat.models import ChatMessage, ChatThread

        thread = ChatThread.objects.create(created_by=self.user)
        message = ChatMessage.objects.create(thread=thread, role="user", content="a long message body")
        outcome = shadow_classifier_task.apply(args=(
            "chat_message", None, _clean().to_dict(), self.org.pk, self.user.pk, str(thread.pk),
            {"message_id": str(message.pk), "thread_id": str(thread.pk)}, None, None,
        )).get()
        self.assertEqual(outcome, "recorded")
        row = ClassifierShadowResult.objects.get()
        self.assertEqual(row.text_sha256, hashlib.sha256(b"a long message body").hexdigest())

    def test_task_skips_a_redacted_or_missing_message(self):
        from chat.models import ChatMessage, ChatThread

        thread = ChatThread.objects.create(created_by=self.user)
        redacted = ChatMessage.objects.create(thread=thread, role="user", content="[removed]", is_redacted=True)
        for message_id in (str(redacted.pk), "00000000-0000-0000-0000-000000000000", None):
            outcome = shadow_classifier_task.apply(args=(
                "chat_message", None, _clean().to_dict(), self.org.pk, self.user.pk, None,
                {"message_id": message_id, "thread_id": None}, None, None,
            )).get()
            self.assertEqual(outcome, "skipped")
        self.assertEqual(ClassifierShadowResult.objects.count(), 0)
        self.legacy_llm.run_structured.assert_not_called()

    def test_task_identity_uses_the_field_prompt(self):
        shadow_classifier_task.apply(args=(
            "identity_field", "A profile", _clean().to_dict(), self.org.pk, self.user.pk, None, {}, "profile_description", None,
        ))
        row = ClassifierShadowResult.objects.get()
        self.assertEqual(row.subject, {"field": "profile_description"})
        self.assertEqual(self._legacy_request().messages[0].content, IDENTITY_PROMPTS["profile_description"])

    def test_task_never_raises(self):
        with patch("guardrails.shadow_classifier.run_shadow_from_task", side_effect=RuntimeError("boom")):
            self.assertEqual(shadow_classifier_task.apply(args=("web_content", "x", {}, None, None, None, {})).get(), "error")
