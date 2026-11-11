"""decide_with_fallback: Decisions first, the legacy callable only when Decisions fails."""
from unittest.mock import MagicMock, patch

from django.core.cache import cache
from django.test import SimpleTestCase

from llm.service import decisions_fallback
from llm.service.decisions_fallback import FALLBACK_KIND, DecisionsRefused, decide_with_fallback
from llm.service.errors import (
    LLMConfigurationError,
    LLMPolicyDenied,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestTooLargeError,
    LLMTimeoutError,
)
from llm.types.decisions import DecisionsResult, PredicateAnswer, Question
from llm.types.responses import Usage

QUESTIONS = [Question(type="predicate", name="flagged", instructions="Is it bad?")]


def _result(p=0.8):
    return DecisionsResult(answers=[PredicateAnswer(name="flagged", probability=p)], usage=Usage(), calls=[])


class DecideWithFallbackTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.service = MagicMock()
        patcher = patch.object(decisions_fallback, "_get_decisions_service", return_value=self.service)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, **kwargs):
        fallback = kwargs.pop("fallback", MagicMock(return_value="legacy"))
        interpret = kwargs.pop("interpret", lambda r: r["flagged"].probability)
        value = decide_with_fallback(
            QUESTIONS, "text", feature_key="pii_scan", org_id=7,
            interpret=interpret, fallback=fallback, **kwargs,
        )
        return value, fallback

    def test_success_uses_interpret_and_never_the_fallback(self):
        self.service.decide.return_value = _result(0.8)
        value, fallback = self._run(prompt_cache_key="decisions:pii_scan:x")
        self.assertEqual(value, 0.8)
        fallback.assert_not_called()
        kwargs = self.service.decide.call_args.kwargs
        self.assertEqual(kwargs["feature_key"], "pii_scan")
        self.assertEqual(kwargs["org_id"], 7)
        self.assertEqual(kwargs["prompt_cache_key"], "decisions:pii_scan:x")
        self.assertIsNotNone(kwargs["context"])

    def test_every_llm_error_takes_the_fallback_with_tagged_metadata(self):
        for exc in (
            LLMConfigurationError("no model"), LLMPolicyDenied("denied"),
            LLMProviderError("x", error_code="invalid_request"), LLMTimeoutError("t"),
            LLMRateLimitError("r"), LLMRequestTooLargeError("big"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.service.decide.side_effect = exc
                value, fallback = self._run()
                self.assertEqual(value, "legacy")
                meta = fallback.call_args.args[0]
                self.assertEqual(meta["kind"], FALLBACK_KIND)
                self.assertEqual(meta["feature"], "pii_scan")
                self.assertEqual(meta["decide_error"], type(exc).__name__)

    def test_error_code_is_carried(self):
        self.service.decide.side_effect = LLMProviderError("x", error_code="invalid_request")
        _value, fallback = self._run()
        self.assertEqual(fallback.call_args.args[0]["error_code"], "invalid_request")

    def test_refusal_from_interpret_takes_the_fallback(self):
        self.service.decide.return_value = _result()

        def interpret(_result):
            raise DecisionsRefused("refused")

        value, fallback = self._run(interpret=interpret)
        self.assertEqual(value, "legacy")
        self.assertEqual(fallback.call_args.args[0]["decide_error"], "DecisionsRefused")

    def test_unexpected_exception_takes_the_fallback(self):
        self.service.decide.side_effect = RuntimeError("structured pathway blew up")
        value, fallback = self._run()
        self.assertEqual(value, "legacy")
        self.assertEqual(fallback.call_args.args[0]["decide_error"], "RuntimeError")

    def test_bad_questions_propagate_without_calling_anything(self):
        fallback = MagicMock()
        with self.assertRaises(ValueError):
            decide_with_fallback(
                [{"type": "predicate", "name": "", "instructions": ""}], "text",
                feature_key="pii_scan", org_id=None, interpret=lambda r: r, fallback=fallback,
            )
        self.service.decide.assert_not_called()
        fallback.assert_not_called()

    def test_fallback_errors_propagate(self):
        self.service.decide.side_effect = LLMTimeoutError("t")
        with self.assertRaises(KeyError):
            self._run(fallback=MagicMock(side_effect=KeyError("legacy failed too")))

    def test_one_warning_per_feature_window_then_info(self):
        self.service.decide.side_effect = LLMTimeoutError("t")
        with self.assertLogs("llm.service.decisions_fallback", level="INFO") as logs:
            self._run()
            self._run()
        levels = [r.levelname for r in logs.records]
        self.assertEqual(levels, ["WARNING", "INFO"])
