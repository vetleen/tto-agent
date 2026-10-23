"""Tests for the quick decisions pathway (llm.service.decisions_quick)."""

import logging
import time
from unittest.mock import MagicMock, patch

import httpx
import openai
from django.core.cache import cache
from django.test import SimpleTestCase, override_settings

from llm.service.decisions_quick import (
    API_PATH,
    QuickAvailability,
    QuickUnavailable,
    api_model_id,
    build_body,
    call_quick,
    classify_quick_error,
    parse_response,
)
from llm.service.errors import LLMProviderError, LLMRequestTooLargeError
from llm.types.decisions import Choice, InputMessage, JsonPart, Level, Question, TextPart, normalize_input

_URL = "https://api.openai.com/v1/decisions"


def _pred(name="disclosed"):
    return Question(type="predicate", name=name, instructions="Publicly disclosed?")


def _choice(name="field"):
    return Question(
        type="choice", name=name, instructions="Which field?",
        choices=[Choice(value="chemistry", description="Catalysis etc."), Choice(value="software")],
    )


def _score(name="potential"):
    return Question(type="score", name=name, instructions="Commercial potential?",
                    levels=[Level(label="Low"), Level(label="Medium"), Level(label="High")])


# The shape observed live on 2026-10-23 (values 0-based, output free).
LIVE_RAW = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "disclosed", "probability": 1.0},
        {"type": "choice", "name": "field", "choice": "chemistry",
         "probabilities": [{"value": "chemistry", "probability": 1.0}, {"value": "software", "probability": 0.0}],
         "confidence": 1.0},
        {"type": "score", "name": "potential", "score": 1.42,
         "probabilities": [{"value": 0, "label": "Low", "probability": 0.01},
                           {"value": 1, "label": "Medium", "probability": 0.56},
                           {"value": 2, "label": "High", "probability": 0.43}],
         "confidence": 0.34},
    ],
    "usage": {"input_tokens": 463, "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 20},
              "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 463},
}


def _response(status):
    return httpx.Response(status, request=httpx.Request("POST", _URL))


class BuildBodyTests(SimpleTestCase):
    def test_wire_shape_from_string_input(self):
        body = build_body([_pred(), _choice()], normalize_input("A poster."), model="openai/gpt-6-luna")
        self.assertEqual(body, {
            "model": "gpt-6-luna",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "A poster."}]}],
            "questions": [
                {"type": "predicate", "name": "disclosed", "instructions": "Publicly disclosed?"},
                {"type": "choice", "name": "field", "instructions": "Which field?",
                 "choices": [{"value": "chemistry", "description": "Catalysis etc."}, {"value": "software"}]},
            ],
        })
        self.assertNotIn("prompt_cache_key", body)

    def test_messages_json_and_cache_key(self):
        input = [
            InputMessage(role="developer", content=[TextPart(text="Be strict.")]),
            InputMessage(content=[JsonPart(data={"venue": "poster"})]),
        ]
        body = build_body([_score()], input, model="openai/gpt-6-luna", prompt_cache_key="decisions:triage")
        self.assertEqual(body["input"], [
            {"role": "developer", "content": [{"type": "input_text", "text": "Be strict."}]},
            {"role": "user", "content": [{"type": "input_json", "data": {"venue": "poster"}}]},
        ])
        self.assertEqual(body["questions"][0]["levels"], [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}])
        self.assertEqual(body["prompt_cache_key"], "decisions:triage")

    def test_api_model_id(self):
        self.assertEqual(api_model_id("openai/gpt-6-luna"), "gpt-6-luna")
        self.assertEqual(api_model_id("openai/gpt-5.6-luna"), "gpt-6-luna")  # retired alias → replacement
        self.assertEqual(api_model_id("custom/x-model"), "x-model")


class ParseResponseTests(SimpleTestCase):
    def test_live_shape_parses_in_batch_order(self):
        # Batch order differs from the response order: answers follow the batch.
        answers, usage, model = parse_response([_score(), _pred(), _choice()], LIVE_RAW)
        self.assertEqual([a.type for a in answers], ["score", "predicate", "choice"])
        self.assertEqual(answers[0].score, 1.42)
        self.assertEqual([p.value for p in answers[0].probabilities], [0, 1, 2])
        self.assertEqual(answers[0].probabilities[1].label, "Medium")
        self.assertEqual(answers[1].probability, 1.0)
        self.assertEqual(answers[2].choice, "chemistry")
        self.assertEqual(answers[2].confidence, 1.0)
        self.assertEqual(model, "gpt-6-luna")
        self.assertEqual(usage.prompt_tokens, 463)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertEqual(usage.cached_tokens, 100)
        self.assertEqual(usage.cache_write_tokens, 20)
        self.assertEqual(usage.total_tokens, 463)

    def test_refusal_and_missing_usage(self):
        raw = {"model": "gpt-6-luna", "answers": [{"type": "refusal", "name": "disclosed"}]}
        answers, usage, _ = parse_response([_pred()], raw)
        self.assertEqual(answers[0].type, "refusal")
        self.assertIsNone(usage.prompt_tokens)
        self.assertEqual(usage.completion_tokens, 0)

    def test_malformed_bodies(self):
        cases = {
            "not a dict": "oops",
            "no answers": {"model": "m"},
            "answers not a list": {"answers": {"a": 1}},
            "unknown type": {"answers": [{"type": "verdict", "name": "disclosed", "value": True}]},
            "missing field": {"answers": [{"type": "predicate", "name": "disclosed"}]},
            "wrong name": {"answers": [{"type": "predicate", "name": "other", "probability": 0.5}]},
            "extra answer": {"answers": [
                {"type": "predicate", "name": "disclosed", "probability": 0.5},
                {"type": "predicate", "name": "disclosed", "probability": 0.6},
            ]},
        }
        for label, raw in cases.items():
            with self.subTest(case=label), self.assertRaises(QuickUnavailable) as cm:
                parse_response([_pred()], raw)
            self.assertEqual(cm.exception.error_code, "malformed_response")
            self.assertEqual(cm.exception.log_level, "error")


class ClassifyTests(SimpleTestCase):
    def test_transient_and_infrastructure_errors_fall_back(self):
        cases = [
            (openai.APITimeoutError(request=httpx.Request("POST", _URL)), "timeout", "warning"),
            (openai.RateLimitError("slow down", response=_response(429), body=None), "rate_limited", "warning"),
            (openai.APIStatusError("busy", response=_response(503), body=None), "overloaded", "warning"),
            (openai.InternalServerError("boom", response=_response(500), body=None), "server_error", "error"),
            (openai.APIConnectionError(request=httpx.Request("POST", _URL)), "connection_error", "warning"),
            (openai.AuthenticationError("bad key", response=_response(401), body=None), "auth_error", "error"),
            (openai.NotFoundError("no such route", response=_response(404), body=None), "unknown", "error"),
        ]
        for exc, code, level in cases:
            with self.subTest(code=code):
                result = classify_quick_error(exc)
                self.assertIsInstance(result, QuickUnavailable)
                self.assertEqual(result.error_code, code)
                self.assertEqual(result.log_level, level)

    def test_programmer_errors_raise_to_the_caller(self):
        too_large = openai.BadRequestError(
            "This model's maximum context length is 1050000 tokens; your messages resulted in more",
            response=_response(400), body=None,
        )
        self.assertIsInstance(classify_quick_error(too_large), LLMRequestTooLargeError)

        bad_question = openai.BadRequestError("Invalid value for questions[0].type", response=_response(400), body=None)
        result = classify_quick_error(bad_question)
        self.assertIsInstance(result, LLMProviderError)
        self.assertNotIsInstance(result, QuickUnavailable)
        self.assertEqual(result.error_code, "invalid_request")

        unprocessable = openai.UnprocessableEntityError("nope", response=_response(422), body=None)
        self.assertEqual(classify_quick_error(unprocessable).error_code, "invalid_request")


@override_settings(DECISIONS_QUICK_TIMEOUT_SECONDS=30, DECISIONS_QUICK_MAX_RETRIES=1)
class CallQuickTests(SimpleTestCase):
    def test_posts_with_pinned_options_and_parses(self):
        client = MagicMock()
        client.post.return_value = LIVE_RAW
        with patch("llm.service.decisions_quick.get_openai_client", return_value=client):
            answers, usage, model = call_quick(
                [_pred(), _choice(), _score()], normalize_input("x"),
                model="openai/gpt-6-luna", prompt_cache_key="decisions:t",
            )
        client.post.assert_called_once()
        args, kwargs = client.post.call_args
        self.assertEqual(args, (API_PATH,))
        self.assertIs(kwargs["cast_to"], object)
        self.assertEqual(kwargs["body"]["model"], "gpt-6-luna")
        self.assertEqual(kwargs["body"]["prompt_cache_key"], "decisions:t")
        self.assertEqual(len(kwargs["body"]["questions"]), 3)
        timeout = kwargs["options"]["timeout"]
        self.assertEqual(timeout.read, 30.0)
        self.assertEqual(timeout.connect, 10.0)
        self.assertEqual(kwargs["options"]["max_retries"], 1)
        self.assertEqual([a.type for a in answers], ["predicate", "choice", "score"])
        self.assertEqual(model, "gpt-6-luna")
        self.assertEqual(usage.prompt_tokens, 463)

    def test_sdk_errors_are_classified(self):
        client = MagicMock()
        client.post.side_effect = openai.APITimeoutError(request=httpx.Request("POST", _URL))
        with patch("llm.service.decisions_quick.get_openai_client", return_value=client):
            with self.assertRaises(QuickUnavailable) as cm:
                call_quick([_pred()], normalize_input("x"), model="openai/gpt-6-luna")
        self.assertEqual(cm.exception.error_code, "timeout")


@override_settings(DECISIONS_QUICK_SKIP_SECONDS=60)
class QuickAvailabilityTests(SimpleTestCase):
    KEY = "decisions_quick_skip:test"

    def setUp(self):
        cache.delete(self.KEY)
        self.addCleanup(cache.delete, self.KEY)

    def _availability(self):
        return QuickAvailability(key=self.KEY)

    def test_a_failure_opens_the_window_with_one_warning(self):
        a = self._availability()
        self.assertFalse(a.is_suspended())
        with self.assertLogs("llm.service.decisions_quick", level="INFO") as cm:
            self.assertTrue(a.suspend("timeout"))
            self.assertFalse(a.suspend("rate_limited"))  # same episode: extended, INFO only
        self.assertTrue(a.is_suspended())
        warnings = [r for r in cm.records if r.levelno >= logging.WARNING]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].levelno, logging.WARNING)
        self.assertIn("timeout", warnings[0].getMessage())

    def test_error_level_for_failures_that_will_not_fix_themselves(self):
        with self.assertLogs("llm.service.decisions_quick", level="ERROR"):
            self._availability().suspend("auth_error", "error")

    def test_held_across_processes_through_the_cache(self):
        a, b = self._availability(), self._availability()
        a.suspend("timeout")
        self.assertTrue(b.is_suspended())
        # The other process sees the same episode: no second event.
        with self.assertNoLogs("llm.service.decisions_quick", level="WARNING"):
            b.suspend("timeout")

    def test_window_expires(self):
        a = self._availability()
        a.suspend("timeout")
        cache.delete(self.KEY)  # the cache entry's TTL ran out
        with patch("llm.service.decisions_quick.time.monotonic", return_value=time.monotonic() + 61):
            self.assertFalse(a.is_suspended())

    def test_local_mirror_holds_when_the_cache_is_down(self):
        a = self._availability()
        with patch("django.core.cache.cache.add", side_effect=RuntimeError("redis down")), \
                patch("django.core.cache.cache.get", side_effect=RuntimeError("redis down")):
            with self.assertLogs("llm.service.decisions_quick", level="WARNING"):
                self.assertTrue(a.suspend("timeout"))
            self.assertTrue(a.is_suspended())
            self.assertFalse(self._availability().is_suspended())  # another process: cache gone, no mirror

    @override_settings(DECISIONS_QUICK_SKIP_SECONDS=0)
    def test_zero_disables_the_window(self):
        a = self._availability()
        self.assertFalse(a.suspend("timeout"))
        self.assertFalse(a.is_suspended())

    def test_reset(self):
        a = self._availability()
        a.suspend("timeout")
        a.reset()
        self.assertFalse(a.is_suspended())
