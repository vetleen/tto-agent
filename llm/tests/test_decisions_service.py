"""Tests for DecisionsService — pathway selection, fallback, batching, logging."""

import asyncio
import logging
import os
from unittest.mock import MagicMock, patch

import httpx
import openai
from django.core.cache import cache
from django.test import TestCase, override_settings

from core.preferences import DecisionsSettings
from llm.models import LLMCallLog
from llm.service.decisions_quick import QuickAvailability
from llm.service.decisions_service import DecisionsService, get_decisions_service
from llm.service.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRequestTooLargeError,
    LLMTimeoutError,
)
from llm.types.context import RunContext
from llm.types.decisions import Choice, Level, Question
from llm.types.responses import Usage

_URL = "https://api.openai.com/v1/decisions"
LUNA = "openai/gpt-6-luna"
NANO = "openai/gpt-5.4-nano"


def _pred(name="disclosed"):
    return Question(type="predicate", name=name, instructions="Publicly disclosed?")


def _choice(name="field"):
    return Question(type="choice", name=name, instructions="Which field?",
                    choices=[Choice(value="chemistry"), Choice(value="software")])


def _score(name="potential"):
    return Question(type="score", name=name, instructions="Potential?",
                    levels=[Level(label="Low"), Level(label="High")])


def _raw_from_body(body):
    """A well-formed endpoint response for whatever questions were posted."""
    answers = []
    for q in body["questions"]:
        if q["type"] == "predicate":
            answers.append({"type": "predicate", "name": q["name"], "probability": 0.9})
        elif q["type"] == "choice":
            answers.append({
                "type": "choice", "name": q["name"], "choice": q["choices"][0]["value"],
                "probabilities": [
                    {"value": c["value"], "probability": 1.0 if i == 0 else 0.0}
                    for i, c in enumerate(q["choices"])
                ],
                "confidence": 1.0,
            })
        else:
            answers.append({
                "type": "score", "name": q["name"], "score": 1.0,
                "probabilities": [
                    {"value": i, "label": lv["label"], "probability": 1.0 if i == 1 else 0.0}
                    for i, lv in enumerate(q["levels"])
                ],
                "confidence": 1.0,
            })
    return {
        "model": "gpt-6-luna", "answers": answers,
        "usage": {"input_tokens": 463, "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 0},
                  "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 463},
    }


def _fill_schema(schema):
    """Plausible structured output for a build_batch_schema() class."""
    data = {}
    for name, field in schema.model_fields.items():
        sub = field.annotation
        if "probability" in sub.model_fields:
            data[name] = {"probability": 0.7}
        else:
            keys = list(sub.model_fields)
            data[name] = {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(keys)}
    return data


def _settings(**overrides):
    base = dict(quick_enabled=True, quick_model=LUNA, structured_model=NANO, structured_thinking_level=None)
    base.update(overrides)
    return DecisionsSettings(**base)


@override_settings(
    DECISIONS_QUICK_TIMEOUT_SECONDS=30, DECISIONS_QUICK_MAX_RETRIES=1,
    DECISIONS_QUICK_SKIP_SECONDS=60, DECISIONS_STRUCTURED_BATCH_SIZE=25,
    DECISIONS_MAX_INPUT_TOKENS=100_000,
)
class DecisionsServiceTests(TestCase):
    SKIP_KEY = "decisions_quick_skip:test-service"

    def setUp(self):
        cache.delete(self.SKIP_KEY)
        self.addCleanup(cache.delete, self.SKIP_KEY)
        patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}).start()
        self.prefs = patch("core.preferences.resolve_decisions_settings", return_value=_settings()).start()
        self.client = MagicMock()
        self.client.post.side_effect = lambda path, **kw: _raw_from_body(kw["body"])
        patch("llm.service.decisions_quick.get_openai_client", return_value=self.client).start()
        self.llm = patch("llm.service.decisions_structured._get_llm_service").start().return_value
        self.llm.run_structured.side_effect = lambda req, schema: (
            schema.model_validate(_fill_schema(schema)),
            Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        )
        self.addCleanup(patch.stopall)
        self.availability = QuickAvailability(key=self.SKIP_KEY)
        self.service = DecisionsService(availability=self.availability)

    def _decide(self, questions=None, **kwargs):
        kwargs.setdefault("feature_key", "smoke")
        kwargs.setdefault("org_id", 1)
        if questions is None:
            questions = [_pred(), _choice(), _score()]
        return self.service.decide(questions, "A poster.", **kwargs)

    def _quick_rows(self):
        return [r for r in LLMCallLog.objects.all() if (r.response_metadata or {}).get("pathway") == "quick"]

    # --- the quick pathway -------------------------------------------------

    def test_quick_happy_path(self):
        context = RunContext.create(user_id=None, conversation_id="conv-9")
        result = self._decide(context=context)

        self.assertEqual(result.pathway, "quick")
        self.assertFalse(result.fell_back)
        self.assertEqual([a.type for a in result.answers], ["predicate", "choice", "score"])
        self.assertEqual(result["field"].choice, "chemistry")
        self.assertEqual(result["potential"].probabilities[1].value, 1)
        self.assertEqual(result.usage.prompt_tokens, 463)
        self.assertGreater(result.usage.cost_usd, 0)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].model, LUNA)
        self.client.post.assert_called_once()
        self.llm.run_structured.assert_not_called()
        self.prefs.assert_called_once_with(1)
        body = self.client.post.call_args.kwargs["body"]
        self.assertEqual(body["prompt_cache_key"], "decisions:smoke")

        rows = self._quick_rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.status, "success")
        self.assertEqual(row.model, LUNA)
        self.assertEqual(row.provider_model_id, "gpt-6-luna")
        self.assertEqual(row.conversation_id, "conv-9")
        self.assertEqual(row.input_tokens, 463)
        self.assertEqual(row.cached_tokens, 100)
        self.assertEqual(row.output_tokens, 0)
        self.assertIsNotNone(row.cost_usd)
        self.assertEqual(row.response_metadata["kind"], "decisions")
        self.assertEqual(row.response_metadata["feature"], "smoke")
        self.assertEqual(row.response_metadata["types"], {"predicate": 1, "choice": 1, "score": 1})

    def test_explicit_prompt_cache_key_and_dict_questions(self):
        result = self._decide(
            [{"type": "predicate", "name": "a", "instructions": "?"}], prompt_cache_key="decisions:custom",
        )
        self.assertEqual(result["a"].type, "predicate")
        self.assertEqual(self.client.post.call_args.kwargs["body"]["prompt_cache_key"], "decisions:custom")

    # --- pathway selection -------------------------------------------------

    def test_structured_when_quick_is_not_available(self):
        cases = {
            "toggle off": _settings(quick_enabled=False),
            "no capable model": _settings(quick_model=""),
        }
        for label, prefs in cases.items():
            with self.subTest(case=label):
                self.prefs.return_value = prefs
                result = self._decide()
                self.assertEqual(result.pathway, "structured")
                self.assertFalse(result.fell_back)
                self.assertEqual([a.type for a in result.answers], ["predicate", "choice", "score"])
                self.assertEqual(result["disclosed"].probability, 0.7)
        self.client.post.assert_not_called()
        # The structured pathway's rows come from LLMService.run (mocked here):
        # the service itself writes nothing for it.
        self.assertEqual(LLMCallLog.objects.count(), 0)

    def test_structured_without_openai_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            result = self._decide()
        self.assertEqual(result.pathway, "structured")
        self.client.post.assert_not_called()

    def test_structured_while_the_skip_window_is_open(self):
        self.availability.suspend("timeout")
        result = self._decide()
        self.assertEqual(result.pathway, "structured")
        self.client.post.assert_not_called()

    def test_structured_request_carries_the_org_settings(self):
        self.prefs.return_value = _settings(quick_enabled=False, structured_thinking_level="low")
        self._decide()
        request = self.llm.run_structured.call_args.args[0]
        self.assertEqual(request.model, NANO)
        self.assertEqual(request.params["thinking_level"], "low")
        self.assertEqual(request.params["_log_metadata"]["feature"], "smoke")

    def test_no_model_at_all(self):
        self.prefs.return_value = _settings(quick_enabled=False, structured_model="")
        with self.assertRaises(LLMConfigurationError):
            self._decide()

    # --- fallback ----------------------------------------------------------

    def test_quick_failure_falls_back_and_opens_the_skip_window(self):
        self.client.post.side_effect = openai.APITimeoutError(request=httpx.Request("POST", _URL))
        with self.assertLogs("llm.service.decisions_quick", level="WARNING") as cm:
            result = self._decide()

        self.assertEqual(result.pathway, "structured")
        self.assertTrue(result.fell_back)
        self.assertEqual([a.type for a in result.answers], ["predicate", "choice", "score"])
        self.assertEqual([c.pathway for c in result.calls], ["quick", "structured"])
        self.assertEqual(result.calls[0].error_code, "timeout")
        self.assertIsNone(result.calls[0].usage)
        self.assertEqual(result.usage.prompt_tokens, 100)  # structured call only
        self.assertTrue(self.availability.is_suspended())
        self.assertEqual(len([r for r in cm.records if r.levelno >= logging.WARNING]), 1)

        rows = self._quick_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "error")
        self.assertEqual(rows[0].error_type, "timeout")
        self.assertEqual(rows[0].response_metadata["error_code"], "timeout")

    def test_malformed_response_falls_back(self):
        self.client.post.side_effect = lambda path, **kw: {"model": "gpt-6-luna"}
        with self.assertLogs("llm.service.decisions_quick", level="ERROR"):
            result = self._decide()
        self.assertEqual(result.pathway, "structured")
        self.assertEqual(result.calls[0].error_code, "malformed_response")

    def test_quick_failure_without_a_structured_model_raises(self):
        self.prefs.return_value = _settings(structured_model="")
        self.client.post.side_effect = openai.APITimeoutError(request=httpx.Request("POST", _URL))
        with self.assertRaises(LLMTimeoutError):
            self._decide()
        self.assertTrue(self.availability.is_suspended())
        self.llm.run_structured.assert_not_called()

    def test_programmer_errors_do_not_fall_back(self):
        response = httpx.Response(400, request=httpx.Request("POST", _URL))
        cases = [
            (openai.BadRequestError("maximum context length is 1050000 tokens", response=response, body=None),
             LLMRequestTooLargeError),
            (openai.BadRequestError("Invalid questions[0].name", response=response, body=None), LLMProviderError),
        ]
        for exc, expected in cases:
            with self.subTest(expected=expected.__name__):
                self.client.post.side_effect = exc
                with self.assertRaises(expected):
                    self._decide()
        self.llm.run_structured.assert_not_called()
        self.assertFalse(self.availability.is_suspended())
        self.assertEqual(len([r for r in self._quick_rows() if r.status == "error"]), 2)

    # --- validation --------------------------------------------------------

    def test_bad_questions_and_input(self):
        with self.assertRaises(ValueError):
            self._decide([])
        with self.assertRaises(ValueError):
            self._decide([_pred("a"), _pred("a")])
        with self.assertRaises(ValueError):
            self.service.decide([_pred()], "", feature_key="smoke", org_id=1)
        with self.assertRaises(ValueError):
            self.service.decide([_pred()], "x", feature_key="", org_id=1)
        self.client.post.assert_not_called()
        self.prefs.assert_not_called()

    @override_settings(DECISIONS_MAX_INPUT_TOKENS=10)
    def test_input_cap_raises_before_any_call(self):
        with self.assertRaises(LLMRequestTooLargeError):
            self.service.decide([_pred()], "many words " * 50, feature_key="smoke", org_id=1)
        self.client.post.assert_not_called()
        self.llm.run_structured.assert_not_called()

    # --- batching ----------------------------------------------------------

    def test_quick_batches_of_400_in_order(self):
        questions = [_pred(f"q{i}") for i in range(401)]
        result = self._decide(questions)
        self.assertEqual(self.client.post.call_count, 2)
        sizes = [len(c.kwargs["body"]["questions"]) for c in self.client.post.call_args_list]
        self.assertEqual(sizes, [400, 1])
        self.assertEqual([a.name for a in result.answers], [q.name for q in questions])
        self.assertEqual(result.usage.prompt_tokens, 463 * 2)
        batches = sorted((r.response_metadata["batch"], r.response_metadata["batches"]) for r in self._quick_rows())
        self.assertEqual(batches, [(1, 2), (2, 2)])

    def test_later_quick_batch_failure_requeues_only_the_rest(self):
        questions = [_pred(f"q{i}") for i in range(401)]
        raw = _raw_from_body
        self.client.post.side_effect = [
            raw({"questions": [q.to_api() for q in questions[:400]]}),
            openai.APITimeoutError(request=httpx.Request("POST", _URL)),
        ]
        result = self._decide(questions)
        self.assertEqual(result.pathway, "mixed")
        self.assertTrue(result.fell_back)
        self.assertEqual([c.pathway for c in result.calls], ["quick", "quick", "structured"])
        self.assertEqual(self.llm.run_structured.call_count, 1)  # the one requeued question
        self.assertEqual([a.name for a in result.answers], [q.name for q in questions])
        self.assertEqual(result.answers[400].probability, 0.7)  # answered by the structured pathway
        self.assertEqual(result.answers[0].probability, 0.9)

    def test_structured_batches(self):
        self.prefs.return_value = _settings(quick_enabled=False)
        questions = [_pred(f"q{i}") for i in range(60)]
        result = self._decide(questions)
        self.assertEqual(self.llm.run_structured.call_count, 3)
        metas = [c.args[0].params["_log_metadata"] for c in self.llm.run_structured.call_args_list]
        self.assertEqual([(m["batch"], m["batches"], m["questions"]) for m in metas], [(1, 3, 25), (2, 3, 25), (3, 3, 10)])
        self.assertEqual([a.name for a in result.answers], [q.name for q in questions])
        self.assertEqual(result.usage.prompt_tokens, 300)
        self.assertEqual(len(result.calls), 3)

    # --- async + singleton -------------------------------------------------

    def test_adecide(self):
        # adecide runs decide() on a worker thread, i.e. on another DB connection
        # outside this test's transaction — take the structured pathway (no row
        # written here) so nothing leaks into the other tests.
        self.prefs.return_value = _settings(quick_enabled=False)
        result = asyncio.run(self.service.adecide([_pred()], "x", feature_key="smoke", org_id=1))
        self.assertEqual(result.pathway, "structured")
        self.assertEqual(result["disclosed"].probability, 0.7)

    def test_singleton(self):
        self.assertIs(get_decisions_service(), get_decisions_service())
