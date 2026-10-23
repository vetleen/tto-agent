"""Tests for llm.types.decisions — question validation, input, answers, result."""

from django.test import SimpleTestCase
from pydantic import ValidationError

from llm.types.decisions import (
    CallRecord,
    Choice,
    ChoiceAnswer,
    ChoiceProbability,
    DecisionsResult,
    InputMessage,
    JsonPart,
    Level,
    PredicateAnswer,
    Question,
    RefusalAnswer,
    TextPart,
    chunk_questions,
    clamp_probability,
    estimate_input_tokens,
    normalize_input,
    sum_usage,
    validate_questions,
)
from llm.types.responses import Usage


def _choices(*values):
    return [Choice(value=v) for v in values]


def _levels(*labels):
    return [Level(label=lb) for lb in labels]


class QuestionValidationTests(SimpleTestCase):
    def test_predicate_takes_neither_choices_nor_levels(self):
        Question(type="predicate", name="is_public", instructions="Was it disclosed?")
        with self.assertRaises(ValidationError):
            Question(type="predicate", name="p", instructions="?", choices=_choices("a", "b"))
        with self.assertRaises(ValidationError):
            Question(type="predicate", name="p", instructions="?", levels=_levels("lo", "hi"))

    def test_choice_needs_two_unique_choices(self):
        Question(type="choice", name="c", instructions="?", choices=_choices("a", "b"))
        for bad in (None, [], _choices("a"), _choices("a", "a")):
            with self.subTest(choices=bad), self.assertRaises(ValidationError):
                Question(type="choice", name="c", instructions="?", choices=bad)
        with self.assertRaises(ValidationError):
            Question(type="choice", name="c", instructions="?", choices=_choices("a", "b"), levels=_levels("x", "y"))

    def test_score_needs_2_to_10_unique_levels(self):
        Question(type="score", name="s", instructions="?", levels=_levels("lo", "hi"))
        Question(type="score", name="s", instructions="?", levels=_levels(*[str(i) for i in range(10)]))
        for bad in (None, _levels("only"), _levels(*[str(i) for i in range(11)]), _levels("x", "x")):
            with self.subTest(levels=bad), self.assertRaises(ValidationError):
                Question(type="score", name="s", instructions="?", levels=bad)
        with self.assertRaises(ValidationError):
            Question(type="score", name="s", instructions="?", levels=_levels("lo", "hi"), choices=_choices("a", "b"))

    def test_name_pattern_and_instructions(self):
        for ok in ("a", "snake_case", "dotted.name", "with-dash", "x" * 64):
            Question(type="predicate", name=ok, instructions="?")
        for bad in ("", "has space", "x" * 65, "ünïcode", "semi;colon"):
            with self.subTest(name=bad), self.assertRaises(ValidationError):
                Question(type="predicate", name=bad, instructions="?")
        with self.assertRaises(ValidationError):
            Question(type="predicate", name="p", instructions="")

    def test_to_api_is_the_wire_shape(self):
        q = Question(
            type="choice", name="dept", instructions="Which department?",
            choices=[Choice(value="billing", description="Payments."), Choice(value="other")],
        )
        self.assertEqual(q.to_api(), {
            "type": "choice", "name": "dept", "instructions": "Which department?",
            "choices": [{"value": "billing", "description": "Payments."}, {"value": "other"}],
        })
        s = Question(type="score", name="sev", instructions="How severe?", levels=_levels("Low", "High"))
        self.assertEqual(s.to_api()["levels"], [{"label": "Low"}, {"label": "High"}])
        self.assertNotIn("choices", s.to_api())
        p = Question(type="predicate", name="p", instructions="?")
        self.assertEqual(p.to_api(), {"type": "predicate", "name": "p", "instructions": "?"})
        self.assertEqual(q.option_count, 2)
        self.assertEqual(p.option_count, 0)

    def test_validate_questions(self):
        qs = validate_questions([
            {"type": "predicate", "name": "a", "instructions": "?"},
            Question(type="predicate", name="b", instructions="?"),
        ])
        self.assertEqual([q.name for q in qs], ["a", "b"])
        with self.assertRaises(ValueError):
            validate_questions([])
        with self.assertRaisesRegex(ValueError, "duplicated: a"):
            validate_questions([
                Question(type="predicate", name="a", instructions="?"),
                Question(type="predicate", name="a", instructions="?"),
            ])

    def test_chunk_questions(self):
        qs = [Question(type="predicate", name=f"q{i}", instructions="?") for i in range(401)]
        batches = chunk_questions(qs, 400)
        self.assertEqual([len(b) for b in batches], [400, 1])
        self.assertEqual(batches[1][0].name, "q400")
        self.assertEqual(chunk_questions(qs[:5], 2), [qs[0:2], qs[2:4], qs[4:5]])
        with self.assertRaises(ValueError):
            chunk_questions(qs, 0)


class InputTests(SimpleTestCase):
    def test_string_becomes_one_user_message(self):
        messages = normalize_input("Export fails in Safari.")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, [TextPart(text="Export fails in Safari.")])
        for empty in ("", "   ", []):
            with self.subTest(input=empty), self.assertRaises(ValueError):
                normalize_input(empty)

    def test_message_list_is_validated(self):
        messages = normalize_input([
            {"role": "developer", "content": [{"type": "input_text", "text": "Be strict."}]},
            InputMessage(content=[JsonPart(data={"a": 1})]),
        ])
        self.assertEqual(messages[0].role, "developer")
        self.assertEqual(messages[1].role, "user")
        self.assertIsInstance(messages[1].content[0], JsonPart)
        with self.assertRaises(ValidationError):
            InputMessage(content=[])
        with self.assertRaises(ValidationError):
            InputMessage(role="tool", content=[TextPart(text="x")])
        with self.assertRaises(ValidationError):
            JsonPart(data={"when": object()})

    def test_json_part_is_capped_at_the_endpoint_limit(self):
        from llm.types.decisions import MAX_JSON_PART_BYTES

        JsonPart(data={"text": "x" * (MAX_JSON_PART_BYTES - 20)})
        with self.assertRaisesRegex(ValidationError, "split it or pass it as text"):
            JsonPart(data={"text": "x" * MAX_JSON_PART_BYTES})

    def test_estimate_input_tokens_counts_text_and_json(self):
        text_only = normalize_input("hello world " * 50)
        with_json = text_only + [InputMessage(content=[JsonPart(data={"k": "v" * 400})])]
        self.assertGreater(estimate_input_tokens(text_only), 50)
        self.assertGreater(estimate_input_tokens(with_json), estimate_input_tokens(text_only))


class AnswerAndResultTests(SimpleTestCase):
    def _result(self, calls):
        answers = [
            PredicateAnswer(name="p", probability=0.9),
            ChoiceAnswer(name="c", choice="a", probabilities=[ChoiceProbability(value="a", probability=1.0)], confidence=1.0),
            RefusalAnswer(name="r"),
        ]
        return DecisionsResult(answers=answers, usage=Usage(), calls=calls)

    def test_lookup_helpers(self):
        result = self._result([CallRecord(pathway="quick", model="m", questions=3, duration_ms=1)])
        self.assertEqual(result["p"].probability, 0.9)
        self.assertEqual(result.by_name["c"].choice, "a")
        self.assertTrue(result.refused("r"))
        self.assertFalse(result.refused("p"))
        with self.assertRaises(KeyError):
            result["missing"]

    def test_pathway_is_derived_from_successful_calls(self):
        quick = CallRecord(pathway="quick", model="m", questions=3, duration_ms=1)
        failed_quick = CallRecord(pathway="quick", model="m", questions=3, duration_ms=1, error_code="timeout")
        structured = CallRecord(pathway="structured", model="n", questions=3, duration_ms=1)
        self.assertEqual(self._result([quick]).pathway, "quick")
        self.assertEqual(self._result([failed_quick, structured]).pathway, "structured")
        self.assertEqual(self._result([quick, failed_quick, structured]).pathway, "mixed")

    def test_sum_usage(self):
        total = sum_usage([
            Usage(prompt_tokens=10, cached_tokens=4, cost_usd=0.001),
            None,
            Usage(prompt_tokens=5, completion_tokens=2, cost_usd=0.002),
        ])
        self.assertEqual(total.prompt_tokens, 15)
        self.assertEqual(total.completion_tokens, 2)
        self.assertEqual(total.cached_tokens, 4)
        self.assertIsNone(total.total_tokens)
        self.assertAlmostEqual(total.cost_usd, 0.003)

    def test_clamp_probability(self):
        self.assertEqual(clamp_probability(1.7), 1.0)
        self.assertEqual(clamp_probability(-0.2), 0.0)
        self.assertEqual(clamp_probability(0.42), 0.42)
        self.assertEqual(clamp_probability(float("nan")), 0.0)
        self.assertEqual(clamp_probability(float("inf")), 0.0)
        self.assertEqual(clamp_probability("x"), 0.0)
