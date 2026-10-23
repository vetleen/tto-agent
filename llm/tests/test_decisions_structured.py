"""Tests for the structured decisions pathway (llm.service.decisions_structured)."""

from unittest.mock import patch

from django.test import SimpleTestCase

from llm.service.decisions_structured import (
    SYSTEM_PROMPT,
    build_batch_schema,
    map_structured_answers,
    render_messages,
    run_structured_batch,
)
from llm.service.errors import LLMProviderError, LLMTimeoutError
from llm.types.context import RunContext
from llm.types.decisions import (
    Choice,
    InputMessage,
    JsonPart,
    Level,
    Question,
    RefusalAnswer,
    TextPart,
    normalize_input,
)
from llm.types.responses import Usage


def _pred(name="disclosed"):
    return Question(type="predicate", name=name, instructions="Was the invention publicly disclosed?")


def _choice(name="dept"):
    return Question(
        type="choice", name=name, instructions="Which department?",
        choices=[Choice(value="billing", description="Payments."), Choice(value="tech"), Choice(value="other")],
    )


def _score(name="severity"):
    return Question(
        type="score", name=name, instructions="How severe?",
        levels=[Level(label="Cosmetic", description="Looks only."), Level(label="Workaround"), Level(label="Blocked")],
    )


def _walk(node, found):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                found.append(key)
            _walk(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk(item, found)


class SchemaTests(SimpleTestCase):
    def test_one_field_per_question_with_indexed_distributions(self):
        schema = build_batch_schema([_pred(), _choice(), _score()])
        self.assertEqual(list(schema.model_fields), ["q_0", "q_1", "q_2"])
        instance = schema.model_validate({
            "q_0": {"probability": 0.9},
            "q_1": {"p_0": 0.7, "p_1": 0.2, "p_2": 0.1},
            "q_2": {"p_0": 0.1, "p_1": 0.7, "p_2": 0.2},
        })
        self.assertEqual(instance.q_1.p_0, 0.7)
        # Every object is strict-mode friendly: all fields required, no extras.
        json_schema = schema.model_json_schema()
        self.assertFalse(json_schema["additionalProperties"])
        self.assertEqual(json_schema["required"], ["q_0", "q_1", "q_2"])
        for name, definition in json_schema["$defs"].items():
            with self.subTest(definition=name):
                self.assertFalse(definition["additionalProperties"])
                self.assertEqual(definition["required"], list(definition["properties"]))
        # Bounds live in descriptions, not numeric keywords (provider support varies).
        bounds = []
        _walk(json_schema, bounds)
        self.assertEqual(bounds, [])
        with self.assertRaises(Exception):
            schema.model_validate({"q_0": {"probability": 0.9}, "q_1": {"p_0": 1.0, "p_1": 0.0}, "q_2": {}})


class PromptTests(SimpleTestCase):
    def test_system_message_enumerates_questions_and_options(self):
        messages = render_messages([_pred(), _choice(), _score()], normalize_input("Shown on a poster."))
        system = messages[0]
        self.assertEqual(system.role, "system")
        self.assertIn(SYSTEM_PROMPT, system.content)
        self.assertIn("0. [predicate] disclosed: Was the invention publicly disclosed?", system.content)
        self.assertIn("1. [choice] dept: Which department?", system.content)
        self.assertIn("p_0 = 'billing' — Payments.", system.content)
        self.assertIn("p_1 = 'tech'", system.content)
        self.assertIn("levels, lowest first", system.content)
        self.assertIn("p_0 = level 0, 'Cosmetic' — Looks only.", system.content)
        self.assertIn("p_2 = level 2, 'Blocked'", system.content)
        self.assertNotIn("CALLER INSTRUCTIONS", system.content)
        # The input, then a closing user instruction.
        self.assertEqual([m.role for m in messages], ["system", "user", "user"])
        self.assertEqual(messages[1].content, "Shown on a poster.")

    def test_developer_parts_go_to_the_system_message_and_json_is_fenced(self):
        input = normalize_input([
            InputMessage(role="developer", content=[TextPart(text="Treat posters as public.")]),
            InputMessage(role="user", content=[TextPart(text="Facts:"), JsonPart(data={"venue": "poster"})]),
            InputMessage(role="assistant", content=[TextPart(text="Noted.")]),
        ])
        messages = render_messages([_pred()], input)
        self.assertIn("CALLER INSTRUCTIONS\nTreat posters as public.", messages[0].content)
        self.assertEqual([m.role for m in messages], ["system", "user", "assistant", "user"])
        self.assertIn('Facts:\n\n```json\n{\n  "venue": "poster"\n}\n```', messages[1].content)
        self.assertEqual(messages[2].content, "Noted.")


class MappingTests(SimpleTestCase):
    def test_predicate_is_clamped(self):
        batch = [_pred("a"), _pred("b"), _pred("c")]
        parsed = build_batch_schema(batch).model_validate({
            "q_0": {"probability": 0.42}, "q_1": {"probability": 1.7}, "q_2": {"probability": -0.3},
        })
        answers = map_structured_answers(batch, parsed)
        self.assertEqual([a.type for a in answers], ["predicate"] * 3)
        self.assertEqual([a.probability for a in answers], [0.42, 1.0, 0.0])
        self.assertEqual([a.name for a in answers], ["a", "b", "c"])

    def test_choice_is_argmax_of_the_renormalised_distribution(self):
        batch = [_choice()]
        parsed = build_batch_schema(batch).model_validate({"q_0": {"p_0": 2.0, "p_1": 1.0, "p_2": 1.0}})
        answer = map_structured_answers(batch, parsed)[0]
        self.assertEqual(answer.type, "choice")
        self.assertEqual(answer.choice, "billing")
        # 2.0 is clamped to 1.0 first, then the distribution is renormalised.
        self.assertEqual([round(p.probability, 4) for p in answer.probabilities], [0.3333, 0.3333, 0.3333])
        self.assertEqual([p.value for p in answer.probabilities], ["billing", "tech", "other"])
        self.assertAlmostEqual(answer.confidence, 1 / 3)

    def test_score_is_the_weighted_mean_of_zero_based_levels(self):
        batch = [_score()]
        parsed = build_batch_schema(batch).model_validate({"q_0": {"p_0": 0.01, "p_1": 0.56, "p_2": 0.43}})
        answer = map_structured_answers(batch, parsed)[0]
        self.assertEqual(answer.type, "score")
        self.assertAlmostEqual(answer.score, 1.42)
        self.assertEqual([p.value for p in answer.probabilities], [0, 1, 2])
        self.assertEqual([p.label for p in answer.probabilities], ["Cosmetic", "Workaround", "Blocked"])
        self.assertAlmostEqual(answer.confidence, 0.56)

    def test_all_zero_distribution_becomes_uniform(self):
        batch = [_choice()]
        parsed = build_batch_schema(batch).model_validate({"q_0": {"p_0": 0.0, "p_1": 0.0, "p_2": 0.0}})
        answer = map_structured_answers(batch, parsed)[0]
        self.assertEqual([round(p.probability, 4) for p in answer.probabilities], [0.3333] * 3)
        self.assertEqual(answer.choice, "billing")  # first wins on a tie


class RunStructuredBatchTests(SimpleTestCase):
    def _service(self, side_effect):
        service = patch("llm.service.decisions_structured._get_llm_service").start()
        self.addCleanup(patch.stopall)
        service.return_value.run_structured.side_effect = side_effect
        return service.return_value

    def test_request_shape_and_answers(self):
        seen = {}

        def run_structured(request, schema):
            seen["request"] = request
            return schema.model_validate({"q_0": {"probability": 0.8}, "q_1": {"p_0": 0.9, "p_1": 0.1, "p_2": 0.0}}), Usage(prompt_tokens=50, completion_tokens=20)

        self._service(run_structured)
        context = RunContext.create(user_id=7)
        answers, usage, duration_ms = run_structured_batch(
            [_pred(), _choice()], normalize_input("x"), model="openai/gpt-5.4-nano",
            thinking_level="low", feature_key="smoke", context=context, batch_index=2, batch_count=3,
        )
        request = seen["request"]
        self.assertEqual(request.model, "openai/gpt-5.4-nano")
        self.assertFalse(request.stream)
        self.assertEqual(request.tools, [])
        self.assertIs(request.context, context)
        self.assertEqual(request.params["thinking_level"], "low")
        self.assertEqual(request.params["_log_metadata"], {
            "kind": "decisions", "feature": "smoke", "pathway": "structured",
            "questions": 2, "batch": 2, "batches": 3,
        })
        self.assertEqual([a.type for a in answers], ["predicate", "choice"])
        self.assertEqual(answers[1].choice, "billing")
        self.assertEqual(usage.prompt_tokens, 50)
        self.assertGreaterEqual(duration_ms, 0)

    def test_no_thinking_level_leaves_the_registry_default_to_llm_service(self):
        seen = {}

        def run_structured(request, schema):
            seen["request"] = request
            return schema.model_validate({"q_0": {"probability": 0.5}}), None

        self._service(run_structured)
        run_structured_batch(
            [_pred()], normalize_input("x"), model="m", thinking_level=None,
            feature_key="f", context=RunContext.create(),
        )
        self.assertNotIn("thinking_level", seen["request"].params)

    def test_provider_refusal_refuses_the_whole_batch(self):
        self._service(LLMProviderError("refused", error_code="provider_refusal"))
        answers, usage, _ = run_structured_batch(
            [_pred("a"), _choice("b")], normalize_input("x"), model="m", thinking_level=None,
            feature_key="f", context=RunContext.create(),
        )
        self.assertEqual(answers, [RefusalAnswer(name="a"), RefusalAnswer(name="b")])
        self.assertIsNone(usage)

    def test_other_errors_propagate(self):
        for exc in (LLMProviderError("parse", error_code="structured_output_parse_error"), LLMTimeoutError("slow")):
            with self.subTest(exc=exc):
                self._service(exc)
                with self.assertRaises(type(exc)):
                    run_structured_batch(
                        [_pred()], normalize_input("x"), model="m", thinking_level=None,
                        feature_key="f", context=RunContext.create(),
                    )
