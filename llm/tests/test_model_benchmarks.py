"""Tests for the benchmark scores behind the model-picker guide."""

import json

from django.test import SimpleTestCase

from llm.model_benchmarks import (
    BENCHMARK_ARENA_BUSINESS,
    BENCHMARK_GDPVAL,
    BENCHMARK_OMNISCIENCE,
    BENCHMARKS,
    build_model_guide,
    get_benchmark,
    get_scores,
    missing_models,
)
from llm.model_registry import get_model_info, get_registered_model_ids


class BenchmarkDataTests(SimpleTestCase):
    def test_rows_reference_registered_models_and_valid_efforts(self):
        registered = set(get_registered_model_ids())
        for key, bench in BENCHMARKS.items():
            self.assertEqual(bench.key, key)
            self.assertTrue(bench.source_url.startswith("https://"))
            for s in bench.scores:
                with self.subTest(bench=key, model=s.model_id, effort=s.effort):
                    self.assertIn(s.model_id, registered)
                    self.assertTrue(s.source_label)
                    if s.effort is not None:
                        info = get_model_info(s.model_id)
                        # "none" is OpenAI's off switch; Anthropic's is "off".
                        allowed = set(info.reasoning_levels) | {"none", "off"}
                        self.assertIn(s.effort, allowed)

    def test_one_row_per_model_and_effort(self):
        for key, bench in BENCHMARKS.items():
            seen = [(s.model_id, s.effort) for s in bench.scores]
            with self.subTest(bench=key):
                self.assertEqual(len(seen), len(set(seen)))

    def test_missing_models(self):
        self.assertEqual(missing_models(BENCHMARK_ARENA_BUSINESS), ["anthropic/claude-sonnet-5-5"])
        self.assertEqual(missing_models(BENCHMARK_GDPVAL), [])
        for key in BENCHMARKS:
            covered = {s.model_id for s in get_scores(key)}
            self.assertEqual(
                set(missing_models(key)) | covered, set(get_registered_model_ids()),
            )

    def test_get_scores_filters_by_model_and_follows_replacements(self):
        rows = get_scores(BENCHMARK_GDPVAL, "anthropic/claude-opus-5-5")
        self.assertEqual(
            {s.effort for s in rows}, {"low", "medium", "high", "xhigh", "max"},
        )
        # Retired ID resolves through MODEL_REPLACEMENTS.
        self.assertEqual(
            get_scores(BENCHMARK_GDPVAL, "openai/gpt-5.5"),
            get_scores(BENCHMARK_GDPVAL, "openai/gpt-5.6-sol"),
        )
        self.assertEqual(get_scores("nope"), [])

    def test_every_benchmark_has_a_plain_language_title_and_question(self):
        for bench in BENCHMARKS.values():
            with self.subTest(bench=bench.key):
                self.assertTrue(bench.title)
                self.assertTrue(bench.question.endswith("?"))

    def test_hallucination_rate_is_lower_is_better(self):
        self.assertFalse(get_benchmark(BENCHMARK_OMNISCIENCE).higher_is_better)


class ModelGuideTests(SimpleTestCase):
    ALLOWED = ["anthropic/claude-opus-5-5", "openai/gpt-6-luna", "anthropic/claude-sonnet-5-5"]

    def test_only_allowed_models_are_included(self):
        guide = build_model_guide(self.ALLOWED)
        self.assertEqual([m["id"] for m in guide["models"]], self.ALLOWED)
        for bench in guide["benchmarks"]:
            with self.subTest(bench=bench["key"]):
                self.assertLessEqual({r["model_id"] for r in bench["rows"]}, set(self.ALLOWED))

    def test_missing_lists_allowed_models_without_scores(self):
        guide = build_model_guide(self.ALLOWED)
        by_key = {b["key"]: b for b in guide["benchmarks"]}
        self.assertEqual(by_key[BENCHMARK_ARENA_BUSINESS]["missing"], ["anthropic/claude-sonnet-5-5"])
        self.assertEqual(by_key[BENCHMARK_GDPVAL]["missing"], [])

    def test_models_carry_picker_price_and_star_levels(self):
        guide = build_model_guide(self.ALLOWED)
        opus = guide["models"][0]
        self.assertEqual(opus["display_name"], "Claude Opus 5.5")
        self.assertEqual(opus["price_level"], 4)  # $20 / 1M output
        self.assertEqual(opus["capability_level"], 4)

    def test_every_effort_row_and_metadata_is_serialisable(self):
        guide = build_model_guide(self.ALLOWED)
        json.dumps(guide)
        gdpval = guide["benchmarks"][0]
        self.assertEqual(gdpval["key"], BENCHMARK_GDPVAL)
        opus_efforts = {r["effort"] for r in gdpval["rows"] if r["model_id"] == "anthropic/claude-opus-5-5"}
        self.assertEqual(opus_efforts, {"low", "medium", "high", "xhigh", "max"})
        omni = next(b for b in guide["benchmarks"] if b["key"] == BENCHMARK_OMNISCIENCE)
        self.assertFalse(omni["higher_is_better"])

    def test_non_canonical_ids_are_ignored(self):
        guide = build_model_guide(["anthropic/claude-opus-5", "bogus"])
        self.assertEqual(guide["models"], [])
