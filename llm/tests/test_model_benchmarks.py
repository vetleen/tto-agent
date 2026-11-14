"""Tests for the benchmark scores behind the model-picker guide."""

import json

from django.test import SimpleTestCase

from llm.model_benchmarks import (
    BENCHMARK_ARENA_BUSINESS,
    BENCHMARK_GDP_PDF,
    BENCHMARK_GDPVAL,
    BENCHMARK_LIVEBENCH_IF,
    BENCHMARK_OMNISCIENCE,
    BENCHMARK_RESPONSE_TIME,
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
        # A model the source has not measured is unranked, never shown with
        # its retired predecessor's score.
        self.assertCountEqual(
            missing_models(BENCHMARK_ARENA_BUSINESS),
            ["openai/gpt-6.1-terra", "anthropic/claude-fable-5-2"],
        )
        self.assertEqual(missing_models(BENCHMARK_GDPVAL), ["gemini/gemini-4-hydrogen"])
        self.assertEqual(missing_models(BENCHMARK_GDP_PDF), [])
        self.assertEqual(missing_models(BENCHMARK_OMNISCIENCE), [])
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
        sol = get_scores(BENCHMARK_GDPVAL, "openai/gpt-6.1-sol")
        self.assertTrue(sol)
        for retired in ("openai/gpt-5.5", "openai/gpt-5.6-sol", "openai/gpt-6-sol"):
            self.assertEqual(get_scores(BENCHMARK_GDPVAL, retired), sol)
        self.assertEqual(get_scores("nope"), [])

    def test_every_benchmark_has_a_plain_language_title_and_question(self):
        for bench in BENCHMARKS.values():
            with self.subTest(bench=bench.key):
                self.assertTrue(bench.title)
                self.assertTrue(bench.question.endswith("?"))

    def test_page_order_and_scales(self):
        self.assertEqual(
            [(b.key, b.scale) for b in BENCHMARKS.values()],
            [
                (BENCHMARK_GDPVAL, "elo"),
                (BENCHMARK_ARENA_BUSINESS, "elo"),
                (BENCHMARK_GDP_PDF, "percent"),
                (BENCHMARK_LIVEBENCH_IF, "score"),
                (BENCHMARK_OMNISCIENCE, "percent"),
                (BENCHMARK_RESPONSE_TIME, "seconds"),
            ],
        )

    def test_livebench_if_missing_models(self):
        # Models released after the last LiveBench release with the IF table
        # (2026-06-25; LiveBench 2.0 replaced it) have no row.
        self.assertCountEqual(
            missing_models(BENCHMARK_LIVEBENCH_IF),
            [
                "openai/gpt-6.1-terra", "anthropic/claude-fable-5-2",
                "gemini/gemini-4-carbon", "gemini/gemini-4-argon", "gemini/gemini-4-hydrogen",
            ],
        )

    def test_gemini_4_carbon_scores(self):
        expected = {
            BENCHMARK_GDPVAL: [("high", 1733)],
            BENCHMARK_GDP_PDF: [("low", 28.6), ("medium", 29.6), ("high", 28.4)],
            BENCHMARK_OMNISCIENCE: [("low", 19.8), ("medium", 4.2), ("high", 11.4)],
            BENCHMARK_ARENA_BUSINESS: [("high", 1513)],
            BENCHMARK_LIVEBENCH_IF: [],
            BENCHMARK_RESPONSE_TIME: [],
        }
        for key, rows in expected.items():
            with self.subTest(key=key):
                got = get_scores(key, "gemini/gemini-4-carbon")
                self.assertEqual([(s.effort, s.score) for s in got], rows)

    def test_gemini_4_hydrogen_scores(self):
        model = "gemini/gemini-4-hydrogen"
        # Real Hydrogen numbers where the source has them (AA's xhigh run is
        # a level we do not offer, so it has no row); nothing elsewhere.
        expected = {
            BENCHMARK_GDP_PDF: [("low", 28.2), ("medium", 32.8), ("high", 30.0)],
            BENCHMARK_OMNISCIENCE: [("low", 18.8), ("medium", 12.5), ("high", 8.7)],
            BENCHMARK_ARENA_BUSINESS: [("high", 1476)],
            BENCHMARK_GDPVAL: [],
            BENCHMARK_LIVEBENCH_IF: [],
            BENCHMARK_RESPONSE_TIME: [],
        }
        for key, rows in expected.items():
            with self.subTest(key=key):
                got = get_scores(key, model)
                self.assertEqual([(s.effort, s.score) for s in got], rows)
        # The retired ID reads the same rows.
        self.assertEqual(
            get_scores(BENCHMARK_GDP_PDF, "gemini/gemini-3.8-flash"),
            get_scores(BENCHMARK_GDP_PDF, model),
        )

    def test_fable_5_2_scores(self):
        efforts = ["low", "medium", "high", "xhigh", "max"]
        for key in (BENCHMARK_GDPVAL, BENCHMARK_GDP_PDF, BENCHMARK_OMNISCIENCE):
            with self.subTest(key=key):
                rows = get_scores(key, "anthropic/claude-fable-5-2")
                self.assertEqual([s.effort for s in rows], efforts)
        gdpval = {s.effort: s.score for s in get_scores(BENCHMARK_GDPVAL, "anthropic/claude-fable-5-2")}
        self.assertEqual(gdpval["max"], 1831)
        # Not measured yet on these three: no row, not Fable 5.1's.
        for key in (BENCHMARK_ARENA_BUSINESS, BENCHMARK_LIVEBENCH_IF, BENCHMARK_RESPONSE_TIME):
            with self.subTest(key=key):
                self.assertEqual(get_scores(key, "anthropic/claude-fable-5-2"), [])

    def test_gemini_4_argon_scores(self):
        # AA runs Argon at medium and high (medium since 2026-11-13); LMArena
        # and the 100K timing only at high.
        expected = {
            BENCHMARK_GDPVAL: [("medium", 1578), ("high", 1633)],
            BENCHMARK_GDP_PDF: [("medium", 25.6), ("high", 21.8)],
            BENCHMARK_OMNISCIENCE: [("medium", 15.4), ("high", 4.3)],
            BENCHMARK_ARENA_BUSINESS: [("high", 1515)],
            BENCHMARK_RESPONSE_TIME: [("high", 86.1)],
        }
        for key, scores in expected.items():
            with self.subTest(key=key):
                rows = get_scores(key, "gemini/gemini-4-argon")
                self.assertEqual([(s.effort, s.score) for s in rows], scores)

    def test_haiku_5_5_scores(self):
        # Every effort level on AA; LiveBench lists only xhigh and max.
        efforts = ["low", "medium", "high", "xhigh", "max"]
        for key in (BENCHMARK_GDPVAL, BENCHMARK_GDP_PDF, BENCHMARK_OMNISCIENCE):
            with self.subTest(key=key):
                rows = get_scores(key, "anthropic/claude-haiku-5-5")
                self.assertEqual([s.effort for s in rows], efforts)
        self.assertEqual(
            [s.effort for s in get_scores(BENCHMARK_LIVEBENCH_IF, "anthropic/claude-haiku-5-5")],
            ["xhigh", "max"],
        )

    def test_gpt_6_1_terra_scores(self):
        # Every effort level, "none" included, on all four AA benchmarks.
        efforts = ["none", "low", "medium", "high", "xhigh", "max"]
        for key in (
            BENCHMARK_GDPVAL, BENCHMARK_GDP_PDF, BENCHMARK_OMNISCIENCE, BENCHMARK_RESPONSE_TIME,
        ):
            with self.subTest(key=key):
                rows = get_scores(key, "openai/gpt-6.1-terra")
                self.assertEqual([s.effort for s in rows], efforts)
        gdpval = {s.effort: s.score for s in get_scores(BENCHMARK_GDPVAL, "openai/gpt-6.1-terra")}
        self.assertEqual(gdpval["max"], 1476)
        # Not rated on LMArena or LiveBench IF: no row, not GPT-5.6 Terra's.
        for key in (BENCHMARK_ARENA_BUSINESS, BENCHMARK_LIVEBENCH_IF):
            with self.subTest(key=key):
                self.assertEqual(get_scores(key, "openai/gpt-6.1-terra"), [])

    def test_hallucination_rate_is_lower_is_better(self):
        self.assertFalse(get_benchmark(BENCHMARK_OMNISCIENCE).higher_is_better)

    def test_response_time(self):
        bench = get_benchmark(BENCHMARK_RESPONSE_TIME)
        self.assertFalse(bench.higher_is_better)
        # AA had no 100K timing for nano (2026-11-03); Fable 5.2, Carbon and
        # Hydrogen came out after the series could last be read.
        self.assertCountEqual(
            missing_models(BENCHMARK_RESPONSE_TIME),
            [
                "openai/gpt-5.4-nano", "anthropic/claude-fable-5-2",
                "gemini/gemini-4-carbon", "gemini/gemini-4-hydrogen",
            ],
        )
        for s in bench.scores:
            with self.subTest(model=s.model_id, effort=s.effort):
                self.assertGreater(s.score, 0)
        # Every effort AA timed is a row, so the guide can rank each model by
        # its fastest effort and list them all in the detail view.
        haiku = {s.effort: s.score for s in get_scores(BENCHMARK_RESPONSE_TIME, "anthropic/claude-haiku-5-5")}
        self.assertEqual(set(haiku), {"low", "medium", "high", "xhigh", "max"})
        self.assertEqual(min(haiku, key=haiku.get), "low")


class ModelGuideTests(SimpleTestCase):
    ALLOWED = ["anthropic/claude-opus-5-5", "openai/gpt-6-luna", "anthropic/claude-sonnet-5-5"]

    def test_only_allowed_models_are_included(self):
        guide = build_model_guide(self.ALLOWED)
        self.assertEqual([m["id"] for m in guide["models"]], self.ALLOWED)
        for bench in guide["benchmarks"]:
            with self.subTest(bench=bench["key"]):
                self.assertLessEqual({r["model_id"] for r in bench["rows"]}, set(self.ALLOWED))

    def test_missing_lists_allowed_models_without_scores(self):
        guide = build_model_guide([*self.ALLOWED, "gemini/gemini-4-carbon"])
        by_key = {b["key"]: b for b in guide["benchmarks"]}
        self.assertEqual(by_key[BENCHMARK_LIVEBENCH_IF]["missing"], ["gemini/gemini-4-carbon"])
        self.assertEqual(by_key[BENCHMARK_ARENA_BUSINESS]["missing"], [])
        self.assertEqual(by_key[BENCHMARK_GDPVAL]["missing"], [])

    def test_an_unmeasured_model_is_missing_not_shown_with_a_predecessors_score(self):
        guide = build_model_guide(["anthropic/claude-fable-5-2", "anthropic/claude-opus-5-5"])
        by_key = {b["key"]: b for b in guide["benchmarks"]}
        arena = by_key[BENCHMARK_ARENA_BUSINESS]
        self.assertEqual({r["model_id"] for r in arena["rows"]}, {"anthropic/claude-opus-5-5"})
        self.assertEqual(arena["missing"], ["anthropic/claude-fable-5-2"])
        for bench in guide["benchmarks"]:
            for r in bench["rows"]:
                self.assertNotIn("inherited_from", r)

    def test_models_carry_picker_price_and_star_levels(self):
        guide = build_model_guide(self.ALLOWED)
        opus = guide["models"][0]
        self.assertEqual(opus["display_name"], "Claude Opus 5.5")
        self.assertEqual(opus["price_level"], 4)
        self.assertEqual(opus["capability_level"], 4)

    def test_models_never_carry_the_blended_price(self):
        # The blended price only sets the rating; users see the rating and
        # the real list prices, never the blend.
        from llm.display import _blended_price

        guide = build_model_guide(self.ALLOWED)
        for m in guide["models"]:
            with self.subTest(model=m["id"]):
                blended = str(_blended_price(m["id"]))
                self.assertNotIn(blended, json.dumps(m))
                self.assertFalse(any("blend" in k for k in m))

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


class NoStandInTests(SimpleTestCase):
    """Every row was measured on the model it is shown for."""

    def test_no_benchmark_holds_a_retired_models_rows(self):
        registered = set(get_registered_model_ids())
        for key, bench in BENCHMARKS.items():
            with self.subTest(bench=key):
                self.assertLessEqual({s.model_id for s in bench.scores}, registered)

    def test_no_row_is_marked_as_measured_on_another_model(self):
        for key, bench in BENCHMARKS.items():
            for s in bench.scores:
                with self.subTest(bench=key, model=s.model_id, effort=s.effort):
                    self.assertNotIn("Measured on", s.note or "")
