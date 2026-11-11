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
    BenchmarkScore,
    _PREDECESSORS,
    _with_inherited,
    build_model_guide,
    get_benchmark,
    get_scores,
    missing_models,
)
from llm.model_registry import canonical_model_id, get_model_info, get_registered_model_ids


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
        # Every model is ranked on the Arena business board (2026-11-09):
        # Fable 5.2 and GPT-6.1 Terra through their predecessor's row.
        self.assertEqual(missing_models(BENCHMARK_ARENA_BUSINESS), [])
        self.assertEqual(missing_models(BENCHMARK_GDPVAL), [])
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
        # The Gemini 4 models are not in a LiveBench release with the IF table
        # (LiveBench 2.0 replaced it) and have no predecessor to borrow from.
        self.assertEqual(
            missing_models(BENCHMARK_LIVEBENCH_IF),
            ["gemini/gemini-4-carbon", "gemini/gemini-4-argon"],
        )

    def test_gemini_4_carbon_scores(self):
        # A new line: only real Carbon numbers, never a stand-in.
        expected = {
            BENCHMARK_GDPVAL: [("high", 1733)],
            BENCHMARK_GDP_PDF: [("low", 28.6), ("medium", 29.6), ("high", 28.4)],
            BENCHMARK_OMNISCIENCE: [("low", 19.8), ("medium", 4.2), ("high", 11.4)],
            BENCHMARK_ARENA_BUSINESS: [("high", 1512)],
            BENCHMARK_LIVEBENCH_IF: [],
            BENCHMARK_RESPONSE_TIME: [],
        }
        for key, rows in expected.items():
            with self.subTest(key=key):
                got = get_scores(key, "gemini/gemini-4-carbon")
                self.assertEqual([(s.effort, s.score) for s in got], rows)
                self.assertFalse(any(s.inherited_from for s in got))

    def test_fable_5_2_scores(self):
        efforts = ["low", "medium", "high", "xhigh", "max"]
        for key in (BENCHMARK_GDPVAL, BENCHMARK_GDP_PDF, BENCHMARK_OMNISCIENCE):
            with self.subTest(key=key):
                rows = get_scores(key, "anthropic/claude-fable-5-2")
                self.assertEqual([s.effort for s in rows], efforts)
                self.assertFalse(any(s.inherited_from for s in rows))
        gdpval = {s.effort: s.score for s in get_scores(BENCHMARK_GDPVAL, "anthropic/claude-fable-5-2")}
        self.assertEqual(gdpval["max"], 1831)
        # Not measured yet on these three: Fable 5.1's rows stand in.
        for key, first in (
            (BENCHMARK_ARENA_BUSINESS, ("max", 1489)),
            (BENCHMARK_LIVEBENCH_IF, ("max", 72.99)),
            (BENCHMARK_RESPONSE_TIME, ("low", 38.2)),
        ):
            with self.subTest(key=key):
                rows = get_scores(key, "anthropic/claude-fable-5-2")
                self.assertEqual((rows[0].effort, rows[0].score), first)
                for s in rows:
                    self.assertEqual(s.inherited_from, "anthropic/claude-fable-5-1")
                    self.assertTrue(s.note.startswith(
                        "Measured on Claude Fable 5.1; no Claude Fable 5.2 score published yet."
                    ))

    def test_gemini_4_argon_scores(self):
        # AA and LMArena only run Argon at high.
        expected = {
            BENCHMARK_GDPVAL: 1633,
            BENCHMARK_GDP_PDF: 21.8,
            BENCHMARK_OMNISCIENCE: 4.3,
            BENCHMARK_ARENA_BUSINESS: 1517,
            BENCHMARK_RESPONSE_TIME: 86.1,
        }
        for key, score in expected.items():
            with self.subTest(key=key):
                rows = get_scores(key, "gemini/gemini-4-argon")
                self.assertEqual([(s.effort, s.score) for s in rows], [("high", score)])

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

    def test_hallucination_rate_is_lower_is_better(self):
        self.assertFalse(get_benchmark(BENCHMARK_OMNISCIENCE).higher_is_better)

    def test_response_time(self):
        bench = get_benchmark(BENCHMARK_RESPONSE_TIME)
        self.assertFalse(bench.higher_is_better)
        # AA had no 100K timing for nano (2026-11-03); Carbon came out after
        # the series could last be read and has no predecessor.
        self.assertEqual(
            missing_models(BENCHMARK_RESPONSE_TIME),
            ["openai/gpt-5.4-nano", "gemini/gemini-4-carbon"],
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

    def test_stand_in_rows_are_marked_for_the_guide(self):
        guide = build_model_guide(["anthropic/claude-fable-5-2", "anthropic/claude-opus-5-5"])
        by_key = {b["key"]: b for b in guide["benchmarks"]}
        arena = {r["model_id"]: r for r in by_key[BENCHMARK_ARENA_BUSINESS]["rows"]}
        fable = arena["anthropic/claude-fable-5-2"]
        self.assertEqual(fable["inherited_from"], "anthropic/claude-fable-5-1")
        self.assertIn("Measured on Claude Fable 5.1", fable["note"])
        self.assertEqual(fable["source_label"], "claude-fable-5.1-max")
        self.assertIsNone(arena["anthropic/claude-opus-5-5"]["inherited_from"])
        # A real score is never marked.
        for r in by_key[BENCHMARK_GDPVAL]["rows"]:
            self.assertIsNone(r["inherited_from"])


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


class PredecessorStandInTests(SimpleTestCase):
    """A new model with no score borrows its retired predecessor's rows."""

    NEW = "anthropic/claude-sonnet-5-5"  # offers low..max, no "off"
    OLD = "anthropic/claude-sonnet-5"
    PAIRS = {NEW: (OLD, "Claude Sonnet 5")}

    def _rows(self, *rows):
        return tuple(BenchmarkScore(m, e, s, f"label-{e}", note=n) for m, e, s, n in rows)

    def test_predecessor_rows_stand_in_when_the_successor_has_none(self):
        scores = self._rows(
            (self.OLD, "off", 10, None), (self.OLD, "high", 20, None),
            (self.OLD, None, 30, "Only 5 votes."),
            ("anthropic/claude-opus-5-5", "high", 40, None),
        )
        result = _with_inherited(scores, self.PAIRS)
        self.assertEqual(
            [(s.model_id, s.effort, s.score, s.inherited_from) for s in result],
            [
                # "off" is dropped: Sonnet 5.5 does not offer it.
                (self.NEW, "high", 20, self.OLD),
                (self.NEW, None, 30, self.OLD),
                ("anthropic/claude-opus-5-5", "high", 40, None),
            ],
        )
        marker = "Measured on Claude Sonnet 5; no Claude Sonnet 5.5 score published yet."
        self.assertEqual(result[0].note, marker)
        self.assertEqual(result[1].note, f"{marker} Only 5 votes.")
        self.assertEqual(result[0].source_label, "label-high")

    def test_a_real_score_ends_the_stand_in(self):
        scores = self._rows((self.OLD, "high", 20, None), (self.NEW, "max", 99, None))
        result = _with_inherited(scores, self.PAIRS)
        self.assertEqual(
            [(s.model_id, s.effort, s.score, s.inherited_from) for s in result],
            [(self.NEW, "max", 99, None)],
        )

    def test_no_benchmark_shows_a_retired_model(self):
        registered = set(get_registered_model_ids())
        for key, bench in BENCHMARKS.items():
            with self.subTest(bench=key):
                self.assertLessEqual({s.model_id for s in bench.scores}, registered)

    def test_every_pair_is_a_live_successor_of_a_retired_model_and_still_needed(self):
        registered = set(get_registered_model_ids())
        borrowed = {
            s.inherited_from for b in BENCHMARKS.values() for s in b.scores if s.inherited_from
        }
        for new, (old, old_name) in _PREDECESSORS.items():
            with self.subTest(successor=new):
                self.assertIn(new, registered)
                self.assertNotIn(old, registered)
                self.assertEqual(canonical_model_id(old), new)
                self.assertTrue(old_name)
                # Delete the pair (and the predecessor's rows) once every
                # benchmark has a real score for the successor.
                self.assertIn(old, borrowed)
