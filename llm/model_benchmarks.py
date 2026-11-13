"""Third-party benchmark scores for the models in the registry.

Feeds the "What model should I pick?" guide. One score per (model, effort)
where the source publishes one — a model's effort variants are separate rows.

Rules for adding a score: it must appear on the benchmark's ``source_url``
(no estimates, no reading values off a different leaderboard); keep the
source's own name for the variant in ``source_label``. ``effort`` is our
reasoning level (see ``ModelInfo.reasoning_levels``; "none"/"off" = thinking
disabled) or None when the source doesn't state it. A registry model with no
row is reported by ``missing_models``.

Our practice when the registry changes (do this EVERY time a model is added or
retired, see ``MODEL_REPLACEMENTS``):

1. Check every benchmark's source for the new model's real scores and add them.
2. Where a source has not measured it yet and the model replaces a retired
   generation of the same line (Fable 5.1 -> 5.2), it borrows that
   predecessor's rows so it is still ranked: add the pair to
   ``_PREDECESSORS`` and keep the predecessor's rows on that benchmark. The
   guide marks such a row ("Measured on ...; no ... score published yet").
   The stand-in ends by itself the day a real row is added for the benchmark.
3. A model with no predecessor (a new line, e.g. Gemini 4 Carbon) stays
   unranked on the benchmarks that have not measured it.
4. Re-check the sources for every model still on a stand-in, and delete the
   predecessor's rows (and the ``_PREDECESSORS`` pair) once nothing borrows
   them. Rows of other retired models are deleted outright.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from llm.model_registry import canonical_model_id, get_model_info, get_registered_model_ids

BENCHMARK_GDPVAL = "gdpval_aa"
BENCHMARK_ARENA_BUSINESS = "arena_business"
BENCHMARK_GDP_PDF = "gdp_pdf"
BENCHMARK_LIVEBENCH_IF = "livebench_if"
BENCHMARK_OMNISCIENCE = "omniscience_hallucination"
BENCHMARK_RESPONSE_TIME = "response_time_100k"


@dataclass(frozen=True)
class BenchmarkScore:
    model_id: str
    effort: str | None
    score: float
    source_label: str
    # Half-width of the 95% CI, when the source publishes one.
    ci: float | None = None
    # True when the model's own vendor produced the number (incl. a
    # vendor-run leaderboard scoring its own model).
    self_reported: bool = False
    note: str | None = None
    # Set on a stand-in row: the retired predecessor the score was measured on
    # (see the module docstring and _PREDECESSORS).
    inherited_from: str | None = None


@dataclass(frozen=True)
class Benchmark:
    key: str
    name: str
    # Plain-language heading and the question the benchmark answers, for
    # users who don't know the benchmark by name.
    title: str
    question: str
    metric: str
    description: str
    source_url: str
    # Date the source's data is current to (the page's own date when it
    # states one, otherwise the day we fetched it).
    as_of: date
    scores: tuple[BenchmarkScore, ...]
    higher_is_better: bool = True
    caveat: str | None = None
    # How the guide formats and scales scores: "elo" (relative, no zero),
    # "percent" (0-100, shown with %), "score" (0-100, shown bare),
    # "seconds" (a duration, lower is better; bars relative to the fastest).
    scale: str = "elo"


def _row(model_id, effort, score, source_label, ci=None, note=None):
    return BenchmarkScore(model_id, effort, score, source_label, ci=ci, note=note)


# Artificial Analysis rows are parsed from the model data embedded in
# artificialanalysis.ai pages (the model object's gdpval, gdpPdfAllPass and
# omniscienceHallucinationRate); source_label is AA's slug. Every AA row of
# the three benchmarks below was re-read from one snapshot of each slug's
# /models/<slug> page on 2026-11-13, when Gemini 4 Hydrogen was added. A
# metric the page no longer shows keeps its earlier value (Omniscience:
# GPT-5.4 nano, Gemini 3.1 Pro). GDPval Elo is relative and drifts as AA adds
# models, so all GDPval rows are re-read together from one snapshot whenever a
# model is added. Unsuffixed OpenAI/Anthropic slugs are the max-effort runs;
# unsuffixed Gemini slugs are "high", except gemini-4-hydrogen, which is the
# xhigh run (a level we do not offer yet, so it has no row). AA runs Gemini 4
# Argon at medium and high, and has a GDPval score for Carbon at high only and
# none for Hydrogen yet (it borrows Gemini 3.8 Flash's, see _PREDECESSORS).
_AA_SCORES = {
    BENCHMARK_GDPVAL: (
        _row("openai/gpt-6-astra", "low", 1403, "gpt-6-astra-low"),
        _row("openai/gpt-6-astra", "medium", 1498, "gpt-6-astra-medium"),
        _row("openai/gpt-6-astra", "high", 1536, "gpt-6-astra-high"),
        _row("openai/gpt-6-astra", "xhigh", 1569, "gpt-6-astra-xhigh"),
        _row("openai/gpt-6-astra", "max", 1586, "gpt-6-astra"),
        _row("openai/gpt-6.1-sol", "low", 1322, "gpt-6-1-sol-low"),
        _row("openai/gpt-6.1-sol", "medium", 1459, "gpt-6-1-sol-medium"),
        _row("openai/gpt-6.1-sol", "high", 1520, "gpt-6-1-sol-high"),
        _row("openai/gpt-6.1-sol", "xhigh", 1555, "gpt-6-1-sol-xhigh"),
        _row("openai/gpt-6.1-sol", "max", 1608, "gpt-6-1-sol"),
        _row("openai/gpt-6.1-terra", "none", 1286, "gpt-6-1-terra-non-reasoning"),
        _row("openai/gpt-6.1-terra", "low", 1204, "gpt-6-1-terra-low"),
        _row("openai/gpt-6.1-terra", "medium", 1325, "gpt-6-1-terra-medium"),
        _row("openai/gpt-6.1-terra", "high", 1353, "gpt-6-1-terra-high"),
        _row("openai/gpt-6.1-terra", "xhigh", 1415, "gpt-6-1-terra-xhigh"),
        _row("openai/gpt-6.1-terra", "max", 1476, "gpt-6-1-terra"),
        _row("openai/gpt-6-luna", "none", 1052, "gpt-6-luna-non-reasoning"),
        _row("openai/gpt-6-luna", "low", 1040, "gpt-6-luna-low"),
        _row("openai/gpt-6-luna", "medium", 1270, "gpt-6-luna-medium"),
        _row("openai/gpt-6-luna", "high", 1352, "gpt-6-luna-high"),
        _row("openai/gpt-6-luna", "xhigh", 1373, "gpt-6-luna-xhigh"),
        _row("openai/gpt-6-luna", "max", 1444, "gpt-6-luna"),
        _row("openai/gpt-5.4-nano", "xhigh", 955, "gpt-5-4-nano"),
        _row("anthropic/claude-fable-5-2", "low", 1546, "claude-fable-5-2-low"),
        _row("anthropic/claude-fable-5-2", "medium", 1591, "claude-fable-5-2-medium"),
        _row("anthropic/claude-fable-5-2", "high", 1718, "claude-fable-5-2-high"),
        _row("anthropic/claude-fable-5-2", "xhigh", 1751, "claude-fable-5-2-xhigh"),
        _row("anthropic/claude-fable-5-2", "max", 1831, "claude-fable-5-2"),
        _row("anthropic/claude-opus-5-5", "low", 1240, "claude-opus-5-5-low"),
        _row("anthropic/claude-opus-5-5", "medium", 1593, "claude-opus-5-5-medium"),
        _row("anthropic/claude-opus-5-5", "high", 1714, "claude-opus-5-5-high"),
        _row("anthropic/claude-opus-5-5", "xhigh", 1835, "claude-opus-5-5-xhigh"),
        _row("anthropic/claude-opus-5-5", "max", 1866, "claude-opus-5-5"),
        _row("anthropic/claude-sonnet-5-5", "low", 1182, "claude-sonnet-5-5-low"),
        _row("anthropic/claude-sonnet-5-5", "medium", 1329, "claude-sonnet-5-5-medium"),
        _row("anthropic/claude-sonnet-5-5", "high", 1557, "claude-sonnet-5-5-high"),
        _row("anthropic/claude-sonnet-5-5", "xhigh", 1735, "claude-sonnet-5-5-xhigh"),
        _row("anthropic/claude-sonnet-5-5", "max", 1838, "claude-sonnet-5-5"),
        _row("anthropic/claude-haiku-5-5", "low", 1130, "claude-haiku-5-5-low"),
        _row("anthropic/claude-haiku-5-5", "medium", 1282, "claude-haiku-5-5-medium"),
        _row("anthropic/claude-haiku-5-5", "high", 1424, "claude-haiku-5-5-high"),
        _row("anthropic/claude-haiku-5-5", "xhigh", 1522, "claude-haiku-5-5-xhigh"),
        _row("anthropic/claude-haiku-5-5", "max", 1629, "claude-haiku-5-5"),
        _row("gemini/gemini-4-carbon", "high", 1733, "gemini-4-carbon"),
        _row("gemini/gemini-4-argon", "medium", 1578, "gemini-4-argon-medium"),
        _row("gemini/gemini-4-argon", "high", 1633, "gemini-4-argon"),
        _row("gemini/gemini-3.1-pro-preview", None, 799, "gemini-3-1-pro-preview"),
        # Retired; stands in for Gemini 4 Hydrogen (see _PREDECESSORS).
        _row("gemini/gemini-3.8-flash", "low", 1316, "gemini-3-8-flash-low"),
        _row("gemini/gemini-3.8-flash", "medium", 1431, "gemini-3-8-flash-medium"),
        _row("gemini/gemini-3.8-flash", "high", 1445, "gemini-3-8-flash"),
        _row("gemini/gemini-3.5-flash-lite", None, 995, "gemini-3-5-flash-lite"),
    ),
    BENCHMARK_GDP_PDF: (
        _row("openai/gpt-6-astra", "low", 30.4, "gpt-6-astra-low"),
        _row("openai/gpt-6-astra", "medium", 30.4, "gpt-6-astra-medium"),
        _row("openai/gpt-6-astra", "high", 31.0, "gpt-6-astra-high"),
        _row("openai/gpt-6-astra", "xhigh", 32.2, "gpt-6-astra-xhigh"),
        _row("openai/gpt-6-astra", "max", 31.0, "gpt-6-astra"),
        _row("openai/gpt-6.1-sol", "low", 27.0, "gpt-6-1-sol-low"),
        _row("openai/gpt-6.1-sol", "medium", 30.0, "gpt-6-1-sol-medium"),
        _row("openai/gpt-6.1-sol", "high", 32.0, "gpt-6-1-sol-high"),
        _row("openai/gpt-6.1-sol", "xhigh", 31.8, "gpt-6-1-sol-xhigh"),
        _row("openai/gpt-6.1-sol", "max", 31.0, "gpt-6-1-sol"),
        _row("openai/gpt-6.1-terra", "none", 23.2, "gpt-6-1-terra-non-reasoning"),
        _row("openai/gpt-6.1-terra", "low", 25.6, "gpt-6-1-terra-low"),
        _row("openai/gpt-6.1-terra", "medium", 28.4, "gpt-6-1-terra-medium"),
        _row("openai/gpt-6.1-terra", "high", 28.0, "gpt-6-1-terra-high"),
        _row("openai/gpt-6.1-terra", "xhigh", 26.6, "gpt-6-1-terra-xhigh"),
        _row("openai/gpt-6.1-terra", "max", 28.6, "gpt-6-1-terra"),
        _row("openai/gpt-6-luna", "none", 5.8, "gpt-6-luna-non-reasoning"),
        _row("openai/gpt-6-luna", "low", 9.0, "gpt-6-luna-low"),
        _row("openai/gpt-6-luna", "medium", 13.8, "gpt-6-luna-medium"),
        _row("openai/gpt-6-luna", "high", 18.2, "gpt-6-luna-high"),
        _row("openai/gpt-6-luna", "xhigh", 18.6, "gpt-6-luna-xhigh"),
        _row("openai/gpt-6-luna", "max", 22.8, "gpt-6-luna"),
        _row("openai/gpt-5.4-nano", "xhigh", 7.8, "gpt-5-4-nano"),
        _row("anthropic/claude-fable-5-2", "low", 26.8, "claude-fable-5-2-low"),
        _row("anthropic/claude-fable-5-2", "medium", 23.8, "claude-fable-5-2-medium"),
        _row("anthropic/claude-fable-5-2", "high", 21.8, "claude-fable-5-2-high"),
        _row("anthropic/claude-fable-5-2", "xhigh", 21.4, "claude-fable-5-2-xhigh"),
        _row("anthropic/claude-fable-5-2", "max", 24.2, "claude-fable-5-2"),
        _row("anthropic/claude-opus-5-5", "low", 25.6, "claude-opus-5-5-low"),
        _row("anthropic/claude-opus-5-5", "medium", 25.6, "claude-opus-5-5-medium"),
        _row("anthropic/claude-opus-5-5", "high", 28.8, "claude-opus-5-5-high"),
        _row("anthropic/claude-opus-5-5", "xhigh", 26.6, "claude-opus-5-5-xhigh"),
        _row("anthropic/claude-opus-5-5", "max", 26.2, "claude-opus-5-5"),
        _row("anthropic/claude-sonnet-5-5", "low", 16.0, "claude-sonnet-5-5-low"),
        _row("anthropic/claude-sonnet-5-5", "medium", 20.2, "claude-sonnet-5-5-medium"),
        _row("anthropic/claude-sonnet-5-5", "high", 25.2, "claude-sonnet-5-5-high"),
        _row("anthropic/claude-sonnet-5-5", "xhigh", 24.6, "claude-sonnet-5-5-xhigh"),
        _row("anthropic/claude-sonnet-5-5", "max", 25.8, "claude-sonnet-5-5"),
        _row("anthropic/claude-haiku-5-5", "low", 11.2, "claude-haiku-5-5-low"),
        _row("anthropic/claude-haiku-5-5", "medium", 15.2, "claude-haiku-5-5-medium"),
        _row("anthropic/claude-haiku-5-5", "high", 17.2, "claude-haiku-5-5-high"),
        _row("anthropic/claude-haiku-5-5", "xhigh", 18.2, "claude-haiku-5-5-xhigh"),
        _row("anthropic/claude-haiku-5-5", "max", 20.8, "claude-haiku-5-5"),
        _row("gemini/gemini-4-carbon", "low", 28.6, "gemini-4-carbon-low"),
        _row("gemini/gemini-4-carbon", "medium", 29.6, "gemini-4-carbon-medium"),
        _row("gemini/gemini-4-carbon", "high", 28.4, "gemini-4-carbon"),
        _row("gemini/gemini-4-argon", "medium", 25.6, "gemini-4-argon-medium"),
        _row("gemini/gemini-4-argon", "high", 21.8, "gemini-4-argon"),
        _row("gemini/gemini-3.1-pro-preview", None, 17.8, "gemini-3-1-pro-preview"),
        _row("gemini/gemini-4-hydrogen", "low", 28.2, "gemini-4-hydrogen-low"),
        _row("gemini/gemini-4-hydrogen", "medium", 32.8, "gemini-4-hydrogen-medium"),
        _row("gemini/gemini-4-hydrogen", "high", 30.0, "gemini-4-hydrogen-high"),
        _row("gemini/gemini-3.5-flash-lite", None, 13.6, "gemini-3-5-flash-lite"),
    ),
    BENCHMARK_OMNISCIENCE: (
        _row("openai/gpt-6-astra", "low", 44.2, "gpt-6-astra-low"),
        _row("openai/gpt-6-astra", "medium", 45.2, "gpt-6-astra-medium"),
        _row("openai/gpt-6-astra", "high", 42.5, "gpt-6-astra-high"),
        _row("openai/gpt-6-astra", "xhigh", 44.6, "gpt-6-astra-xhigh"),
        _row("openai/gpt-6-astra", "max", 49.8, "gpt-6-astra"),
        _row("openai/gpt-6.1-sol", "low", 49.9, "gpt-6-1-sol-low"),
        _row("openai/gpt-6.1-sol", "medium", 48.1, "gpt-6-1-sol-medium"),
        _row("openai/gpt-6.1-sol", "high", 49.0, "gpt-6-1-sol-high"),
        _row("openai/gpt-6.1-sol", "xhigh", 48.0, "gpt-6-1-sol-xhigh"),
        _row("openai/gpt-6.1-sol", "max", 51.9, "gpt-6-1-sol"),
        _row("openai/gpt-6.1-terra", "none", 77.1, "gpt-6-1-terra-non-reasoning"),
        _row("openai/gpt-6.1-terra", "low", 41.6, "gpt-6-1-terra-low"),
        _row("openai/gpt-6.1-terra", "medium", 39.6, "gpt-6-1-terra-medium"),
        _row("openai/gpt-6.1-terra", "high", 37.0, "gpt-6-1-terra-high"),
        _row("openai/gpt-6.1-terra", "xhigh", 36.8, "gpt-6-1-terra-xhigh"),
        _row("openai/gpt-6.1-terra", "max", 34.5, "gpt-6-1-terra"),
        _row("openai/gpt-6-luna", "none", 80.0, "gpt-6-luna-non-reasoning"),
        _row("openai/gpt-6-luna", "low", 84.8, "gpt-6-luna-low"),
        _row("openai/gpt-6-luna", "medium", 85.3, "gpt-6-luna-medium"),
        _row("openai/gpt-6-luna", "high", 84.9, "gpt-6-luna-high"),
        _row("openai/gpt-6-luna", "xhigh", 84.0, "gpt-6-luna-xhigh"),
        _row("openai/gpt-6-luna", "max", 79.3, "gpt-6-luna"),
        _row("openai/gpt-5.4-nano", "none", 61.5, "gpt-5-4-nano-non-reasoning"),
        _row("openai/gpt-5.4-nano", "medium", 51.1, "gpt-5-4-nano-medium"),
        _row("openai/gpt-5.4-nano", "xhigh", 74.2, "gpt-5-4-nano"),
        _row("anthropic/claude-fable-5-2", "low", 56.4, "claude-fable-5-2-low"),
        _row("anthropic/claude-fable-5-2", "medium", 58.5, "claude-fable-5-2-medium"),
        _row("anthropic/claude-fable-5-2", "high", 58.0, "claude-fable-5-2-high"),
        _row("anthropic/claude-fable-5-2", "xhigh", 60.4, "claude-fable-5-2-xhigh"),
        _row("anthropic/claude-fable-5-2", "max", 59.3, "claude-fable-5-2"),
        _row("anthropic/claude-opus-5-5", "low", 66.3, "claude-opus-5-5-low"),
        _row("anthropic/claude-opus-5-5", "medium", 66.9, "claude-opus-5-5-medium"),
        _row("anthropic/claude-opus-5-5", "high", 63.9, "claude-opus-5-5-high"),
        _row("anthropic/claude-opus-5-5", "xhigh", 65.0, "claude-opus-5-5-xhigh"),
        _row("anthropic/claude-opus-5-5", "max", 50.3, "claude-opus-5-5"),
        _row("anthropic/claude-sonnet-5-5", "low", 47.7, "claude-sonnet-5-5-low"),
        _row("anthropic/claude-sonnet-5-5", "medium", 50.0, "claude-sonnet-5-5-medium"),
        _row("anthropic/claude-sonnet-5-5", "high", 63.6, "claude-sonnet-5-5-high"),
        _row("anthropic/claude-sonnet-5-5", "xhigh", 61.7, "claude-sonnet-5-5-xhigh"),
        _row("anthropic/claude-sonnet-5-5", "max", 43.3, "claude-sonnet-5-5"),
        _row("anthropic/claude-haiku-5-5", "low", 40.5, "claude-haiku-5-5-low"),
        _row("anthropic/claude-haiku-5-5", "medium", 40.4, "claude-haiku-5-5-medium"),
        _row("anthropic/claude-haiku-5-5", "high", 38.9, "claude-haiku-5-5-high"),
        _row("anthropic/claude-haiku-5-5", "xhigh", 39.2, "claude-haiku-5-5-xhigh"),
        _row("anthropic/claude-haiku-5-5", "max", 34.2, "claude-haiku-5-5"),
        _row("gemini/gemini-4-carbon", "low", 19.8, "gemini-4-carbon-low"),
        _row("gemini/gemini-4-carbon", "medium", 4.2, "gemini-4-carbon-medium"),
        _row("gemini/gemini-4-carbon", "high", 11.4, "gemini-4-carbon"),
        _row("gemini/gemini-4-argon", "medium", 15.4, "gemini-4-argon-medium"),
        _row("gemini/gemini-4-argon", "high", 4.3, "gemini-4-argon"),
        _row("gemini/gemini-3.1-pro-preview", None, 50.9, "gemini-3-1-pro-preview"),
        _row("gemini/gemini-4-hydrogen", "low", 18.8, "gemini-4-hydrogen-low"),
        _row("gemini/gemini-4-hydrogen", "medium", 12.5, "gemini-4-hydrogen-medium"),
        _row("gemini/gemini-4-hydrogen", "high", 8.7, "gemini-4-hydrogen-high"),
        _row("gemini/gemini-3.5-flash-lite", None, 26.7, "gemini-3-5-flash-lite"),
    ),
}

# LMArena rows are all read from one snapshot of the board (2026-11-11; the
# page shows whole points). It could only be read down to rank 149: Fable 5.2
# and GPT-6.1 Terra are not in that part (not rated yet), and GPT-5.4 nano
# ranks below it, so its row keeps the 2026-09-25 value.
_ARENA_SCORES = (
    _row("openai/gpt-6-astra", "max", 1478, "gpt-6-astra-max", ci=11),
    _row("openai/gpt-6.1-sol", "max", 1480, "gpt-6.1-sol-max", ci=14),
    _row("openai/gpt-6-luna", "max", 1439, "gpt-6-luna-max", ci=6),
    # Retired; stands in for GPT-6.1 Terra (see _PREDECESSORS).
    _row("openai/gpt-5.6-terra", "xhigh", 1466, "gpt-5.6-terra-xhigh", ci=7),
    _row("openai/gpt-5.4-nano", "high", 1405.1, "gpt-5.4-nano-high", ci=6.6),
    # Retired; stands in for Fable 5.2 (see _PREDECESSORS).
    _row("anthropic/claude-fable-5-1", "max", 1488, "claude-fable-5.1-max", ci=10),
    _row("anthropic/claude-opus-5-5", "high", 1493, "claude-opus-5.5-high", ci=12),
    _row("anthropic/claude-sonnet-5-5", "xhigh", 1474, "claude-sonnet-5.5-xhigh", ci=13),
    _row("anthropic/claude-haiku-5-5", "high", 1430, "claude-haiku-5.5-high", ci=13),
    _row(
        "gemini/gemini-4-carbon", "high", 1513, "gemini-4-carbon-high", ci=22,
        note="Preliminary: only 715 votes so far.",
    ),
    _row("gemini/gemini-4-argon", "high", 1515, "gemini-4-argon-high", ci=11),
    _row("gemini/gemini-3.1-pro-preview", None, 1478, "gemini-3.1-pro-preview", ci=5),
    _row(
        "gemini/gemini-4-hydrogen", "high", 1476, "gemini-4-hydrogen-high", ci=21,
        note="Preliminary: only 791 votes so far.",
    ),
    _row("gemini/gemini-3.5-flash-lite", None, 1453, "gemini-3.5-flash-lite", ci=8),
)

# LiveBench release 2026-06-25 (livebench.ai/table_2026_06_25.csv): the IF
# category is the mean of paraphrase, simplify, story_generation and summarize.
# All rows share one question set. LiveBench 2.0 has since replaced the IF
# table, so models released after it (GPT-6.1 Terra, Fable 5.2, Gemini 4) have
# no row; those with a retired predecessor (Terra, Fable 5.2, Gemini 4
# Hydrogen) borrow its row (_PREDECESSORS).
_LIVEBENCH_IF_SCORES = (
    _row("openai/gpt-6-astra", "max", 75.58, "gpt-6-astra-max"),
    _row("openai/gpt-6.1-sol", "max", 74.15, "gpt-6.1-sol-max"),
    _row("openai/gpt-6.1-sol", "xhigh", 71.33, "gpt-6.1-sol-xhigh"),
    _row("openai/gpt-6-luna", "max", 55.93, "gpt-6-luna-max"),
    _row("openai/gpt-5.6-terra", "max", 64.62, "gpt-5.6-terra-max"),
    _row("openai/gpt-5.4-nano", "xhigh", 67.20, "gpt-5.4-nano-xhigh"),
    _row("anthropic/claude-fable-5-1", "max", 72.99, "claude-fable-5-1-max-effort"),
    _row("anthropic/claude-opus-5-5", "xhigh", 67.05, "claude-opus-5-5-xhigh-effort"),
    _row("anthropic/claude-opus-5-5", "max", 65.74, "claude-opus-5-5-max-effort"),
    _row("anthropic/claude-sonnet-5-5", "xhigh", 70.52, "claude-sonnet-5-5-xhigh-effort"),
    _row("anthropic/claude-sonnet-5-5", "max", 56.79, "claude-sonnet-5-5-max-effort"),
    _row("anthropic/claude-haiku-5-5", "xhigh", 66.47, "claude-haiku-5-5-xhigh-effort"),
    _row("anthropic/claude-haiku-5-5", "max", 54.66, "claude-haiku-5-5-max-effort"),
    _row("gemini/gemini-3.1-pro-preview", "high", 79.10, "gemini-3.1-pro-preview-high"),
    # Retired; stands in for Gemini 4 Hydrogen (see _PREDECESSORS).
    _row("gemini/gemini-3.8-flash", "high", 81.41, "gemini-3.8-flash-high"),
    _row("gemini/gemini-3.5-flash-lite", "high", 67.24, "gemini-3.5-flash-lite-high"),
)

# Artificial Analysis end-to-end response time (seconds to receive a complete
# 500-token answer, thinking included) for its 100K-token prompt: the median in
# the "by prompt type" data on each artificialanalysis.ai/models/<slug> page,
# read 2026-11-03. Effort variants AA has no 100K measurement for are left out
# (Gemini 3.8 Flash low/medium); GPT-5.4 nano has none. On 2026-11-11 the pages
# no longer carried that series in a form we could read (only time to first
# token per prompt size; still so on 2026-11-13), so Fable 5.2 borrows Fable
# 5.1's rows, Gemini 4 Hydrogen borrows Gemini 3.8 Flash's and Gemini 4 Carbon
# has none. Re-read all rows together when the series is back.
_RESPONSE_TIME_SCORES = (
    _row("openai/gpt-6-astra", "low", 12.2, "gpt-6-astra-low"),
    _row("openai/gpt-6-astra", "medium", 26.0, "gpt-6-astra-medium"),
    _row("openai/gpt-6-astra", "high", 146.9, "gpt-6-astra-high"),
    _row("openai/gpt-6-astra", "xhigh", 230.4, "gpt-6-astra-xhigh"),
    _row("openai/gpt-6-astra", "max", 171.8, "gpt-6-astra"),
    _row("openai/gpt-6.1-sol", "low", 10.2, "gpt-6-1-sol-low"),
    _row("openai/gpt-6.1-sol", "medium", 27.9, "gpt-6-1-sol-medium"),
    _row("openai/gpt-6.1-sol", "high", 116.6, "gpt-6-1-sol-high"),
    _row("openai/gpt-6.1-sol", "xhigh", 233.0, "gpt-6-1-sol-xhigh"),
    _row("openai/gpt-6.1-sol", "max", 225.5, "gpt-6-1-sol"),
    _row("openai/gpt-6.1-terra", "none", 6.9, "gpt-6-1-terra-non-reasoning"),
    _row("openai/gpt-6.1-terra", "low", 7.4, "gpt-6-1-terra-low"),
    _row("openai/gpt-6.1-terra", "medium", 9.8, "gpt-6-1-terra-medium"),
    _row("openai/gpt-6.1-terra", "high", 10.7, "gpt-6-1-terra-high"),
    _row("openai/gpt-6.1-terra", "xhigh", 35.3, "gpt-6-1-terra-xhigh"),
    _row("openai/gpt-6.1-terra", "max", 218.7, "gpt-6-1-terra"),
    _row("openai/gpt-6-luna", "none", 5.2, "gpt-6-luna-non-reasoning"),
    _row("openai/gpt-6-luna", "low", 7.6, "gpt-6-luna-low"),
    _row("openai/gpt-6-luna", "medium", 16.4, "gpt-6-luna-medium"),
    _row("openai/gpt-6-luna", "high", 29.4, "gpt-6-luna-high"),
    _row("openai/gpt-6-luna", "xhigh", 80.8, "gpt-6-luna-xhigh"),
    _row("openai/gpt-6-luna", "max", 179.0, "gpt-6-luna"),
    _row("anthropic/claude-fable-5-1", "low", 38.2, "claude-fable-5-1-low"),
    _row("anthropic/claude-fable-5-1", "medium", 38.5, "claude-fable-5-1-medium"),
    _row("anthropic/claude-fable-5-1", "high", 57.5, "claude-fable-5-1-high"),
    _row("anthropic/claude-fable-5-1", "xhigh", 189.7, "claude-fable-5-1-xhigh"),
    _row("anthropic/claude-fable-5-1", "max", 127.8, "claude-fable-5-1"),
    _row("anthropic/claude-opus-5-5", "low", 24.9, "claude-opus-5-5-low"),
    _row("anthropic/claude-opus-5-5", "medium", 100.6, "claude-opus-5-5-medium"),
    _row("anthropic/claude-opus-5-5", "high", 114.2, "claude-opus-5-5-high"),
    _row("anthropic/claude-opus-5-5", "xhigh", 48.1, "claude-opus-5-5-xhigh"),
    _row("anthropic/claude-opus-5-5", "max", 485.7, "claude-opus-5-5"),
    _row("anthropic/claude-sonnet-5-5", "low", 25.7, "claude-sonnet-5-5-low"),
    _row("anthropic/claude-sonnet-5-5", "medium", 35.3, "claude-sonnet-5-5-medium"),
    _row("anthropic/claude-sonnet-5-5", "high", 65.1, "claude-sonnet-5-5-high"),
    _row("anthropic/claude-sonnet-5-5", "xhigh", 183.2, "claude-sonnet-5-5-xhigh"),
    _row("anthropic/claude-sonnet-5-5", "max", 649.6, "claude-sonnet-5-5"),
    _row("anthropic/claude-haiku-5-5", "low", 30.2, "claude-haiku-5-5-low"),
    _row("anthropic/claude-haiku-5-5", "medium", 47.6, "claude-haiku-5-5-medium"),
    _row("anthropic/claude-haiku-5-5", "high", 78.0, "claude-haiku-5-5-high"),
    _row("anthropic/claude-haiku-5-5", "xhigh", 89.6, "claude-haiku-5-5-xhigh"),
    _row("anthropic/claude-haiku-5-5", "max", 371.6, "claude-haiku-5-5"),
    _row("gemini/gemini-4-argon", "high", 86.1, "gemini-4-argon"),
    _row("gemini/gemini-3.1-pro-preview", None, 69.6, "gemini-3-1-pro-preview"),
    # Retired; stands in for Gemini 4 Hydrogen (see _PREDECESSORS).
    _row("gemini/gemini-3.8-flash", "high", 76.8, "gemini-3-8-flash"),
    _row("gemini/gemini-3.5-flash-lite", None, 14.3, "gemini-3-5-flash-lite"),
)

_FETCHED = date(2026, 11, 13)
_RESPONSE_TIME_FETCHED = date(2026, 11, 3)

# Successor -> (retired predecessor, its display name). On a benchmark where
# the successor has no row of its own, the predecessor's rows stand in for it
# (module docstring, step 2). Remove a pair once no benchmark needs it.
_PREDECESSORS: dict[str, tuple[str, str]] = {
    "anthropic/claude-fable-5-2": ("anthropic/claude-fable-5-1", "Claude Fable 5.1"),
    "openai/gpt-6.1-terra": ("openai/gpt-5.6-terra", "GPT-5.6 Terra"),
    "gemini/gemini-4-hydrogen": ("gemini/gemini-3.8-flash", "Gemini 3.8 Flash"),
}


def _with_inherited(
    scores: tuple[BenchmarkScore, ...],
    predecessors: dict[str, tuple[str, str]] | None = None,
) -> tuple[BenchmarkScore, ...]:
    """One benchmark's rows as the guide sees them: predecessor rows are
    re-labelled as stand-ins for a successor with no row of its own, and
    dropped otherwise (a retired model is never shown itself)."""
    predecessors = _PREDECESSORS if predecessors is None else predecessors
    measured = {s.model_id for s in scores}
    stand_in_for = {
        old: (new, old_name)
        for new, (old, old_name) in predecessors.items()
        if new not in measured
    }
    retired = {old for old, _ in predecessors.values()}
    result = []
    for s in scores:
        if s.model_id not in retired:
            result.append(s)
            continue
        if s.model_id not in stand_in_for:
            continue
        new, old_name = stand_in_for[s.model_id]
        info = get_model_info(new)
        if info is None:
            continue
        # A level the successor does not offer says nothing about it.
        if s.effort is not None and s.effort not in info.reasoning_levels:
            continue
        note = f"Measured on {old_name}; no {info.display_name} score published yet."
        result.append(replace(
            s, model_id=new, inherited_from=s.model_id,
            note=f"{note} {s.note}" if s.note else note,
        ))
    return tuple(result)


BENCHMARKS: dict[str, Benchmark] = {
    b.key: b for b in (
        Benchmark(
            key=BENCHMARK_GDPVAL,
            name="GDPval-AA",
            title="General office work",
            question=(
                "How good is the model at real knowledge work, such as reports, "
                "memos and analyses, as judged by people?"
            ),
            metric="Elo",
            description=(
                "Real knowledge-work deliverables (reports, memos, analyses, "
                "slides and spreadsheets) for 220 tasks across 44 occupations, "
                "compared head to head by blind graders."
            ),
            source_url="https://artificialanalysis.ai/evaluations/gdpval-aa",
            as_of=_FETCHED,
            scores=_with_inherited(_AA_SCORES[BENCHMARK_GDPVAL]),
        ),
        Benchmark(
            key=BENCHMARK_ARENA_BUSINESS,
            name="LMArena: Business, Management & Financial Ops",
            title="Business questions",
            question=(
                "Whose answers do people prefer on business, management and "
                "finance questions?"
            ),
            metric="Elo",
            description=(
                "Which answer people prefer, in blind side-by-side votes on "
                "business, management and finance prompts."
            ),
            source_url=(
                "https://arena.ai/leaderboard/text/"
                "industry-business-and-management-and-financial-operations"
            ),
            as_of=date(2026, 11, 11),
            scores=_with_inherited(_ARENA_SCORES),
            caveat="Measures preference on business topics, not writing quality as such.",
        ),
        Benchmark(
            key=BENCHMARK_GDP_PDF,
            name="GDP.pdf",
            title="Working with PDFs",
            question=(
                "How reliably can the model answer questions from real "
                "professional PDFs?"
            ),
            metric="Tasks fully correct (%)",
            description=(
                "Answering questions about real professional PDFs (4,592 pages): "
                "tables, charts, cross-references, and knowing when the "
                "document doesn't say."
            ),
            source_url="https://artificialanalysis.ai/evaluations/gdp-pdf",
            as_of=_FETCHED,
            scores=_with_inherited(_AA_SCORES[BENCHMARK_GDP_PDF]),
            scale="percent",
        ),
        Benchmark(
            key=BENCHMARK_LIVEBENCH_IF,
            name="LiveBench — Instruction Following",
            title="Following instructions",
            question=(
                "How precisely does the model follow explicit instructions on "
                "format, length and content?"
            ),
            metric="Score (0–100)",
            description=(
                "Rewriting tasks (paraphrasing, simplifying, summarizing and "
                "story writing) under explicit constraints, graded automatically "
                "against those constraints rather than by an AI judge."
            ),
            source_url="https://livebench.ai/",
            as_of=date(2026, 6, 25),
            scores=_with_inherited(_LIVEBENCH_IF_SCORES),
            scale="score",
        ),
        Benchmark(
            key=BENCHMARK_OMNISCIENCE,
            name="AA-Omniscience",
            title="Admitting what it doesn't know",
            question=(
                "When the model doesn't know the answer, how often does it make "
                "one up?"
            ),
            metric="Hallucination rate (%)",
            description=(
                "When the model doesn't know the answer, how often it makes "
                "one up instead of saying so."
            ),
            source_url="https://artificialanalysis.ai/evaluations/omniscience",
            as_of=_FETCHED,
            scores=_with_inherited(_AA_SCORES[BENCHMARK_OMNISCIENCE]),
            higher_is_better=False,
            scale="percent",
            caveat=(
                "Tested without documents (from memory), which is not the same "
                "as sticking to sources you provide."
            ),
        ),
        Benchmark(
            key=BENCHMARK_RESPONSE_TIME,
            name="AA End-to-End Response Time (100K)",
            title="Speed",
            question="How long do you wait for a reply in a long conversation?",
            metric="Seconds",
            description=(
                "Seconds from sending a request with about 100,000 tokens of "
                "context (a long chat with documents) until a complete "
                "500-token answer has arrived, thinking time included. Median "
                "of repeated measurements on the model maker's own service."
            ),
            source_url="https://artificialanalysis.ai/models",
            as_of=_RESPONSE_TIME_FETCHED,
            scores=_with_inherited(_RESPONSE_TIME_SCORES),
            higher_is_better=False,
            scale="seconds",
            caveat=(
                "Times one reply. When the assistant uses tools, such as "
                "searching or reading documents, it makes several calls, so "
                "the full wait is longer."
            ),
        ),
    )
}


def get_benchmark(key: str) -> Benchmark | None:
    return BENCHMARKS.get(key)


def get_scores(key: str, model_id: str | None = None) -> list[BenchmarkScore]:
    """A benchmark's rows (registered models only), optionally for one model."""
    bench = BENCHMARKS.get(key)
    if bench is None:
        return []
    registered = set(get_registered_model_ids())
    wanted = canonical_model_id(model_id) if model_id else None
    return [
        s for s in bench.scores
        if s.model_id in registered and (wanted is None or s.model_id == wanted)
    ]


def missing_models(key: str) -> list[str]:
    """Registered models with no score on the benchmark, in registry order."""
    covered = {s.model_id for s in get_scores(key)}
    return [m for m in get_registered_model_ids() if m not in covered]


def build_model_guide(allowed_models: list[str]) -> dict:
    """JSON-ready data for the chat "What model should I pick?" modal.

    ``allowed_models`` are the user's enabled (canonical) model IDs; only
    their scores are included.
    """
    from llm.display import get_capability_level, get_display_name, get_price_level

    allowed = [m for m in allowed_models if canonical_model_id(m) == m]
    allowed_set = set(allowed)
    benchmarks = []
    for bench in BENCHMARKS.values():
        rows = [s for s in bench.scores if s.model_id in allowed_set]
        covered = {s.model_id for s in rows}
        benchmarks.append({
            "key": bench.key,
            "name": bench.name,
            "title": bench.title,
            "question": bench.question,
            "metric": bench.metric,
            "description": bench.description,
            "caveat": bench.caveat,
            "source_url": bench.source_url,
            "as_of": bench.as_of.isoformat(),
            "higher_is_better": bench.higher_is_better,
            "scale": bench.scale,
            "rows": [
                {
                    "model_id": s.model_id,
                    "effort": s.effort,
                    "score": s.score,
                    "ci": s.ci,
                    "note": s.note,
                    "source_label": s.source_label,
                    "self_reported": s.self_reported,
                    "inherited_from": s.inherited_from,
                }
                for s in rows
            ],
            "missing": [m for m in allowed if m not in covered],
        })
    return {
        "benchmarks": benchmarks,
        "models": [
            {
                "id": m,
                "display_name": get_display_name(m),
                "price_level": get_price_level(m),
                "capability_level": get_capability_level(m),
            }
            for m in allowed
        ],
    }


__all__ = [
    "Benchmark", "BenchmarkScore", "BENCHMARKS",
    "BENCHMARK_GDPVAL", "BENCHMARK_ARENA_BUSINESS",
    "BENCHMARK_GDP_PDF", "BENCHMARK_LIVEBENCH_IF", "BENCHMARK_OMNISCIENCE",
    "BENCHMARK_RESPONSE_TIME", "get_benchmark", "get_scores", "missing_models", "build_model_guide",
]
