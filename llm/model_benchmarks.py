"""Third-party benchmark scores for the models in the registry.

Feeds the "What model should I pick?" guide. One score per (model, effort)
where the source publishes one — a model's effort variants are separate rows.

Rules for adding a score: it must appear on the benchmark's ``source_url``
(no estimates, no reading values off a different leaderboard); keep the
source's own name for the variant in ``source_label``. ``effort`` is our
reasoning level (see ``ModelInfo.reasoning_levels``; "none"/"off" = thinking
disabled) or None when the source doesn't state it. A registry model with no
row is reported by ``missing_models``. Rows for retired models are deleted —
a retired model's score is not its successor's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from llm.model_registry import canonical_model_id, get_registered_model_ids

BENCHMARK_GDPVAL = "gdpval_aa"
BENCHMARK_ARENA_BUSINESS = "arena_business"
BENCHMARK_GDP_PDF = "gdp_pdf"
BENCHMARK_LIVEBENCH_IF = "livebench_if"
BENCHMARK_OMNISCIENCE = "omniscience_hallucination"


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
    # "percent" (0-100, shown with %), "score" (0-100, shown bare).
    scale: str = "elo"


def _row(model_id, effort, score, source_label, ci=None, note=None):
    return BenchmarkScore(model_id, effort, score, source_label, ci=ci, note=note)


# Artificial Analysis rows are parsed from the model data embedded in
# artificialanalysis.ai pages (fetched 2026-09-29; GPT-6.1 Sol rows and AA's
# re-runs of GPT-6 Sol/Luna max fetched 2026-10-01); source_label is AA's slug.
# Unsuffixed OpenAI/Anthropic slugs are the max-effort runs; unsuffixed Gemini
# Flash slugs are "high". "claude-4-5-haiku-reasoning" states no effort level.
_AA_SCORES = {
    BENCHMARK_GDPVAL: (
        _row("openai/gpt-6-astra", "low", 1366, "gpt-6-astra-low"),
        _row("openai/gpt-6-astra", "medium", 1468, "gpt-6-astra-medium"),
        _row("openai/gpt-6-astra", "high", 1485, "gpt-6-astra-high"),
        _row("openai/gpt-6-astra", "xhigh", 1516, "gpt-6-astra-xhigh"),
        _row("openai/gpt-6-astra", "max", 1542, "gpt-6-astra"),
        _row("openai/gpt-6.1-sol", "low", 1297, "gpt-6-1-sol-low"),
        _row("openai/gpt-6.1-sol", "medium", 1433, "gpt-6-1-sol-medium"),
        _row("openai/gpt-6.1-sol", "high", 1486, "gpt-6-1-sol-high"),
        _row("openai/gpt-6.1-sol", "xhigh", 1510, "gpt-6-1-sol-xhigh"),
        _row("openai/gpt-6.1-sol", "max", 1575, "gpt-6-1-sol"),
        _row("openai/gpt-6-sol", "none", 1225, "gpt-6-sol-non-reasoning"),
        _row("openai/gpt-6-sol", "low", 1176, "gpt-6-sol-low"),
        _row("openai/gpt-6-sol", "medium", 1320, "gpt-6-sol-medium"),
        _row("openai/gpt-6-sol", "high", 1376, "gpt-6-sol-high"),
        _row("openai/gpt-6-sol", "xhigh", 1437, "gpt-6-sol-xhigh"),
        _row("openai/gpt-6-sol", "max", 1505, "gpt-6-sol"),
        _row("openai/gpt-6-luna", "none", 1034, "gpt-6-luna-non-reasoning"),
        _row("openai/gpt-6-luna", "low", 992, "gpt-6-luna-low"),
        _row("openai/gpt-6-luna", "medium", 1218, "gpt-6-luna-medium"),
        _row("openai/gpt-6-luna", "high", 1290, "gpt-6-luna-high"),
        _row("openai/gpt-6-luna", "xhigh", 1297, "gpt-6-luna-xhigh"),
        _row("openai/gpt-6-luna", "max", 1438, "gpt-6-luna"),
        _row("openai/gpt-5.6-sol", "none", 1226, "gpt-5-6-sol-non-reasoning"),
        _row("openai/gpt-5.6-sol", "low", 1289, "gpt-5-6-sol-low"),
        _row("openai/gpt-5.6-sol", "medium", 1403, "gpt-5-6-sol-medium"),
        _row("openai/gpt-5.6-sol", "high", 1480, "gpt-5-6-sol-high"),
        _row("openai/gpt-5.6-sol", "xhigh", 1548, "gpt-5-6-sol-xhigh"),
        _row("openai/gpt-5.6-sol", "max", 1588, "gpt-5-6-sol"),
        _row("openai/gpt-5.6-terra", "none", 1085, "gpt-5-6-terra-non-reasoning"),
        _row("openai/gpt-5.6-terra", "low", 1096, "gpt-5-6-terra-low"),
        _row("openai/gpt-5.6-terra", "medium", 1255, "gpt-5-6-terra-medium"),
        _row("openai/gpt-5.6-terra", "high", 1363, "gpt-5-6-terra-high"),
        _row("openai/gpt-5.6-terra", "xhigh", 1428, "gpt-5-6-terra-xhigh"),
        _row("openai/gpt-5.6-terra", "max", 1432, "gpt-5-6-terra"),
        _row("openai/gpt-5.4-nano", "xhigh", 937, "gpt-5-4-nano"),
        _row("anthropic/claude-fable-5-1", "low", 1450, "claude-fable-5-1-low"),
        _row("anthropic/claude-fable-5-1", "medium", 1536, "claude-fable-5-1-medium"),
        _row("anthropic/claude-fable-5-1", "high", 1617, "claude-fable-5-1-high"),
        _row("anthropic/claude-fable-5-1", "xhigh", 1721, "claude-fable-5-1-xhigh"),
        _row("anthropic/claude-fable-5-1", "max", 1735, "claude-fable-5-1"),
        _row("anthropic/claude-fable-5", "max", 1595, "claude-fable-5"),
        _row("anthropic/claude-opus-5-5", "low", 1224, "claude-opus-5-5-low"),
        _row("anthropic/claude-opus-5-5", "medium", 1576, "claude-opus-5-5-medium"),
        _row("anthropic/claude-opus-5-5", "high", 1692, "claude-opus-5-5-high"),
        _row("anthropic/claude-opus-5-5", "xhigh", 1820, "claude-opus-5-5-xhigh"),
        _row("anthropic/claude-opus-5-5", "max", 1846, "claude-opus-5-5"),
        _row("anthropic/claude-sonnet-5-5", "low", 1168, "claude-sonnet-5-5-low"),
        _row("anthropic/claude-sonnet-5-5", "medium", 1292, "claude-sonnet-5-5-medium"),
        _row("anthropic/claude-sonnet-5-5", "high", 1517, "claude-sonnet-5-5-high"),
        _row("anthropic/claude-sonnet-5-5", "xhigh", 1725, "claude-sonnet-5-5-xhigh"),
        _row("anthropic/claude-sonnet-5-5", "max", 1844, "claude-sonnet-5-5"),
        _row("anthropic/claude-sonnet-5", "off", 1200, "claude-sonnet-5-non-reasoning"),
        _row("anthropic/claude-sonnet-5", "low", 1059, "claude-sonnet-5-low"),
        _row("anthropic/claude-sonnet-5", "medium", 1145, "claude-sonnet-5-medium"),
        _row("anthropic/claude-sonnet-5", "high", 1246, "claude-sonnet-5-high"),
        _row("anthropic/claude-sonnet-5", "xhigh", 1344, "claude-sonnet-5-xhigh"),
        _row("anthropic/claude-sonnet-5", "max", 1449, "claude-sonnet-5"),
        _row("anthropic/claude-haiku-4-5", None, 719, "claude-4-5-haiku-reasoning"),
        _row("gemini/gemini-3.1-pro-preview", None, 776, "gemini-3-1-pro-preview"),
        _row("gemini/gemini-3.8-flash", "low", 1294, "gemini-3-8-flash-low"),
        _row("gemini/gemini-3.8-flash", "medium", 1409, "gemini-3-8-flash-medium"),
        _row("gemini/gemini-3.8-flash", "high", 1412, "gemini-3-8-flash"),
        _row("gemini/gemini-3.5-flash-lite", None, 970, "gemini-3-5-flash-lite"),
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
        _row("openai/gpt-6-sol", "none", 15.6, "gpt-6-sol-non-reasoning"),
        _row("openai/gpt-6-sol", "low", 21.8, "gpt-6-sol-low"),
        _row("openai/gpt-6-sol", "medium", 25.4, "gpt-6-sol-medium"),
        _row("openai/gpt-6-sol", "high", 28.0, "gpt-6-sol-high"),
        _row("openai/gpt-6-sol", "xhigh", 23.8, "gpt-6-sol-xhigh"),
        _row("openai/gpt-6-sol", "max", 25.2, "gpt-6-sol"),
        _row("openai/gpt-6-luna", "none", 5.4, "gpt-6-luna-non-reasoning"),
        _row("openai/gpt-6-luna", "low", 6.4, "gpt-6-luna-low"),
        _row("openai/gpt-6-luna", "medium", 14.0, "gpt-6-luna-medium"),
        _row("openai/gpt-6-luna", "high", 13.8, "gpt-6-luna-high"),
        _row("openai/gpt-6-luna", "xhigh", 16.8, "gpt-6-luna-xhigh"),
        _row("openai/gpt-6-luna", "max", 22.8, "gpt-6-luna"),
        _row("openai/gpt-5.6-sol", "none", 15.2, "gpt-5-6-sol-non-reasoning"),
        _row("openai/gpt-5.6-sol", "low", 21.0, "gpt-5-6-sol-low"),
        _row("openai/gpt-5.6-sol", "medium", 26.2, "gpt-5-6-sol-medium"),
        _row("openai/gpt-5.6-sol", "high", 27.8, "gpt-5-6-sol-high"),
        _row("openai/gpt-5.6-sol", "xhigh", 27.6, "gpt-5-6-sol-xhigh"),
        _row("openai/gpt-5.6-sol", "max", 27.2, "gpt-5-6-sol"),
        _row("openai/gpt-5.6-terra", "none", 11.2, "gpt-5-6-terra-non-reasoning"),
        _row("openai/gpt-5.6-terra", "low", 17.6, "gpt-5-6-terra-low"),
        _row("openai/gpt-5.6-terra", "medium", 17.0, "gpt-5-6-terra-medium"),
        _row("openai/gpt-5.6-terra", "high", 20.8, "gpt-5-6-terra-high"),
        _row("openai/gpt-5.6-terra", "xhigh", 24.6, "gpt-5-6-terra-xhigh"),
        _row("openai/gpt-5.6-terra", "max", 24.0, "gpt-5-6-terra"),
        _row("openai/gpt-5.4-nano", "xhigh", 7.8, "gpt-5-4-nano"),
        _row("anthropic/claude-fable-5-1", "low", 28.0, "claude-fable-5-1-low"),
        _row("anthropic/claude-fable-5-1", "medium", 26.8, "claude-fable-5-1-medium"),
        _row("anthropic/claude-fable-5-1", "high", 26.8, "claude-fable-5-1-high"),
        _row("anthropic/claude-fable-5-1", "xhigh", 26.2, "claude-fable-5-1-xhigh"),
        _row("anthropic/claude-fable-5-1", "max", 26.2, "claude-fable-5-1"),
        _row("anthropic/claude-fable-5", "max", 24.0, "claude-fable-5"),
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
        _row("anthropic/claude-sonnet-5", "low", 9.4, "claude-sonnet-5-low"),
        _row("anthropic/claude-sonnet-5", "medium", 11.6, "claude-sonnet-5-medium"),
        _row("anthropic/claude-sonnet-5", "high", 9.4, "claude-sonnet-5-high"),
        _row("anthropic/claude-sonnet-5", "xhigh", 12.6, "claude-sonnet-5-xhigh"),
        _row("anthropic/claude-sonnet-5", "max", 13.2, "claude-sonnet-5"),
        _row("anthropic/claude-haiku-4-5", None, 3.8, "claude-4-5-haiku-reasoning"),
        _row("gemini/gemini-3.1-pro-preview", None, 17.8, "gemini-3-1-pro-preview"),
        _row("gemini/gemini-3.8-flash", "low", 18.6, "gemini-3-8-flash-low"),
        _row("gemini/gemini-3.8-flash", "medium", 22.8, "gemini-3-8-flash-medium"),
        _row("gemini/gemini-3.8-flash", "high", 21.0, "gemini-3-8-flash"),
        _row("gemini/gemini-3.5-flash-lite", None, 13.6, "gemini-3-5-flash-lite"),
    ),
    BENCHMARK_OMNISCIENCE: (
        _row("openai/gpt-6-astra", "low", 46.9, "gpt-6-astra-low"),
        _row("openai/gpt-6-astra", "medium", 46.5, "gpt-6-astra-medium"),
        _row("openai/gpt-6-astra", "high", 44.8, "gpt-6-astra-high"),
        _row("openai/gpt-6-astra", "xhigh", 48.3, "gpt-6-astra-xhigh"),
        _row("openai/gpt-6-astra", "max", 51.3, "gpt-6-astra"),
        _row("openai/gpt-6.1-sol", "high", 49.4, "gpt-6-1-sol-high"),
        _row("openai/gpt-6.1-sol", "xhigh", 50.9, "gpt-6-1-sol-xhigh"),
        _row("openai/gpt-6.1-sol", "max", 54.3, "gpt-6-1-sol"),
        _row("openai/gpt-6-sol", "none", 84.0, "gpt-6-sol-non-reasoning"),
        _row("openai/gpt-6-sol", "low", 50.7, "gpt-6-sol-low"),
        _row("openai/gpt-6-sol", "medium", 56.8, "gpt-6-sol-medium"),
        _row("openai/gpt-6-sol", "high", 58.1, "gpt-6-sol-high"),
        _row("openai/gpt-6-sol", "xhigh", 58.9, "gpt-6-sol-xhigh"),
        _row("openai/gpt-6-sol", "max", 60.1, "gpt-6-sol"),
        _row("openai/gpt-6-luna", "none", 78.9, "gpt-6-luna-non-reasoning"),
        _row("openai/gpt-6-luna", "low", 84.3, "gpt-6-luna-low"),
        _row("openai/gpt-6-luna", "medium", 84.7, "gpt-6-luna-medium"),
        _row("openai/gpt-6-luna", "high", 84.4, "gpt-6-luna-high"),
        _row("openai/gpt-6-luna", "xhigh", 82.4, "gpt-6-luna-xhigh"),
        _row("openai/gpt-6-luna", "max", 76.7, "gpt-6-luna"),
        _row("openai/gpt-5.6-sol", "none", 92.8, "gpt-5-6-sol-non-reasoning"),
        _row("openai/gpt-5.6-sol", "low", 89.4, "gpt-5-6-sol-low"),
        _row("openai/gpt-5.6-sol", "medium", 90.8, "gpt-5-6-sol-medium"),
        _row("openai/gpt-5.6-sol", "high", 91.2, "gpt-5-6-sol-high"),
        _row("openai/gpt-5.6-sol", "xhigh", 91.9, "gpt-5-6-sol-xhigh"),
        _row("openai/gpt-5.6-sol", "max", 92.2, "gpt-5-6-sol"),
        _row("openai/gpt-5.6-terra", "none", 95.0, "gpt-5-6-terra-non-reasoning"),
        _row("openai/gpt-5.6-terra", "low", 89.9, "gpt-5-6-terra-low"),
        _row("openai/gpt-5.6-terra", "medium", 89.8, "gpt-5-6-terra-medium"),
        _row("openai/gpt-5.6-terra", "high", 89.8, "gpt-5-6-terra-high"),
        _row("openai/gpt-5.6-terra", "xhigh", 89.0, "gpt-5-6-terra-xhigh"),
        _row("openai/gpt-5.6-terra", "max", 87.9, "gpt-5-6-terra"),
        _row("openai/gpt-5.4-nano", "none", 61.5, "gpt-5-4-nano-non-reasoning"),
        _row("openai/gpt-5.4-nano", "medium", 51.1, "gpt-5-4-nano-medium"),
        _row("openai/gpt-5.4-nano", "xhigh", 74.2, "gpt-5-4-nano"),
        _row("anthropic/claude-fable-5-1", "low", 65.6, "claude-fable-5-1-low"),
        _row("anthropic/claude-fable-5-1", "medium", 69.1, "claude-fable-5-1-medium"),
        _row("anthropic/claude-fable-5-1", "high", 68.8, "claude-fable-5-1-high"),
        _row("anthropic/claude-fable-5-1", "xhigh", 70.5, "claude-fable-5-1-xhigh"),
        _row("anthropic/claude-fable-5-1", "max", 72.6, "claude-fable-5-1"),
        _row("anthropic/claude-fable-5", "max", 63.6, "claude-fable-5"),
        _row("anthropic/claude-opus-5-5", "low", 67.6, "claude-opus-5-5-low"),
        _row("anthropic/claude-opus-5-5", "medium", 68.4, "claude-opus-5-5-medium"),
        _row("anthropic/claude-opus-5-5", "high", 67.6, "claude-opus-5-5-high"),
        _row("anthropic/claude-opus-5-5", "xhigh", 65.7, "claude-opus-5-5-xhigh"),
        _row("anthropic/claude-opus-5-5", "max", 58.6, "claude-opus-5-5"),
        _row("anthropic/claude-sonnet-5-5", "low", 50.2, "claude-sonnet-5-5-low"),
        _row("anthropic/claude-sonnet-5-5", "medium", 51.2, "claude-sonnet-5-5-medium"),
        _row("anthropic/claude-sonnet-5-5", "high", 64.6, "claude-sonnet-5-5-high"),
        _row("anthropic/claude-sonnet-5-5", "xhigh", 62.9, "claude-sonnet-5-5-xhigh"),
        _row("anthropic/claude-sonnet-5-5", "max", 47.0, "claude-sonnet-5-5"),
        _row("anthropic/claude-sonnet-5", "off", 52.0, "claude-sonnet-5-non-reasoning"),
        _row("anthropic/claude-sonnet-5", "low", 72.8, "claude-sonnet-5-low"),
        _row("anthropic/claude-sonnet-5", "medium", 69.9, "claude-sonnet-5-medium"),
        _row("anthropic/claude-sonnet-5", "high", 65.7, "claude-sonnet-5-high"),
        _row("anthropic/claude-sonnet-5", "xhigh", 58.6, "claude-sonnet-5-xhigh"),
        _row("anthropic/claude-sonnet-5", "max", 39.4, "claude-sonnet-5"),
        _row("anthropic/claude-haiku-4-5", "off", 25.7, "claude-4-5-haiku"),
        _row("anthropic/claude-haiku-4-5", None, 27.3, "claude-4-5-haiku-reasoning"),
        _row("gemini/gemini-3.1-pro-preview", None, 50.9, "gemini-3-1-pro-preview"),
        _row("gemini/gemini-3.8-flash", "low", 64.6, "gemini-3-8-flash-low"),
        _row("gemini/gemini-3.8-flash", "medium", 51.9, "gemini-3-8-flash-medium"),
        _row("gemini/gemini-3.8-flash", "high", 55.2, "gemini-3-8-flash"),
        _row("gemini/gemini-3.5-flash-lite", None, 34.4, "gemini-3-5-flash-lite"),
    ),
}

_ARENA_SCORES = (
    _row("openai/gpt-6-astra", "max", 1482.7, "gpt-6-astra-max", ci=16.3),
    _row("openai/gpt-6-sol", "max", 1464.6, "gpt-6-sol-max", ci=19.6),
    _row("openai/gpt-6-luna", "max", 1452.5, "gpt-6-luna-max", ci=20.2),
    _row("openai/gpt-5.6-sol", "xhigh", 1485.4, "gpt-5.6-sol-xhigh", ci=8.1),
    _row("openai/gpt-5.6-terra", "xhigh", 1465.6, "gpt-5.6-terra-xhigh", ci=8.0),
    _row("openai/gpt-5.4-nano", "high", 1405.1, "gpt-5.4-nano-high", ci=6.6),
    _row("anthropic/claude-fable-5-1", "max", 1485.1, "claude-fable-5.1-max", ci=13.7),
    _row("anthropic/claude-fable-5", "high", 1503.1, "claude-fable-5-high", ci=7.6),
    _row(
        "anthropic/claude-opus-5-5", "high", 1499.4, "claude-opus-5.5-high", ci=28.2,
        note="Only 436 votes so far.",
    ),
    _row("anthropic/claude-sonnet-5", "high", 1464.8, "claude-sonnet-5-high", ci=7.3),
    _row("anthropic/claude-haiku-4-5", None, 1419.8, "claude-haiku-4-5-20251001", ci=4.5),
    _row("gemini/gemini-3.1-pro-preview", None, 1477.2, "gemini-3.1-pro-preview", ci=5.1),
    _row("gemini/gemini-3.8-flash", "high", 1481.7, "gemini-3.8-flash-high", ci=9.8),
    _row("gemini/gemini-3.5-flash-lite", None, 1453.6, "gemini-3.5-flash-lite", ci=8.2),
)

# LiveBench release 2026-06-25 (livebench.ai/table_2026_06_25.csv): the IF
# category is the mean of paraphrase, simplify, story_generation and summarize.
# New models are added to the current release, so all rows share one question
# set. Haiku 4.5 is not listed.
_LIVEBENCH_IF_SCORES = (
    _row("openai/gpt-6-astra", "max", 75.58, "gpt-6-astra-max"),
    _row("openai/gpt-6.1-sol", "max", 74.15, "gpt-6.1-sol-max"),
    _row("openai/gpt-6.1-sol", "xhigh", 71.33, "gpt-6.1-sol-xhigh"),
    _row("openai/gpt-6-sol", "max", 68.57, "gpt-6-sol-max"),
    _row("openai/gpt-6-luna", "max", 55.93, "gpt-6-luna-max"),
    _row("openai/gpt-5.6-sol", "max", 71.85, "gpt-5.6-sol-max"),
    _row("openai/gpt-5.6-terra", "max", 64.62, "gpt-5.6-terra-max"),
    _row("openai/gpt-5.4-nano", "xhigh", 67.20, "gpt-5.4-nano-xhigh"),
    _row("anthropic/claude-fable-5-1", "max", 72.99, "claude-fable-5-1-max-effort"),
    _row("anthropic/claude-fable-5", "max", 75.77, "claude-fable-5-max-effort"),
    _row("anthropic/claude-opus-5-5", "xhigh", 67.05, "claude-opus-5-5-xhigh-effort"),
    _row("anthropic/claude-opus-5-5", "max", 65.74, "claude-opus-5-5-max-effort"),
    _row("anthropic/claude-sonnet-5-5", "xhigh", 70.52, "claude-sonnet-5-5-xhigh-effort"),
    _row("anthropic/claude-sonnet-5-5", "max", 56.79, "claude-sonnet-5-5-max-effort"),
    _row("anthropic/claude-sonnet-5", "xhigh", 63.86, "claude-sonnet-5-xhigh-effort"),
    _row("gemini/gemini-3.1-pro-preview", "high", 79.10, "gemini-3.1-pro-preview-high"),
    _row("gemini/gemini-3.8-flash", "high", 81.41, "gemini-3.8-flash-high"),
    _row("gemini/gemini-3.5-flash-lite", "high", 67.24, "gemini-3.5-flash-lite-high"),
)

_FETCHED = date(2026, 9, 29)

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
            scores=_AA_SCORES[BENCHMARK_GDPVAL],
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
            as_of=date(2026, 9, 25),
            scores=_ARENA_SCORES,
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
            scores=_AA_SCORES[BENCHMARK_GDP_PDF],
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
            scores=_LIVEBENCH_IF_SCORES,
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
            scores=_AA_SCORES[BENCHMARK_OMNISCIENCE],
            higher_is_better=False,
            scale="percent",
            caveat=(
                "Tested without documents (from memory), which is not the same "
                "as sticking to sources you provide."
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
    "get_benchmark", "get_scores", "missing_models", "build_model_guide",
]
