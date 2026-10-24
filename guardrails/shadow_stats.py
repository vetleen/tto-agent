"""Pure aggregation helpers for the shadow-comparison reports.

Shared by ``classifier_shadow_report`` and the PII shadow report; the chunk report
(``shadow_chunk_scan_report``) predates this module and keeps its own ``summarize``.
Everything here works on plain values — ``(prod_flagged, probability)`` pairs and
verdict strings — so each report maps its rows once and the arithmetic is tested in
one place.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

DECILES = 10
THRESHOLDS = (0.3, 0.5, 0.7)


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}% ({num}/{den})" if den else "n/a (0)"


def decile(p: float) -> int:
    return min(DECILES - 1, max(0, int(p * DECILES)))


def mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def confusion(pairs: Iterable[tuple[bool, bool]]) -> dict:
    """``(prod_flagged, quick_flagged)`` → the four cells (missing cells are 0)."""
    cells = Counter()
    for prod, quick in pairs:
        if prod and quick:
            cells["both_positive"] += 1
        elif prod:
            cells["classifier_only"] += 1
        elif quick:
            cells["quick_only"] += 1
        else:
            cells["both_negative"] += 1
    return {k: cells.get(k, 0) for k in ("both_negative", "both_positive", "classifier_only", "quick_only")}


def agreement(pairs: Iterable[tuple[bool, float]], thresholds=THRESHOLDS) -> dict[float, str]:
    """``(prod_flagged, probability)`` → agreement share when the quick verdict is
    re-thresholded at each value."""
    pairs = list(pairs)
    return {t: pct(sum(1 for prod, p in pairs if prod == (p >= t)), len(pairs)) for t in thresholds}


def deciles(pairs: Iterable[tuple[bool, float]]) -> dict[str, list[int]]:
    """Quick-probability histograms split by the production verdict."""
    flagged, clean = [0] * DECILES, [0] * DECILES
    for prod, p in pairs:
        (flagged if prod else clean)[decile(p)] += 1
    return {"prod_flagged": flagged, "prod_not_flagged": clean}


def allow_share(verdicts: Iterable[str], allow: set[str], act: set[str]) -> str:
    """The "false-positive rate": the allow share among judged verdicts."""
    judged = [v for v in verdicts if v in allow or v in act]
    return pct(sum(1 for v in judged if v in allow), len(judged))


# Which production / blind-review verdict strings count as "allow" (the classifier
# was wrong) and which as "act" (the classifier was right), per classifier family.
# Only verdicts a REVIEWER made count: the skill scan's reviewer-less threshold
# outcomes (threshold_logged / threshold_quarantine), no_model, error, not_reviewed
# and "" are excluded from the FP denominators — as the chunk report excludes them.
FAMILY_VERDICTS: dict[str, dict[str, set[str]]] = {
    "web_content": {"allow": {"allow"}, "act": {"withhold"}},
    "skill_content": {"allow": {"allow"}, "act": {"quarantine"}},
    "chat_message": {"allow": {"dismiss"}, "act": {"warn", "block", "suspend"}},
    "identity_field": {"allow": set(), "act": {"blocked"}},
}


def family_verdicts(family: str) -> dict[str, set[str]]:
    return FAMILY_VERDICTS.get(family, {"allow": set(), "act": set()})
