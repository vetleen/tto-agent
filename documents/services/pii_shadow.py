"""Nightly shadow comparison of the PII category classifier against the decisions service.

Production (``documents.services.pii_scan.scan_pii_categories_for_version``) windows a
version's chunks, asks the cheap model for ten GDPR category booleans per window and
escalates the two gated Article 9/10 candidates to the PII reviewer. It stores the
per-version union as ``DataRoomDocumentTag`` rows and a ``PIIReviewEvent`` per reviewed
window — not the per-window ordinary verdicts. This module re-runs the previous day's
READY versions through the decisions quick pathway (the same windows via
``iter_pii_windows``, the same system prompt and head/tail truncation, ten predicates
named after the categories) and records, per window, the quick probabilities next to
what production recorded for the window (the review event) and the version (its tags).

The PII reviewer is already blind — its Layer-1 block lists candidate categories only,
no confidence or reasoning — so production's own review IS the blind verdict for the
windows it flagged. The shadow therefore reviews only quick-only gated positives
(quick ≥ 0.5 on Art. 9/10, no production candidate in that window and the category
absent from the version's tags), with the same reviewer and prompt.

Side-effect free: it writes only ``PIIScanShadowRun``/``PIIScanShadowResult`` rows (ids
and numbers, never text) and the ``LLMCallLog`` rows of its own calls (grouped by the
run's trace_id). It never touches versions, tags or review events, and it shares the
chunk shadow's decisions service and skip window (``decisions_quick_skip:shadow``) so a
failure never routes live traffic to the structured pathway. The PII scan itself is
unchanged. Report: ``manage.py pii_shadow_report``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from datetime import date

from django.conf import settings

from documents.services.pii_scan import (
    GATED_CATEGORIES,
    PII_CATEGORIES,
    _PII_SYSTEM_PROMPT,
    iter_pii_windows,
    pii_window_text,
)
from llm.types.decisions import InputMessage, Question, TextPart

logger = logging.getLogger(__name__)

SHADOW_FEATURE_KEY = "shadow_pii_scan"
# Quick probability at/above which a category counts as present (the report
# recomputes agreement at other thresholds from the stored probabilities).
SHADOW_FLAG_THRESHOLD = 0.5
# Consecutive failed decide() calls after which the night is abandoned.
CONSECUTIVE_DECIDE_ERRORS_ABORT = 3
PATHWAY_QUICK = "quick"


# ---------------------------------------------------------------------------
# The decisions call
# ---------------------------------------------------------------------------


def build_questions() -> list[Question]:
    """One predicate per GDPR category, worded from the production schema's own
    field descriptions (``PIICategoryOutput``) so both systems judge the same thing."""
    from llm.types.structured import PIICategoryOutput

    questions = []
    for category in PII_CATEGORIES:
        description = (PIICategoryOutput.model_fields[category].description or category).strip().rstrip(".")
        # Continue the sentence: lowercase a leading capitalised word ("Personal identity
        # data" → "personal …") but leave acronyms alone ("GDPR Article 9 …").
        if len(description) > 1 and description[0].isupper() and description[1].islower():
            description = description[0].lower() + description[1:]
        questions.append(Question(type="predicate", name=category, instructions=f"The text contains {description}."))
    return questions


def build_input(window_text: str) -> list[InputMessage]:
    """The PII system prompt as the developer message + the window text as production
    sends it (same head/tail truncation as ``scan_pii_categories``)."""
    from documents.services.description import _prepare_document_text

    return [
        InputMessage(role="developer", content=[TextPart(text=_PII_SYSTEM_PROMPT)]),
        InputMessage(role="user", content=[TextPart(text=_prepare_document_text(window_text))]),
    ]


def decide_window(service, window_text: str, *, org_id, context) -> tuple[dict[str, float | None], str]:
    """One decide() over a window → ``({category: probability | None}, pathway)``."""
    result = service.decide(
        build_questions(), build_input(window_text),
        feature_key=SHADOW_FEATURE_KEY, org_id=org_id, context=context,
    )
    probabilities = {}
    for category in PII_CATEGORIES:
        answer = result[category]
        probabilities[category] = answer.probability if answer.type == "predicate" else None
    return probabilities, result.pathway


# ---------------------------------------------------------------------------
# Selection and production verdicts
# ---------------------------------------------------------------------------


def select_version_ids(day: date) -> list[int]:
    """READY versions created on ``day`` (Oslo) that have chunks."""
    from django.db.models import Exists, OuterRef

    from documents.models import DataRoomDocument, DataRoomDocumentChunk, DataRoomDocumentVersion
    from guardrails.shadow import day_bounds

    start, end = day_bounds(day)
    has_chunks = DataRoomDocumentChunk.objects.filter(version_id=OuterRef("pk"))
    return list(
        DataRoomDocumentVersion.objects.filter(
            created_at__gte=start, created_at__lt=end, status=DataRoomDocument.Status.READY,
        )
        .filter(Exists(has_chunks))
        .order_by("pk")
        .values_list("pk", flat=True)
    )


def recorded_windows(day: date, version_id: int) -> set[int]:
    from documents.models import PIIScanShadowResult

    return set(
        PIIScanShadowResult.objects.filter(run_date=day, version_id=version_id)
        .values_list("window_index", flat=True)
    )


def production_window_events(version_id: int) -> dict[int, dict]:
    """The PII review event per window index (the classifier's gated candidates and
    the reviewer's verdict); windows without a candidate have none."""
    from documents.models import PIIReviewEvent

    rows = (
        PIIReviewEvent.objects.filter(version_id=version_id)
        .order_by("created_at")
        .values("window_index", "candidate_categories", "action", "article_9", "article_10", "confidence")
    )
    return {row["window_index"]: row for row in rows}


def production_version_categories(version_id: int) -> list[str]:
    """The PII categories production tagged the version with (its final union)."""
    from documents.models import DataRoomDocumentTag

    return sorted(
        DataRoomDocumentTag.objects.filter(version_id=version_id, key__in=PII_CATEGORIES, value="true")
        .values_list("key", flat=True)
    )


# ---------------------------------------------------------------------------
# The blind review
# ---------------------------------------------------------------------------


def blind_review(window_text: str, candidates: list[str], *, document_title: str, org_id, context):
    """The same PII reviewer production uses (already blind: candidates only).

    Returns ``(verdict, article_9, article_10, confidence)``; verdict is ``reviewed``,
    ``error`` when the reviewer raised, ``no_model`` when none resolves.
    """
    from documents.services.pii_review import review_flagged_pii_window

    try:
        decision = review_flagged_pii_window(
            window_text, list(candidates), document_title, org_id, user_id=None, context=context,
        )
    except Exception:
        logger.exception("shadow_pii_scan: reviewer errored candidates=%s", candidates)
        return "error", None, None, None
    if decision is None:
        return "no_model", None, None, None
    return "reviewed", bool(decision.article_9), bool(decision.article_10), decision.confidence


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_pii_shadow_for_day(day: date, *, max_windows: int | None = None, max_reviews: int | None = None) -> dict:
    """Compare ``day``'s versions window by window; returns the summary (also stored
    on the run row). Idempotent: windows already recorded for ``day`` are skipped, so a
    retry or a manual ``--run`` fills in what a partial night left out.
    """
    import os

    from django.db.models import Count, Sum
    from django.db.utils import InterfaceError, OperationalError
    from django.utils import timezone

    from documents.models import DataRoomDocumentVersion, PIIScanShadowResult, PIIScanShadowRun
    from documents.services.pii_scan import org_id_for_document, resolve_pii_gate
    from guardrails.shadow import get_shadow_decisions_service, shadow_availability
    from llm.models import LLMCallLog
    from llm.types.context import RunContext

    t0 = time.perf_counter()
    trace_id = f"shadow-pii-{day:%Y%m%d}-{uuid.uuid4().hex[:12]}"
    context = RunContext(trace_id=trace_id)
    windows_left = max_windows if max_windows is not None else int(settings.PII_SHADOW_MAX_WINDOWS)
    reviews_left = max_reviews if max_reviews is not None else int(settings.PII_SHADOW_MAX_REVIEWS)
    counters: Counter = Counter()
    skipped: Counter = Counter()
    versions_seen: set[int] = set()
    orgs_seen: set = set()

    def summary(run=None) -> dict:
        out = dict(counters)
        out.update({
            "day": day.isoformat(),
            "trace_id": trace_id,
            "versions": len(versions_seen),
            "orgs": len(orgs_seen),
            "skipped": dict(skipped),
            "duration_s": round(time.perf_counter() - t0, 1),
        })
        if run is not None:
            cost = LLMCallLog.objects.filter(trace_id=trace_id).aggregate(total=Sum("cost_usd"), calls=Count("id"))
            out["cost_usd"] = float(cost["total"] or 0)
            out["llm_calls"] = cost["calls"] or 0
            run.summary = out
            run.finished_at = timezone.now()
            run.save(update_fields=["summary", "finished_at"])
        logger.info(
            "shadow_pii_scan: " + " ".join(f"{k}={v}" for k, v in out.items() if k != "trace_id")
            + f" trace_id={trace_id}",
        )
        return out

    try:
        run = PIIScanShadowRun.objects.create(run_date=day, trace_id=trace_id)
        if not os.environ.get("OPENAI_API_KEY"):
            skipped["no_openai_key"] += 1
            return summary(run)
        version_ids = select_version_ids(day)
    except (OperationalError, InterfaceError):
        logger.info("shadow_pii_scan: DB unavailable; will retry next tick.", exc_info=True)
        return {"day": day.isoformat(), "skipped": {"db_unavailable": 1}}

    from core.preferences import resolve_decisions_settings

    service = get_shadow_decisions_service()
    org_skip: dict = {}  # org_id -> skip reason or None
    consecutive_errors = 0
    aborted = False

    for version_id in version_ids:
        if aborted:
            break
        version = DataRoomDocumentVersion.objects.select_related("document").filter(pk=version_id).first()
        if version is None:
            continue
        doc = version.document
        org_id = org_id_for_document(doc)
        if org_id not in org_skip:
            reason = None
            pii_model, pii_enabled, _quarantine = resolve_pii_gate(org_id)
            if not pii_model or not pii_enabled:
                reason = "pii_off"
            else:
                prefs = resolve_decisions_settings(org_id)
                if not prefs.quick_enabled or not prefs.quick_model:
                    reason = "quick_off"
            org_skip[org_id] = reason
        if org_skip[org_id]:
            skipped[org_skip[org_id]] += 1
            continue

        already = recorded_windows(day, version_id)
        prod_events = production_window_events(version_id)
        prod_categories = production_version_categories(version_id)
        document_title = getattr(doc, "display_name", "") or ""
        rows = []
        for index, window in iter_pii_windows(version_id):
            if index in already:
                continue
            if windows_left <= 0:
                skipped["cap_windows"] += 1
                continue
            text = pii_window_text(window)
            if not text.strip():
                skipped["empty_window"] += 1
                continue
            if shadow_availability().is_suspended():
                skipped["quick_suspended"] += 1
                continue
            try:
                probabilities, pathway = decide_window(service, text, org_id=org_id, context=context)
            except Exception:
                logger.exception("shadow_pii_scan: decide failed version_id=%s window=%s", version_id, index)
                counters["decide_errors"] += 1
                consecutive_errors += 1
                if consecutive_errors >= CONSECUTIVE_DECIDE_ERRORS_ABORT:
                    skipped["aborted"] += 1
                    aborted = True
                    break
                continue
            consecutive_errors = 0
            windows_left -= 1
            versions_seen.add(version_id)
            orgs_seen.add(org_id)
            counters["windows"] += 1
            if pathway != PATHWAY_QUICK:
                counters["non_quick"] += 1
            if any(p is None for p in probabilities.values()):
                counters["refusals"] += 1

            event = prod_events.get(index)
            prod_candidates = list((event or {}).get("candidate_categories") or [])
            quick_gated = [
                c for c in GATED_CATEGORIES
                if probabilities.get(c) is not None and probabilities[c] >= SHADOW_FLAG_THRESHOLD
            ]
            counters["quick_gated_windows"] += 1 if quick_gated else 0
            counters["prod_gated_windows"] += 1 if prod_candidates else 0
            # Production's own review is already blind, so only quick-only gated
            # positives the version was never tagged with need a second opinion.
            to_review = [c for c in quick_gated if c not in prod_candidates and c not in prod_categories]
            blind, a9, a10, conf = "", None, None, None
            if to_review:
                if reviews_left > 0:
                    reviews_left -= 1
                    counters["reviews"] += 1
                    blind, a9, a10, conf = blind_review(
                        text, to_review, document_title=document_title, org_id=org_id, context=context,
                    )
                    if blind == "reviewed":
                        confirmed = (a9 and "pii_special_category" in to_review) or (a10 and "pii_criminal_offence" in to_review)
                        counters["blind_confirmed" if confirmed else "blind_dismissed"] += 1
                    else:
                        counters["blind_error"] += 1
                else:
                    blind = "not_reviewed"
                    counters["reviews_capped"] += 1

            rows.append(PIIScanShadowResult(
                run_date=day, trace_id=trace_id, org_id=org_id,
                version_id=version_id, document_id=doc.pk if doc else None,
                window_index=index,
                chunk_index_from=window[0]["chunk_index"], chunk_index_to=window[-1]["chunk_index"],
                window_tokens=sum(int(c.get("token_count") or 0) for c in window), window_chars=len(text),
                quick_probabilities=probabilities, quick_pathway=pathway,
                prod_candidates=prod_candidates,
                prod_action=str((event or {}).get("action") or ""),
                prod_article_9=(event or {}).get("article_9") if event else None,
                prod_article_10=(event or {}).get("article_10") if event else None,
                prod_version_categories=prod_categories,
                blind_review=blind, blind_candidates=to_review if blind else [],
                blind_article_9=a9, blind_article_10=a10, blind_confidence=conf,
            ))
        if rows:
            PIIScanShadowResult.objects.bulk_create(rows, ignore_conflicts=True)

    return summary(run)


__all__ = [
    "SHADOW_FEATURE_KEY",
    "SHADOW_FLAG_THRESHOLD",
    "build_input",
    "build_questions",
    "decide_window",
    "production_version_categories",
    "production_window_events",
    "run_pii_shadow_for_day",
    "select_version_ids",
]
