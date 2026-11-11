"""Nightly shadow of the PII scan: the legacy structured classifier against production.

Production (``documents.services.pii_scan.scan_pii_categories_for_version``) windows a
version's chunks and asks the Decisions service ten GDPR category predicates per window
(``scan_pii_window``); Art. 9/10 positives go to the PII reviewer. For every window it
scans it calls :func:`record_window_verdict`, which writes one *pending*
``PIIScanShadowResult`` (``direction="legacy_shadow"``): the Decisions probabilities,
pathway and the window's chunk range — ids and numbers only. Every night
:func:`run_pii_shadow_for_day` fills in the other half: it rebuilds each pending
window with ``iter_pii_windows`` (skipped when the chunk range no longer matches), runs
the legacy structured classifier (``pii_scan._scan_pii_categories_legacy``, same prompt
and truncation) and records production's review event for the window and the
version's tags.

The PII reviewer is blind (its Layer-1 block lists candidate categories only), so
production's own review is the blind verdict for every window Decisions flagged. The
shadow therefore reviews only legacy-only gated positives — the legacy classifier says
Art. 9/10, Decisions said < 0.5, no production candidate in that window and the
category absent from the version's tags — with the same reviewer and prompt.

Side-effect free: it writes only ``PIIScanShadowRun``/``PIIScanShadowResult`` rows
(never text) and the ``LLMCallLog`` rows of its own calls (grouped by the run's
trace_id). It never touches versions, tags or review events. Kill switch
``PII_SHADOW_ENABLED`` (it also gates the pending-row writes); caps
``PII_SHADOW_MAX_WINDOWS``/``_MAX_REVIEWS``. Report: ``manage.py pii_shadow_report``.
Rows before 2026-11 (``direction="decisions_shadow"``) are the opposite comparison.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter, defaultdict
from datetime import date

from django.conf import settings

from documents.services.pii_scan import (
    GATED_CATEGORIES,
    PII_CATEGORIES,
    PII_FLAG_THRESHOLD,
    iter_pii_windows,
    pii_window_text,
)

logger = logging.getLogger(__name__)

SHADOW_FEATURE_KEY = "shadow_pii_scan"
DIRECTION = "legacy_shadow"
# Consecutive failed legacy calls after which the night is abandoned.
CONSECUTIVE_ERRORS_ABORT = 3
# Failed legacy calls after which a window is given up (skipped, "classifier_error").
# Without a bound, windows that always fail stay first in line and abort every night
# before anything behind them is reached.
MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Production side: the pending half
# ---------------------------------------------------------------------------


def record_window_verdict(*, version_id, document_id, org_id, window_index, window: list[dict], text: str, verdict) -> None:
    """One pending shadow row for a window production just scanned. Never raises.

    ``verdict`` is the production ``PIIWindowVerdict``; a verdict the legacy classifier
    produced (Decisions failed) is recorded with ``shadow_note="prod_legacy:<error>"``
    and no probabilities.
    """
    if not getattr(settings, "PII_SHADOW_ENABLED", False) or not version_id or not window:
        return
    try:
        from django.utils import timezone

        from documents.models import PIIScanShadowResult

        legacy = verdict.source != "decisions"
        PIIScanShadowResult.objects.bulk_create([PIIScanShadowResult(
            run_date=timezone.localdate(), org_id=org_id, version_id=version_id, document_id=document_id,
            window_index=window_index,
            chunk_index_from=window[0]["chunk_index"], chunk_index_to=window[-1]["chunk_index"],
            window_tokens=sum(int(c.get("token_count") or 0) for c in window), window_chars=len(text),
            quick_probabilities={} if legacy else dict(verdict.probabilities or {}),
            quick_pathway="" if legacy else (verdict.pathway or "")[:12],
            classifier_categories=sorted(verdict.categories) if legacy else [],
            direction=DIRECTION, state="pending",
            shadow_note=(f"prod_legacy:{verdict.decide_error}" if legacy else "")[:64],
        )], ignore_conflicts=True)
    except Exception:
        logger.warning("shadow_pii_scan: recording a production verdict failed (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# Production verdicts
# ---------------------------------------------------------------------------


def production_window_events(version_id: int) -> dict[int, dict]:
    """The PII review event per window index (production's gated candidates and the
    reviewer's verdict); windows without a candidate have none."""
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
# The nightly run
# ---------------------------------------------------------------------------


def pending_rows(day: date):
    """Pending rows recorded on or before ``day``."""
    from documents.models import PIIScanShadowResult

    return PIIScanShadowResult.objects.filter(direction=DIRECTION, state="pending", run_date__lte=day)


_DONE_FIELDS = [
    "classifier_categories", "prod_candidates", "prod_action", "prod_article_9", "prod_article_10",
    "prod_version_categories", "blind_review", "blind_candidates", "blind_article_9", "blind_article_10",
    "blind_confidence", "state", "shadow_note", "trace_id",
]


def run_pii_shadow_for_day(day: date, *, max_windows: int | None = None, max_reviews: int | None = None) -> dict:
    """Fill in the legacy half of every pending window up to ``day``; returns the
    summary (also stored on the run row). Resumable: only pending rows are selected
    and each version is saved as it finishes.
    """
    from django.db.models import Count, Sum
    from django.db.utils import InterfaceError, OperationalError
    from django.utils import timezone

    from documents.models import DataRoomDocumentVersion, PIIScanShadowResult, PIIScanShadowRun
    from documents.services.description import _prepare_document_text
    from documents.services.pii_scan import _scan_pii_categories_legacy, org_id_for_document, resolve_pii_gate
    from llm.models import LLMCallLog
    from llm.types.context import RunContext

    t0 = time.perf_counter()
    trace_id = f"shadow-pii-{day:%Y%m%d}-{uuid.uuid4().hex[:12]}"
    context = RunContext(trace_id=trace_id)
    log_metadata = {"kind": "shadow", "feature": SHADOW_FEATURE_KEY}
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
            "direction": DIRECTION,
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

    def mark_skipped(rows, note):
        for row in rows:
            row.state, row.shadow_note = "skipped", note[:64]
        PIIScanShadowResult.objects.bulk_update(rows, ["state", "shadow_note"])
        skipped[note] += len(rows)

    def record_failure(row):
        """Count a failed legacy call against its window; give up at MAX_ATTEMPTS."""
        row.attempts = (row.attempts or 0) + 1
        if row.attempts >= MAX_ATTEMPTS:
            row.state, row.shadow_note = "skipped", "classifier_error"
            skipped["classifier_error"] += 1
        row.save(update_fields=["attempts", "state", "shadow_note"])

    try:
        run = PIIScanShadowRun.objects.create(run_date=day, trace_id=trace_id)
        by_version: dict[int, list] = defaultdict(list)
        for row in pending_rows(day).order_by("version_id", "window_index", "run_date"):
            by_version[row.version_id].append(row)
    except (OperationalError, InterfaceError):
        logger.info("shadow_pii_scan: DB unavailable; will retry next tick.", exc_info=True)
        return {"day": day.isoformat(), "skipped": {"db_unavailable": 1}}

    consecutive_errors = 0
    for version_id, rows in by_version.items():
        if windows_left <= 0:
            skipped["cap_windows"] += len(rows)  # they stay pending
            continue
        version = DataRoomDocumentVersion.objects.select_related("document").filter(pk=version_id).first()
        if version is None:
            mark_skipped(rows, "version_gone")
            continue
        doc = version.document
        org_id = org_id_for_document(doc)
        pii_model, pii_enabled, _quarantine = resolve_pii_gate(org_id)
        if not pii_model or not pii_enabled:
            mark_skipped(rows, "pii_off")
            continue
        wanted = {r.window_index: r for r in rows}
        windows = {index: window for index, window in iter_pii_windows(version_id) if index in wanted}
        drifted = [
            r for r in rows
            if r.window_index not in windows
            or (windows[r.window_index][0]["chunk_index"], windows[r.window_index][-1]["chunk_index"])
            != (r.chunk_index_from, r.chunk_index_to)
        ]
        if drifted:
            mark_skipped(drifted, "drifted")
        rows = [r for r in rows if r not in drifted]
        if not rows:
            continue
        versions_seen.add(version_id)
        orgs_seen.add(org_id)
        prod_events = production_window_events(version_id)
        prod_categories = production_version_categories(version_id)
        document_title = getattr(doc, "display_name", "") or ""
        done = []
        for row in rows:
            if windows_left <= 0:
                skipped["cap_windows"] += 1
                continue
            text = pii_window_text(windows[row.window_index])
            if not row.shadow_note.startswith("prod_legacy"):
                try:
                    categories = _scan_pii_categories_legacy(
                        _prepare_document_text(text), None, org_id, context=context, log_metadata=log_metadata,
                    )
                except Exception:
                    logger.exception("shadow_pii_scan: legacy classifier failed version_id=%s window=%s",
                                     version_id, row.window_index)
                    counters["classifier_errors"] += 1
                    record_failure(row)
                    consecutive_errors += 1
                    if consecutive_errors >= CONSECUTIVE_ERRORS_ABORT:
                        skipped["aborted"] += 1
                        if done:
                            PIIScanShadowResult.objects.bulk_update(done, _DONE_FIELDS)
                        return summary(run)
                    continue
                consecutive_errors = 0
                row.classifier_categories = sorted(categories)
                windows_left -= 1
                counters["windows"] += 1
            event = prod_events.get(row.window_index)
            row.prod_candidates = list((event or {}).get("candidate_categories") or [])
            row.prod_action = str((event or {}).get("action") or "")
            row.prod_article_9 = event.get("article_9") if event else None
            row.prod_article_10 = event.get("article_10") if event else None
            row.prod_version_categories = prod_categories
            row.trace_id = trace_id
            row.state = "done"
            done.append(row)
            if row.shadow_note.startswith("prod_legacy"):
                continue  # production already was the legacy classifier
            probabilities = row.quick_probabilities or {}
            legacy_gated = [c for c in GATED_CATEGORIES if c in row.classifier_categories]
            counters["legacy_gated_windows"] += 1 if legacy_gated else 0
            counters["prod_gated_windows"] += 1 if row.prod_candidates else 0
            # Production's review is already blind, so only legacy-only gated positives
            # the version was never tagged with need a second opinion.
            to_review = [
                c for c in legacy_gated
                if (probabilities.get(c) or 0.0) < PII_FLAG_THRESHOLD
                and c not in row.prod_candidates and c not in prod_categories
            ]
            if not to_review:
                continue
            if reviews_left <= 0:
                row.blind_review = "not_reviewed"
                row.blind_candidates = to_review
                counters["reviews_capped"] += 1
                continue
            reviews_left -= 1
            counters["reviews"] += 1
            row.blind_review, row.blind_article_9, row.blind_article_10, row.blind_confidence = blind_review(
                text, to_review, document_title=document_title, org_id=org_id, context=context,
            )
            row.blind_candidates = to_review
            if row.blind_review == "reviewed":
                confirmed = (row.blind_article_9 and "pii_special_category" in to_review) or (
                    row.blind_article_10 and "pii_criminal_offence" in to_review)
                counters["blind_confirmed" if confirmed else "blind_dismissed"] += 1
            else:
                counters["blind_error"] += 1
        if done:
            PIIScanShadowResult.objects.bulk_update(done, _DONE_FIELDS)

    return summary(run)


__all__ = [
    "DIRECTION",
    "SHADOW_FEATURE_KEY",
    "blind_review",
    "pending_rows",
    "production_version_categories",
    "production_window_events",
    "record_window_verdict",
    "run_pii_shadow_for_day",
]
