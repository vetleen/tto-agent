"""Nightly shadow of the chunk scan: the legacy chunk classifier against production.

Production (``guardrails.tasks._scan_chunks_for_version``) classifies every document
chunk through the Decisions service (``guardrails.decisions``) and sends each flag to
the Layer-2 reviewer. When a batch's verdicts are final it calls
:func:`record_chunk_verdicts`, which writes one *pending* ``ChunkScanShadowResult`` per
chunk (``direction="legacy_shadow"``): the Decisions probability, concern and pathway —
ids and numbers only. Every night :func:`run_shadow_for_day` fills in the other half:
it re-reads the pending chunks' text, runs the legacy batch classifier
(``guardrails.tasks._run_batch_classifier``) with the same batching and system prompt,
records production's reviewer outcome from the chunks' ``GuardrailEvent`` rows, and
gives ONE blind review — the same reviewer, a placeholder Layer-1 block — to each chunk
only the legacy classifier flagged (production's reviewer already judged every
Decisions positive with that same placeholder).

Side-effect free: it writes only ``ChunkScanShadowRun``/``ChunkScanShadowResult`` rows
(never chunk text) and the ``LLMCallLog`` rows of its own calls (grouped by the run's
trace_id). It never touches chunks, versions or ``GuardrailEvent`` rows. Kill switch
``GUARDRAIL_SHADOW_ENABLED`` (it also gates the pending-row writes); caps
``GUARDRAIL_SHADOW_MAX_CHUNKS``/``_MAX_REVIEWS``. Report: ``manage.py
shadow_chunk_scan_report``. Rows before 2026-11 (``direction="decisions_shadow"``) are
the opposite comparison: production = the legacy classifier, shadow = Decisions.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as dt_time

from django.conf import settings

from guardrails.decisions import REASONING_PLACEHOLDER_CHUNK
from guardrails.tasks import _batch_chunks_by_budget, _build_neighbor_context

logger = logging.getLogger(__name__)

SHADOW_FEATURE_KEY = "shadow_chunk_scan"
DIRECTION = "legacy_shadow"
SHADOW_REVIEW_PLACEHOLDER = REASONING_PLACEHOLDER_CHUNK
# Consecutive failed legacy calls after which the night is abandoned.
CONSECUTIVE_ERRORS_ABORT = 3


@dataclass(frozen=True)
class ShadowFlag:
    """Duck-types ``ChunkClassification`` for ``review_flagged_chunk``."""

    concern_tags: list[str]
    confidence: float
    reasoning: str = SHADOW_REVIEW_PLACEHOLDER


@dataclass
class ProdVerdict:
    """What production decided for a chunk, from its GuardrailEvents."""

    flagged: bool = False
    confidence: float | None = None
    tags: list = field(default_factory=list)
    # allow | quarantine | threshold_quarantine | threshold_logged | ""
    review: str = ""


# Own quick skip window for shadows that still call Decisions (removed once every
# shadow runs the legacy classifier instead).
SHADOW_SKIP_KEY = "decisions_quick_skip:shadow"
_availability = None
_service = None
_lock = threading.Lock()


def shadow_availability():
    """The shadow's own quick skip window (never the live one)."""
    global _availability
    if _availability is None:
        with _lock:
            if _availability is None:
                from llm.service.decisions_quick import QuickAvailability

                _availability = QuickAvailability(key=SHADOW_SKIP_KEY)
    return _availability


def get_shadow_decisions_service():
    """A DecisionsService bound to the shadow skip window."""
    global _service
    if _service is None:
        availability = shadow_availability()  # takes the same lock: resolve it first
        with _lock:
            if _service is None:
                from llm.service.decisions_service import DecisionsService

                _service = DecisionsService(availability=availability)
    return _service


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """[start, end) of ``day`` in the project time zone (Europe/Oslo)."""
    from django.utils import timezone

    start = timezone.make_aware(datetime.combine(day, dt_time.min))
    return start, start + timedelta(days=1)


# ---------------------------------------------------------------------------
# Production side: the pending half
# ---------------------------------------------------------------------------


def record_chunk_verdicts(*, version, chunks: list[dict], results: dict, org_id) -> None:
    """One pending shadow row per chunk of a finished production batch. Never raises.

    ``results`` maps ``chunk_index`` to the production verdict
    (``PrimaryChunkClassification``). A verdict the legacy classifier produced
    (Decisions failed) is recorded with ``shadow_note="prod_legacy:<error>"`` and no
    probability; the nightly run still compares it.
    """
    if not getattr(settings, "GUARDRAIL_SHADOW_ENABLED", False) or version is None:
        return
    try:
        from django.utils import timezone

        from guardrails.models import ChunkScanShadowResult

        today = timezone.localdate()
        rows = []
        for chunk in chunks:
            result = results.get(chunk["chunk_index"])
            if result is None:
                continue
            source = getattr(result, "source", "decisions")
            note = f"prod_legacy:{getattr(result, 'decide_error', '')}" if source == "legacy" else ""
            rows.append(ChunkScanShadowResult(
                run_date=today, org_id=org_id, version_id=version.id,
                chunk_id=chunk["id"], chunk_index=chunk["chunk_index"], chunk_chars=len(chunk["text"]),
                quick_probability=getattr(result, "probability", None) if source == "decisions" else None,
                quick_concern=(getattr(result, "concern", "") or "")[:32],
                quick_flagged=bool(result.is_suspicious),
                quick_pathway=(getattr(result, "pathway", "") or "")[:12],
                direction=DIRECTION, state="pending", shadow_note=note[:64],
            ))
        if rows:
            ChunkScanShadowResult.objects.bulk_create(rows, ignore_conflicts=True)
    except Exception:
        logger.warning("shadow_chunk_scan: recording production verdicts failed (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# Production verdicts and the blind review
# ---------------------------------------------------------------------------


def production_verdicts(chunk_ids: list[int]) -> dict[int, ProdVerdict]:
    """What production's classifier (and its reviewer) decided, from the chunks' events."""
    from guardrails.models import GuardrailEvent

    verdicts = {chunk_id: ProdVerdict() for chunk_id in chunk_ids}
    rows = (
        GuardrailEvent.objects.filter(
            trigger_source="document_chunk", chunk_id__in=chunk_ids,
            check_type__in=["classifier", "llm_review"],
        )
        .order_by("created_at")
        .values("chunk_id", "check_type", "action_taken", "confidence", "tags")
    )
    for row in rows:
        verdict = verdicts[row["chunk_id"]]
        if row["check_type"] == "classifier":
            if not verdict.flagged:
                # The first classifier event (the escalation when a reviewer ran)
                # carries the classifier's own confidence and tags.
                verdict.flagged = True
                verdict.confidence = row["confidence"]
                verdict.tags = list(row["tags"] or [])
            if row["action_taken"] == "blocked" and not verdict.review:
                verdict.review = "threshold_quarantine"
            elif row["action_taken"] == "logged" and not verdict.review:
                verdict.review = "threshold_logged"
        elif row["action_taken"] == "dismissed":
            verdict.review = "allow"
        elif row["action_taken"] == "blocked":
            verdict.review = "quarantine"
    return verdicts


def blind_review(*, chunk: dict, version, doc, org_id, flag: ShadowFlag, context):
    """The same reviewer, with the placeholder Layer-1 block.

    Returns ``(verdict, confidence, severity)``; verdict is the reviewer's action,
    ``"error"`` when it raised, ``"no_model"`` when no reviewer model resolves.
    """
    from guardrails.reviewer import review_flagged_chunk

    try:
        decision = review_flagged_chunk(
            chunk_text=chunk["text"],
            classifier_result=flag,
            document_title=getattr(doc, "display_name", "") or "",
            neighbor_context=_build_neighbor_context(version.id, chunk["chunk_index"]),
            org_id=org_id,
            user_id=None,
            context=context,
        )
    except Exception:
        logger.exception("shadow_chunk_scan: reviewer errored chunk_id=%s", chunk["id"])
        return "error", None, ""
    if decision is None:
        return "no_model", None, ""
    return decision.action, decision.confidence, decision.severity


# ---------------------------------------------------------------------------
# The nightly run
# ---------------------------------------------------------------------------


def pending_rows(day: date):
    """Pending rows recorded on or before ``day`` (older ones a capped night left)."""
    from guardrails.models import ChunkScanShadowResult

    return ChunkScanShadowResult.objects.filter(direction=DIRECTION, state="pending", run_date__lte=day)


_DONE_FIELDS = [
    "prod_review", "classifier_flagged", "classifier_confidence", "classifier_tags",
    "blind_review", "blind_review_source", "blind_review_confidence", "blind_review_severity",
    "state", "shadow_note", "trace_id",
]


def run_shadow_for_day(day: date, *, max_chunks: int | None = None, max_reviews: int | None = None) -> dict:
    """Fill in the legacy half of every pending row up to ``day``; returns the
    summary (also stored on the run row).

    Idempotent and resumable: only ``state="pending"`` rows are selected and each
    batch is saved as it finishes, so a retry or a manual ``--run`` continues where
    a partial night stopped.
    """
    from django.db.models import Count, Sum
    from django.db.utils import InterfaceError, OperationalError
    from django.utils import timezone

    from core.preferences import resolve_org_feature_model
    from documents.models import DataRoomDocumentChunk, DataRoomDocumentVersion
    from documents.services.pii_scan import org_id_for_document
    from guardrails.models import ChunkScanShadowResult, ChunkScanShadowRun
    from guardrails.tasks import _run_batch_classifier
    from llm.models import LLMCallLog
    from llm.types.context import RunContext

    t0 = time.perf_counter()
    trace_id = f"shadow-{day:%Y%m%d}-{uuid.uuid4().hex[:12]}"
    context = RunContext(trace_id=trace_id)
    log_metadata = {"kind": "shadow", "feature": SHADOW_FEATURE_KEY}
    chunks_left = max_chunks if max_chunks is not None else int(settings.GUARDRAIL_SHADOW_MAX_CHUNKS)
    reviews_left = max_reviews if max_reviews is not None else int(settings.GUARDRAIL_SHADOW_MAX_REVIEWS)
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
            "shadow_chunk_scan: " + " ".join(f"{k}={v}" for k, v in out.items() if k != "trace_id")
            + f" trace_id={trace_id}",
        )
        return out

    def mark_skipped(rows, note):
        for row in rows:
            row.state, row.shadow_note = "skipped", note[:64]
        ChunkScanShadowResult.objects.bulk_update(rows, ["state", "shadow_note"])
        skipped[note] += len(rows)

    try:
        run = ChunkScanShadowRun.objects.create(run_date=day, trace_id=trace_id)
        by_version: dict[int, list] = defaultdict(list)
        for row in pending_rows(day).order_by("version_id", "chunk_index"):
            by_version[row.version_id].append(row)
    except (OperationalError, InterfaceError):
        logger.info("shadow_chunk_scan: DB unavailable; will retry next tick.", exc_info=True)
        return {"day": day.isoformat(), "skipped": {"db_unavailable": 1}}

    consecutive_errors = 0
    for version_id, rows in by_version.items():
        if chunks_left <= 0:
            skipped["cap_chunks"] += len(rows)  # they stay pending for the next night
            continue
        version = DataRoomDocumentVersion.objects.select_related("document").filter(pk=version_id).first()
        if version is None:
            mark_skipped(rows, "version_gone")
            continue
        doc = version.document
        org_id = org_id_for_document(doc)
        cheap_model = resolve_org_feature_model(org_id, "guardrail_chunk_scan")
        if not cheap_model:
            mark_skipped(rows, "classifier_off")
            continue
        texts = dict(
            DataRoomDocumentChunk.objects.filter(pk__in=[r.chunk_id for r in rows]).values_list("id", "text")
        )
        gone = [r for r in rows if r.chunk_id not in texts]
        if gone:
            mark_skipped(gone, "chunk_gone")
        rows = [r for r in rows if r.chunk_id in texts]
        if len(rows) > chunks_left:
            skipped["cap_chunks"] += len(rows) - chunks_left
            rows = rows[:chunks_left]
        if not rows:
            continue
        versions_seen.add(version_id)
        orgs_seen.add(org_id)
        row_by_index = {r.chunk_index: r for r in rows}
        chunks = [{"id": r.chunk_id, "chunk_index": r.chunk_index, "text": texts[r.chunk_id]} for r in rows]

        for batch in _batch_chunks_by_budget(chunks):
            try:
                legacy = _run_batch_classifier(
                    doc, batch, cheap_model, org_id, context=context, log_metadata=log_metadata,
                )
            except Exception:
                logger.exception("shadow_chunk_scan: legacy classifier failed version_id=%s", version_id)
                counters["classifier_errors"] += 1
                consecutive_errors += 1
                if consecutive_errors >= CONSECUTIVE_ERRORS_ABORT:
                    skipped["aborted"] += 1
                    return summary(run)
                continue
            consecutive_errors = 0
            prod = production_verdicts([c["id"] for c in batch])
            done = []
            for chunk in batch:
                row = row_by_index[chunk["chunk_index"]]
                result = legacy.get(chunk["chunk_index"])
                row.prod_review = prod[chunk["id"]].review
                row.trace_id = trace_id
                done.append(row)
                counters["chunks"] += 1
                if result is None:
                    row.state, row.shadow_note = "skipped", "legacy_omitted"
                    skipped["legacy_omitted"] += 1
                    continue
                row.state = "done"
                row.classifier_flagged = bool(result.is_suspicious)
                row.classifier_confidence = result.confidence
                row.classifier_tags = list(result.concern_tags or [])
                if row.classifier_flagged and row.quick_flagged:
                    counters["both_positive"] += 1
                elif row.classifier_flagged:
                    counters["classifier_only"] += 1
                elif row.quick_flagged:
                    counters["quick_only"] += 1
                else:
                    counters["both_negative"] += 1
                if not (row.classifier_flagged and not row.quick_flagged):
                    continue
                # Only the legacy classifier flagged it: production's reviewer never saw it.
                row.blind_review_source = "classifier"
                if reviews_left <= 0:
                    row.blind_review = "not_reviewed"
                    counters["reviews_capped"] += 1
                    continue
                reviews_left -= 1
                counters["reviews"] += 1
                flag = ShadowFlag(concern_tags=list(row.classifier_tags), confidence=float(result.confidence or 0.0))
                row.blind_review, row.blind_review_confidence, row.blind_review_severity = blind_review(
                    chunk=chunk, version=version, doc=doc, org_id=org_id, flag=flag, context=context,
                )
                if row.blind_review in ("allow", "quarantine"):
                    counters[f"blind_{row.blind_review}_classifier"] += 1
                else:
                    counters["blind_error"] += 1
            ChunkScanShadowResult.objects.bulk_update(done, _DONE_FIELDS)
            chunks_left -= len(batch)

    return summary(run)


__all__ = [
    "DIRECTION",
    "SHADOW_FEATURE_KEY",
    "SHADOW_REVIEW_PLACEHOLDER",
    "ProdVerdict",
    "ShadowFlag",
    "blind_review",
    "day_bounds",
    "pending_rows",
    "production_verdicts",
    "record_chunk_verdicts",
    "run_shadow_for_day",
]
