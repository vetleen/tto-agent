"""Nightly shadow comparison of the chunk-scan guardrail against the decisions service.

Production (``guardrails.tasks._scan_chunks_for_version``) classifies every document
chunk with the cheap model and sends each flag to the Layer-2 reviewer. This module
re-runs the previous day's chunks through the decisions service's quick pathway
(OpenAI's Decisions endpoint) with the SAME batching, system prompt and concern
options, and compares:

* per chunk, the production verdict (from ``GuardrailEvent`` rows carrying
  ``chunk_id``) against the quick probability + concern;
* for every positive under either system, ONE blind call to the same reviewer with
  a neutral placeholder Layer-1 block (tag + number, no written reasoning), so the
  false-positive rate — the reviewer's "allow" share — is measured under identical
  conditions. The production reviewer's own verdict is recorded alongside.

Side-effect free: it writes only ``ChunkScanShadowRun``/``ChunkScanShadowResult``
rows (ids and numbers, never chunk text) and the ``LLMCallLog`` rows of its own calls
(grouped by the run's trace_id). It never touches chunks, versions or
``GuardrailEvent`` rows, and it uses its own decisions skip window so a nightly
failure never routes live traffic to the structured pathway. The chunk scan itself is
unchanged. Report: ``manage.py shadow_chunk_scan_report``.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as dt_time

from django.conf import settings

from guardrails.schemas import CONCERN_TAGS
from guardrails.tasks import (
    CHUNK_SCAN_SYSTEM_PROMPT,
    _MAX_CHUNK_CHARS,
    _batch_chunks_by_budget,
    _build_neighbor_context,
)
from llm.types.decisions import Choice, InputMessage, JsonPart, Question, TextPart

logger = logging.getLogger(__name__)

SHADOW_FEATURE_KEY = "shadow_chunk_scan"
# Quick probability at/above which a chunk counts as a positive (the report
# recomputes agreement at other thresholds from the stored probability).
SHADOW_FLAG_THRESHOLD = 0.5
# Own skip window: a failure here must never route live traffic to the structured
# pathway (llm.service.decisions_quick.QuickAvailability).
SHADOW_SKIP_KEY = "decisions_quick_skip:shadow"
# The Layer-1 block the reviewer sees for EVERY shadow-reviewed positive, whichever
# system flagged it — the production reviewer gets the classifier's written reasoning,
# the decisions pathway has none, so both are judged without it.
SHADOW_REVIEW_PLACEHOLDER = (
    "Flagged by a cheap probabilistic classifier that reports a probability only; no written "
    "reasoning is available. Judge the chunk on its own merits and make the final decision."
)
# Consecutive failed decide() calls after which the night is abandoned.
CONSECUTIVE_DECIDE_ERRORS_ABORT = 3
PATHWAY_QUICK = "quick"


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


@dataclass
class QuickVerdict:
    probability: float | None  # None = refusal
    concern: str
    pathway: str

    @property
    def flagged(self) -> bool:
        return self.probability is not None and self.probability >= SHADOW_FLAG_THRESHOLD


# ---------------------------------------------------------------------------
# Service singletons (tests patch get_shadow_decisions_service)
# ---------------------------------------------------------------------------

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
        # Resolve the availability BEFORE taking the lock: it takes the same
        # (non-reentrant) lock itself, so doing it inside would deadlock.
        availability = shadow_availability()
        with _lock:
            if _service is None:
                from llm.service.decisions_service import DecisionsService

                _service = DecisionsService(availability=availability)
    return _service


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """[start, end) of ``day`` in the project time zone (Europe/Oslo)."""
    from django.utils import timezone

    start = timezone.make_aware(datetime.combine(day, dt_time.min))
    return start, start + timedelta(days=1)


def has_legacy_events(day: date) -> bool:
    """Whether the day's chunk events predate ``chunk_id`` (the night after the
    deploy): production verdicts could not be matched, so the day is skipped."""
    from guardrails.models import GuardrailEvent

    start, end = day_bounds(day)
    return GuardrailEvent.objects.filter(
        trigger_source="document_chunk", created_at__gte=start, created_at__lt=end,
        chunk_id__isnull=True,
    ).exists()


def _eligible_chunks(day: date):
    """Chunks the classifier saw on ``day`` and the shadow has not yet recorded."""
    from documents.models import DataRoomDocumentChunk
    from guardrails.models import ChunkScanShadowResult, GuardrailEvent

    start, end = day_bounds(day)
    scan_state = DataRoomDocumentChunk.GuardrailScanState
    # Heuristic-blocked chunks never reached the classifier.
    heuristic_blocked = GuardrailEvent.objects.filter(
        trigger_source="document_chunk", check_type="heuristic", action_taken="blocked",
        chunk_id__isnull=False, created_at__gte=start,
    ).values("chunk_id")
    recorded = ChunkScanShadowResult.objects.filter(run_date=day).values("chunk_id")
    return (
        DataRoomDocumentChunk.objects.filter(
            created_at__gte=start, created_at__lt=end, guardrail_scan_state=scan_state.DONE,
        )
        .exclude(pk__in=heuristic_blocked)
        .exclude(pk__in=recorded)
    )


def select_version_ids(day: date) -> list[int]:
    return list(
        _eligible_chunks(day).order_by("version_id").values_list("version_id", flat=True).distinct()
    )


def select_chunks(day: date, version_id: int) -> list[dict]:
    return list(
        _eligible_chunks(day).filter(version_id=version_id)
        .order_by("chunk_index").values("id", "chunk_index", "text")
    )


# ---------------------------------------------------------------------------
# The decisions call
# ---------------------------------------------------------------------------


def build_questions(batch: list[dict]) -> list[Question]:
    """Two questions per chunk: is it adversarial, and which concern (or none)."""
    questions: list[Question] = []
    for chunk in batch:
        idx = chunk["chunk_index"]
        questions.append(Question(
            type="predicate",
            name=f"c{idx}_adversarial",
            instructions=(
                f"The chunk with chunk_index {idx} contains deliberate prompt injection or "
                "adversarial manipulation aimed at an AI system. Ordinary patent, legal, "
                "scientific, technical, financial or commercial content — including numbered "
                "or step-by-step instructions — does not count."
            ),
        ))
        questions.append(Question(
            type="choice",
            name=f"c{idx}_concern",
            instructions=(
                f"The main concern category for chunk_index {idx}; none if the chunk is not adversarial."
            ),
            choices=[Choice(value=tag) for tag in CONCERN_TAGS] + [Choice(value="none")],
        ))
    return questions


def build_input(batch: list[dict]) -> list[InputMessage]:
    """The classifier's system prompt as a developer message + the same JSON payload."""
    payload = [
        {"chunk_index": chunk["chunk_index"], "text": chunk["text"][:_MAX_CHUNK_CHARS]}
        for chunk in batch
    ]
    return [
        InputMessage(role="developer", content=[TextPart(text=CHUNK_SCAN_SYSTEM_PROMPT)]),
        InputMessage(role="user", content=[JsonPart(data=payload)]),
    ]


def decide_batch(service, batch: list[dict], *, org_id, context) -> dict[int, QuickVerdict]:
    """One decide() over a production-shaped batch → verdicts keyed by chunk_index."""
    result = service.decide(
        build_questions(batch), build_input(batch),
        feature_key=SHADOW_FEATURE_KEY, org_id=org_id, context=context,
    )
    verdicts: dict[int, QuickVerdict] = {}
    for chunk in batch:
        idx = chunk["chunk_index"]
        predicate = result[f"c{idx}_adversarial"]
        choice = result[f"c{idx}_concern"]
        verdicts[idx] = QuickVerdict(
            probability=predicate.probability if predicate.type == "predicate" else None,
            concern=choice.choice if choice.type == "choice" else "",
            pathway=result.pathway,
        )
    return verdicts


# ---------------------------------------------------------------------------
# Production verdicts and the blind review
# ---------------------------------------------------------------------------


def production_verdicts(chunk_ids: list[int]) -> dict[int, ProdVerdict]:
    """What the classifier (and its reviewer) decided, from the chunks' events."""
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
# The run
# ---------------------------------------------------------------------------


def run_shadow_for_day(day: date, *, max_chunks: int | None = None, max_reviews: int | None = None) -> dict:
    """Compare ``day``'s chunks; returns the summary (also stored on the run row).

    Idempotent: chunks already recorded for ``day`` are skipped, so a retry or a
    manual ``--run`` fills in what a partial night left out.
    """
    import os

    from django.db.models import Count, Sum
    from django.db.utils import InterfaceError, OperationalError
    from django.utils import timezone

    from documents.models import DataRoomDocumentVersion
    from documents.services.pii_scan import org_id_for_document
    from guardrails.models import ChunkScanShadowResult, ChunkScanShadowRun
    from llm.models import LLMCallLog
    from llm.types.context import RunContext

    t0 = time.perf_counter()
    trace_id = f"shadow-{day:%Y%m%d}-{uuid.uuid4().hex[:12]}"
    context = RunContext(trace_id=trace_id)
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
            "trace_id": trace_id,
            "versions": len(versions_seen),
            "orgs": len(orgs_seen),
            "skipped": dict(skipped),
            "duration_s": round(time.perf_counter() - t0, 1),
        })
        if run is not None:
            cost = LLMCallLog.objects.filter(trace_id=trace_id).aggregate(
                total=Sum("cost_usd"), calls=Count("id"),
            )
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

    try:
        run = ChunkScanShadowRun.objects.create(run_date=day, trace_id=trace_id)
        if not os.environ.get("OPENAI_API_KEY"):
            skipped["no_openai_key"] += 1
            return summary(run)
        if has_legacy_events(day):
            # The night after the deploy: that day's events were written without
            # chunk_id, so production verdicts cannot be matched.
            logger.warning("shadow_chunk_scan: day=%s has chunk events without chunk_id; skipping", day)
            skipped["legacy_events"] += 1
            return summary(run)
        version_ids = select_version_ids(day)
    except (OperationalError, InterfaceError):
        logger.info("shadow_chunk_scan: DB unavailable; will retry next tick.", exc_info=True)
        return {"day": day.isoformat(), "skipped": {"db_unavailable": 1}}

    from core.preferences import resolve_decisions_settings, resolve_org_feature_model

    service = get_shadow_decisions_service()
    org_skip: dict = {}  # org_id -> skip reason or None
    consecutive_errors = 0
    aborted = False

    for version_id in version_ids:
        if aborted:
            break
        if chunks_left <= 0:
            skipped["cap_chunks"] += len(select_chunks(day, version_id))
            continue
        version = (
            DataRoomDocumentVersion.objects.select_related("document").filter(pk=version_id).first()
        )
        if version is None:
            continue
        doc = version.document
        org_id = org_id_for_document(doc)
        if org_id not in org_skip:
            reason = None
            if not resolve_org_feature_model(org_id, "guardrail_chunk_scan"):
                reason = "classifier_off"
            else:
                prefs = resolve_decisions_settings(org_id)
                if not prefs.quick_enabled or not prefs.quick_model:
                    reason = "quick_off"
            org_skip[org_id] = reason
        chunks = select_chunks(day, version_id)
        if org_skip[org_id]:
            skipped[org_skip[org_id]] += len(chunks)
            continue
        if len(chunks) > chunks_left:
            skipped["cap_chunks"] += len(chunks) - chunks_left
            chunks = chunks[:chunks_left]
        versions_seen.add(version_id)
        orgs_seen.add(org_id)

        for batch in _batch_chunks_by_budget(chunks):
            if shadow_availability().is_suspended():
                skipped["quick_suspended"] += len(batch)
                continue
            try:
                quick = decide_batch(service, batch, org_id=org_id, context=context)
            except Exception:
                logger.exception(
                    "shadow_chunk_scan: decide failed version_id=%s chunks=%d", version_id, len(batch),
                )
                counters["decide_errors"] += 1
                consecutive_errors += 1
                if consecutive_errors >= CONSECUTIVE_DECIDE_ERRORS_ABORT:
                    skipped["aborted"] += 1
                    aborted = True
                    break
                continue
            consecutive_errors = 0
            prod = production_verdicts([c["id"] for c in batch])

            rows = []
            for chunk in batch:
                q = quick[chunk["chunk_index"]]
                p = prod[chunk["id"]]
                counters["chunks"] += 1
                if q.pathway != PATHWAY_QUICK:
                    counters["non_quick"] += 1
                if q.probability is None:
                    counters["refusals"] += 1
                if p.flagged and q.flagged:
                    counters["both_positive"] += 1
                elif p.flagged:
                    counters["classifier_only"] += 1
                elif q.flagged:
                    counters["quick_only"] += 1
                else:
                    counters["both_negative"] += 1

                blind, blind_conf, blind_sev, source = "", None, "", ""
                if p.flagged or q.flagged:
                    if p.flagged:
                        source = "classifier"
                        flag = ShadowFlag(concern_tags=list(p.tags), confidence=float(p.confidence or 0.0))
                    else:
                        source = "quick"
                        tags = [q.concern] if q.concern not in ("", "none") else []
                        flag = ShadowFlag(concern_tags=tags, confidence=float(q.probability or 0.0))
                    if reviews_left > 0:
                        reviews_left -= 1
                        counters["reviews"] += 1
                        blind, blind_conf, blind_sev = blind_review(
                            chunk=chunk, version=version, doc=doc, org_id=org_id, flag=flag, context=context,
                        )
                        if blind in ("allow", "quarantine"):
                            if p.flagged:
                                counters[f"blind_{blind}_classifier"] += 1
                            if q.flagged:
                                counters[f"blind_{blind}_quick"] += 1
                        else:
                            counters["blind_error"] += 1
                    else:
                        blind = "not_reviewed"
                        counters["reviews_capped"] += 1

                rows.append(ChunkScanShadowResult(
                    run_date=day, trace_id=trace_id, org_id=org_id,
                    version_id=version_id, chunk_id=chunk["id"], chunk_index=chunk["chunk_index"],
                    chunk_chars=len(chunk["text"]),
                    classifier_flagged=p.flagged, classifier_confidence=p.confidence,
                    classifier_tags=list(p.tags), prod_review=p.review,
                    quick_probability=q.probability, quick_concern=q.concern,
                    quick_flagged=q.flagged, quick_pathway=q.pathway,
                    blind_review=blind, blind_review_source=source,
                    blind_review_confidence=blind_conf, blind_review_severity=blind_sev,
                ))
            ChunkScanShadowResult.objects.bulk_create(rows, ignore_conflicts=True)
            chunks_left -= len(batch)

    return summary(run)


__all__ = [
    "SHADOW_FEATURE_KEY",
    "SHADOW_FLAG_THRESHOLD",
    "SHADOW_REVIEW_PLACEHOLDER",
    "SHADOW_SKIP_KEY",
    "ProdVerdict",
    "QuickVerdict",
    "ShadowFlag",
    "blind_review",
    "build_input",
    "build_questions",
    "day_bounds",
    "decide_batch",
    "get_shadow_decisions_service",
    "has_legacy_events",
    "production_verdicts",
    "run_shadow_for_day",
    "select_chunks",
    "select_version_ids",
    "shadow_availability",
]
