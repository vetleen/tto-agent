"""Shadow comparison of the text classifiers: the legacy classifier against production.

Four classifier families take their production verdict from the Decisions service
(``guardrails.classifier._classify_text``): chat messages, fetched web content, skill
content and the identity fields (profile/org description, SOUL, names). For each
production call, the call site hands the text and the production outcome (the
Decisions probability/concern/pathway and its reviewer's verdict, :class:`ProdOutcome`)
to :func:`run_classifier_shadow`, which runs the legacy cheap-model classifier
(``guardrails.classifier._run_classifier``, same system prompt) and records both
verdicts in a ``ClassifierShadowResult`` (``direction="legacy_shadow"``). A text only
the legacy classifier flagged gets ONE blind call to the family's own Layer-2 reviewer
with a placeholder Layer-1 block (tag + number, no written reasoning) — the same block
production's reviewer now sees for every Decisions positive, so false-positive rates
are comparable.

Where it runs: off the request path, as before. The web-content and skill scans
already run on the worker, so their shadow runs inline there. The chat classifier runs
on the web dyno and the identity classifiers in web views, so those call sites only
enqueue ``shadow_classifier_task``. A short chat message travels in the task args; a
long one is re-read from its ``ChatMessage`` row on the worker (skipped if it was
redacted meanwhile), so the broker never carries large payloads. When production
itself fell back to the legacy classifier there is nothing to compare: the row is
written ``state="skipped"`` with ``shadow_note="prod_legacy"`` and no call is made.

Side-effect free for production: never changes a verdict, never writes a
``GuardrailEvent``; only ``ClassifierShadowResult`` rows and its own ``LLMCallLog`` rows,
grouped per family and day by ``trace_id`` (``shadow-<family>-<YYYYMMDD>``). Rows hold
ids, a hash and numbers — never text, including user-authored names. Bounded by
per-family daily caps and a daily review cap (consumed only by comparisons that
actually run); off under ``manage.py test``. Never raises into a caller.
Report: ``manage.py classifier_shadow_report``. Rows before 2026-11
(``direction="decisions_shadow"``) are the opposite comparison.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime

from django.conf import settings
from django.utils import timezone

from guardrails import classifier as _classifier
from guardrails.decisions import REASONING_PLACEHOLDER_TEXT

logger = logging.getLogger(__name__)

DIRECTION = "legacy_shadow"
FAMILIES = ("chat_message", "web_content", "skill_content", "identity_field")
FAMILY_PROMPTS = {
    "chat_message": _classifier._CLASSIFIER_SYSTEM_PROMPT,
    "web_content": _classifier._WEB_CLASSIFIER_SYSTEM_PROMPT,
    "skill_content": _classifier._SKILL_CLASSIFIER_PROMPT,
}
# Which per-feature cheap model the family's legacy classifier resolves.
FAMILY_FEATURE_KEYS = {
    "chat_message": "guardrails_classifier",
    "identity_field": "guardrails_classifier",
    "web_content": "guardrail_web_scan",
    "skill_content": "guardrail_web_scan",
}
# The identity fields run three different prompts; ``subject["field"]`` picks one.
IDENTITY_PROMPTS = {
    "profile_name": _classifier._DESCRIPTION_CLASSIFIER_PROMPT,
    "profile_description": _classifier._DESCRIPTION_CLASSIFIER_PROMPT,
    "org_name": _classifier._DESCRIPTION_CLASSIFIER_PROMPT,
    "org_description": _classifier._ORG_DESCRIPTION_CLASSIFIER_PROMPT,
    "soul": _classifier._SOUL_CLASSIFIER_PROMPT,
    "org_soul": _classifier._SOUL_CLASSIFIER_PROMPT,
}
# The same cap the web scan and the skill scan apply to their own input.
MAX_SHADOW_TEXT_CHARS = 40_000
# A chat message longer than this is not carried through the broker; the worker
# re-reads it from the ChatMessage row instead (the broker Redis is small).
CHAT_QUEUED_TEXT_MAX_CHARS = 4_000
# The Layer-1 block the reviewer sees for a shadow-reviewed positive.
SHADOW_REVIEW_PLACEHOLDER = REASONING_PLACEHOLDER_TEXT


@dataclass(frozen=True)
class ProdOutcome:
    """What production decided: the Decisions verdict and its reviewer's.

    ``probability``/``concern``/``pathway`` are the Decisions answers (``None``/""
    when production fell back to the legacy classifier: ``source == "legacy"``).
    """

    flagged: bool
    confidence: float | None
    tags: list
    review: str  # the family's verdict string; "" when production did not flag
    probability: float | None = None
    concern: str = ""
    pathway: str = ""
    source: str = "decisions"
    decide_error: str = ""

    @classmethod
    def from_result(cls, result, review: str = "") -> "ProdOutcome":
        """From a production ``PrimaryClassifierResult`` (a plain ``ClassifierResult``
        counts as a legacy verdict)."""
        source = getattr(result, "source", "legacy")
        return cls(
            flagged=bool(result.is_suspicious),
            confidence=result.confidence,
            tags=list(result.concern_tags or []),
            review=review if result.is_suspicious else "",
            probability=getattr(result, "probability", None) if source == "decisions" else None,
            concern=str(getattr(result, "concern", "") or ""),
            pathway=str(getattr(result, "pathway", "") or ""),
            source=source,
            decide_error=str(getattr(result, "decide_error", "") or ""),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ProdOutcome":
        return cls(
            flagged=bool(data.get("flagged")),
            confidence=data.get("confidence"),
            tags=list(data.get("tags") or []),
            review=str(data.get("review") or ""),
            probability=data.get("probability"),
            concern=str(data.get("concern") or ""),
            pathway=str(data.get("pathway") or ""),
            source=str(data.get("source") or "decisions"),
            decide_error=str(data.get("decide_error") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "flagged": self.flagged, "confidence": self.confidence, "tags": list(self.tags),
            "review": self.review, "probability": self.probability, "concern": self.concern,
            "pathway": self.pathway, "source": self.source, "decide_error": self.decide_error,
        }


@dataclass(frozen=True)
class ShadowClassifierFlag:
    """Duck-types ``ClassifierResult`` for the reviewers (the chat reviewer also
    reads ``is_suspicious``)."""

    concern_tags: list
    confidence: float
    is_suspicious: bool = True
    reasoning: str = SHADOW_REVIEW_PLACEHOLDER


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def shadow_enabled(family: str) -> bool:
    return bool(getattr(settings, "CLASSIFIER_SHADOW_ENABLED", False)) and (
        family in (getattr(settings, "CLASSIFIER_SHADOW_FAMILIES", None) or [])
    )


def _daily_key(name: str, day: date) -> str:
    return f"classifier_shadow:{name}:{day:%Y%m%d}"


def take_daily_slot(name: str, cap: int) -> bool:
    """One slot of today's ``cap`` for ``name`` (a family, or ``reviews``).

    A cache failure fails open: a call is cheap and the kill switch remains.
    """
    from django.core.cache import cache

    key = _daily_key(name, timezone.localdate())
    try:
        cache.add(key, 0, timeout=2 * 86400)
        count = cache.incr(key)
    except Exception:
        logger.debug("classifier shadow: daily counter unavailable; allowing", exc_info=True)
        return True
    return count <= cap


def take_review_slot() -> bool:
    return take_daily_slot("reviews", int(getattr(settings, "CLASSIFIER_SHADOW_MAX_REVIEWS_PER_DAY", 30)))


def day_trace_id(family: str, day: date | None = None) -> str:
    """The ``LLMCallLog.trace_id`` of every call the shadow makes for ``family`` on ``day``."""
    return f"shadow-{family}-{(day or timezone.localdate()):%Y%m%d}"


def prompt_for(family: str, subject: dict | None) -> str | None:
    if family == "identity_field":
        return IDENTITY_PROMPTS.get(str((subject or {}).get("field") or ""))
    return FAMILY_PROMPTS.get(family)


# ---------------------------------------------------------------------------
# The blind review
# ---------------------------------------------------------------------------


def blind_review_for_family(
    family: str, text: str, flag: ShadowClassifierFlag, *, org_id, user_id, conversation_id,
    subject: dict | None, context, history_before: datetime | None, label: str = "",
) -> tuple[str, float | None, str]:
    """The family's own reviewer with the placeholder Layer-1 block.

    Returns ``(verdict, confidence, severity)``: the reviewer's action, ``"error"``
    when it raised, ``"no_model"`` when no reviewer model resolves, ``""`` for a
    family without a Layer-2 (identity fields). ``label`` is the skill reviewer's
    trusted label (shown in its prompt, never stored). Pure: the chat reviewer's
    sync core never suspends anyone — that happens in ``guardrails.service``.
    """
    from guardrails import reviewer

    subject = subject or {}
    try:
        if family == "web_content":
            decision = reviewer.review_flagged_web_content(
                content=text, classifier_result=flag,
                source_label=str(subject.get("source_label") or "web"),
                org_id=org_id, user_id=None, conversation_id=conversation_id, context=context,
            )
        elif family == "skill_content":
            decision = reviewer.review_flagged_skill_content(
                text, flag, label, org_id=org_id, user_id=None, context=context,
            )
        elif family == "chat_message":
            decision = reviewer.review_flagged_message_sync(
                text, flag, user_id, org_id, conversation_id,
                history_before=history_before, context=context,
            )
        else:
            return "", None, ""
    except Exception:
        logger.exception("classifier shadow: reviewer errored family=%s", family)
        return "error", None, ""
    if decision is None:
        return "no_model", None, ""
    return decision.action, decision.confidence, decision.severity


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def run_classifier_shadow(
    family: str, text: str, *, prod: ProdOutcome, org_id, user_id, conversation_id=None,
    subject: dict | None = None, system_prompt: str | None = None, history_before: datetime | None = None,
    review_label: str = "",
):
    """Compare one production classification with the legacy classifier and record it.

    Never raises. Returns the stored ``ClassifierShadowResult`` or ``None`` when the
    shadow skipped (disabled, empty text, no legacy model, daily cap) or could not be
    stored. ``subject`` must hold ids only — it is stored; ``review_label`` is shown
    to the skill reviewer, not stored.
    """
    try:
        return _run(
            family, text, prod=prod, org_id=org_id, user_id=user_id, conversation_id=conversation_id,
            subject=subject, system_prompt=system_prompt, history_before=history_before,
            review_label=review_label,
        )
    except Exception:
        logger.exception("classifier shadow: unexpected error family=%s", family)
        return None


def _row(family, text, *, prod, org_id, user_id, conversation_id, subject, trace_id, **fields):
    from guardrails.models import ClassifierShadowResult

    return ClassifierShadowResult.objects.create(
        family=family,
        org_id=org_id,
        user_id=int(user_id) if user_id is not None else None,
        conversation_id=conversation_id or "",
        subject=_json_safe(subject or {}),
        text_chars=len(text),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        prod_review=prod.review[:24],
        quick_probability=prod.probability,
        quick_concern=prod.concern[:32],
        quick_flagged=prod.flagged,
        quick_pathway=prod.pathway[:12],
        decide_error=prod.decide_error[:64],
        trace_id=trace_id,
        direction=DIRECTION,
        **fields,
    )


def _run(family, text, *, prod, org_id, user_id, conversation_id, subject, system_prompt, history_before, review_label):
    from core.preferences import resolve_org_feature_model
    from llm.types import RunContext

    if family not in FAMILIES or not shadow_enabled(family) or not text or not text.strip():
        return None
    system_prompt = system_prompt or prompt_for(family, subject)
    if not system_prompt:
        logger.warning("classifier shadow: no prompt for family=%s subject keys=%s", family, sorted((subject or {}).keys()))
        return None
    text = text[:MAX_SHADOW_TEXT_CHARS]
    conversation_id = str(conversation_id) if conversation_id else None
    common = dict(prod=prod, org_id=org_id, user_id=user_id, conversation_id=conversation_id, subject=subject)
    if prod.source != "decisions":
        # Production already used the legacy classifier: nothing to compare.
        return _row(family, text, trace_id="", state="skipped", shadow_note="prod_legacy",
                    classifier_flagged=prod.flagged, classifier_confidence=prod.confidence,
                    classifier_tags=list(prod.tags), **common)
    feature_key = FAMILY_FEATURE_KEYS[family]
    if not resolve_org_feature_model(org_id, feature_key):
        return None
    # Every gate above is free; only a comparison that will actually run consumes a slot.
    if not take_daily_slot(family, int(getattr(settings, "CLASSIFIER_SHADOW_MAX_PER_DAY", 2000))):
        logger.debug("classifier shadow: daily cap reached family=%s", family)
        return None

    context = RunContext(trace_id=day_trace_id(family), conversation_id=conversation_id)
    try:
        legacy = _classifier._run_classifier(
            text, user_id, conversation_id, system_prompt, org_id, feature_key,
            context=context, log_metadata={"kind": "shadow", "feature": f"shadow_{family}"},
        )
    except Exception as exc:
        logger.warning("classifier shadow: legacy classifier failed family=%s error=%s", family,
                       type(exc).__name__, exc_info=True)
        return _row(family, text, trace_id=context.trace_id, state="skipped",
                    shadow_note=f"classifier_error:{type(exc).__name__}"[:64], **common)

    blind_verdict, blind_source, blind_confidence, blind_severity = "", "", None, ""
    if legacy.is_suspicious and not prod.flagged and family != "identity_field":
        # Only the legacy classifier flagged it: production's reviewer never saw it.
        blind_source = "classifier"
        flag = ShadowClassifierFlag(list(legacy.concern_tags or []), float(legacy.confidence or 0.0))
        if not resolve_org_feature_model(org_id, "guardrails_reviewer"):
            blind_verdict = "no_model"  # nothing to call; consumes no review slot
        elif not take_review_slot():
            blind_verdict = "not_reviewed"
        else:
            blind_verdict, blind_confidence, blind_severity = blind_review_for_family(
                family, text, flag, org_id=org_id, user_id=user_id, conversation_id=conversation_id,
                subject=subject, context=context, history_before=history_before, label=review_label,
            )

    row = _row(
        family, text, trace_id=context.trace_id, state="done",
        classifier_flagged=bool(legacy.is_suspicious),
        classifier_confidence=legacy.confidence,
        classifier_tags=list(legacy.concern_tags or []),
        blind_review=blind_verdict[:24],
        blind_review_source=blind_source,
        blind_review_confidence=blind_confidence,
        blind_review_severity=(blind_severity or "")[:10],
        **common,
    )
    logger.info(
        "classifier shadow: family=%s decisions_p=%s flagged=%s prod_review=%s legacy_flagged=%s blind=%s row=%s",
        family, prod.probability, prod.flagged, prod.review or "-", legacy.is_suspicious,
        blind_verdict or "-", row.pk,
    )
    return row


def _json_safe(subject: dict) -> dict:
    return {str(k): (str(v) if not isinstance(v, (int, float, bool, type(None))) else v) for k, v in subject.items()}


# ---------------------------------------------------------------------------
# Call-site helpers for the web dyno (enqueue only)
# ---------------------------------------------------------------------------


def run_shadow_from_task(
    family, text, prod: dict, org_id, user_id, conversation_id, subject, system_prompt_key=None, started_at=None,
) -> str:
    """The worker side of ``shadow_classifier_task``.

    ``text`` is ``None`` for a long chat message: it is re-read from the ``ChatMessage``
    row named by ``subject["message_id"]`` and skipped when the row is gone or was
    redacted (a blocked message) in the meantime.
    """
    outcome = ProdOutcome.from_dict(prod or {})
    history_before = None
    if started_at:
        try:
            history_before = datetime.fromisoformat(started_at)
        except ValueError:
            history_before = None
    subject = dict(subject or {})
    if system_prompt_key and "field" not in subject:
        subject["field"] = system_prompt_key
    if text is None:
        text = _load_chat_message_text(subject.get("message_id"))
        if text is None:
            return "skipped"
    row = run_classifier_shadow(
        family, text, prod=outcome, org_id=org_id, user_id=user_id, conversation_id=conversation_id,
        subject=subject, history_before=history_before,
    )
    return "recorded" if row is not None else "skipped"


def _load_chat_message_text(message_id) -> str | None:
    if not message_id:
        return None
    from chat.models import ChatMessage

    row = ChatMessage.objects.filter(pk=message_id).values("content", "is_redacted").first()
    if row is None or row["is_redacted"] or not (row["content"] or "").strip():
        logger.debug("classifier shadow: chat message unavailable for the shadow (missing or redacted)")
        return None
    return row["content"]


def enqueue_chat_message_shadow(*, text, verdict, org_id, user_id, thread_id, message_id, started_at) -> None:
    """Queue the chat-message shadow from the consumer's guardrail pipeline.

    Skips when the pipeline produced no classifier verdict (classifier error) so the
    comparison only covers real production classifications. The production review
    outcome is the one ``run_classifier_pipeline`` recorded on the verdict. A short
    text travels with the task; a long one is re-read on the worker by message id.
    """
    if not shadow_enabled("chat_message") or not text or not text.strip():
        return
    classifier_result = getattr(verdict, "classifier_result", None)
    if classifier_result is None or "classifier_error" in (classifier_result.concern_tags or []):
        return
    prod = ProdOutcome.from_result(classifier_result, str(getattr(verdict, "review_outcome", "") or ""))
    from guardrails.tasks import shadow_classifier_task

    thread = str(thread_id) if thread_id else None
    queued_text = text[:MAX_SHADOW_TEXT_CHARS]
    if len(queued_text) > CHAT_QUEUED_TEXT_MAX_CHARS:
        if not message_id:
            return  # nothing to re-read on the worker
        queued_text = None
    shadow_classifier_task.delay(
        "chat_message", queued_text, prod.to_dict(), org_id, user_id, thread,
        {"message_id": str(message_id) if message_id else None, "thread_id": thread},
        None, started_at.isoformat() if started_at else None,
    )


def shadow_identity_field(field: str, text: str, *, user, org_id, result) -> None:
    """Queue the identity-field shadow from a settings view (agreement only; the
    production path has no Layer-2 — a flag blocks the save). Never raises."""
    try:
        if not shadow_enabled("identity_field") or not text or not text.strip() or result is None:
            return
        prod = ProdOutcome.from_result(result, "blocked")
        from guardrails.tasks import shadow_classifier_task

        shadow_classifier_task.delay(
            "identity_field", text[:MAX_SHADOW_TEXT_CHARS], prod.to_dict(), org_id,
            getattr(user, "pk", None), None, {"field": field}, field, None,
        )
    except Exception:
        logger.debug("classifier shadow: identity enqueue failed (non-fatal) field=%s", field, exc_info=True)
