"""Guardrail orchestrator: ties heuristic, classifier, and reviewer layers together.

Entry point for the chat consumer to check user messages before streaming.

Public API:
- check_heuristics() — instant Layer 0 scan, used as a blocking gate
- run_classifier_pipeline() — Layers 1+2 (classifier + reviewer), run in parallel with LLM stream
- check_user_message() — sequential wrapper calling both (backward compat / tests)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.utils import timezone

from guardrails.schemas import ClassifierResult, HeuristicResult, ReviewerDecision

logger = logging.getLogger(__name__)

# Actions that warrant cancelling an in-progress LLM stream.
STREAM_INTERCEPT_ACTIONS = frozenset({"block", "suspend"})


@dataclass
class GuardrailVerdict:
    """Result of the full guardrail check pipeline."""

    action: str = "allow"  # allow | warn | block | suspend
    message: str = ""
    heuristic_result: HeuristicResult | None = None
    classifier_result: ClassifierResult | None = None
    reviewer_decision: ReviewerDecision | None = None
    events: list = field(default_factory=list)  # list of created GuardrailEvent records


async def check_heuristics(
    text: str,
    user,
    thread_id: str | None = None,
    org_id: int | None = None,
) -> GuardrailVerdict:
    """Layer 0: instant heuristic scan.

    Returns a GuardrailVerdict with action="block" (high-confidence match) or
    action="allow" (clean or merely suspicious — escalate to classifier).
    """
    from guardrails.heuristics import heuristic_scan

    verdict = GuardrailVerdict()
    heuristic_result = heuristic_scan(text)
    verdict.heuristic_result = heuristic_result

    if heuristic_result.should_block:
        event = await _create_event_async(
            user=user,
            org_id=org_id,
            thread_id=thread_id,
            trigger_source="user_message",
            check_type="heuristic",
            tags=heuristic_result.tags,
            confidence=heuristic_result.confidence,
            severity="high",
            action_taken="blocked",
            raw_input=text[:2000],
        )
        verdict.action = "block"
        verdict.message = "Your message was blocked by our content safety system. Please rephrase your request."
        verdict.events.append(event)
        logger.info(
            "guardrail: heuristic block user_id=%s tags=%s confidence=%.2f",
            user.pk, heuristic_result.tags, heuristic_result.confidence,
        )
        return verdict

    if heuristic_result.is_suspicious:
        event = await _create_event_async(
            user=user,
            org_id=org_id,
            thread_id=thread_id,
            trigger_source="user_message",
            check_type="heuristic",
            tags=heuristic_result.tags,
            confidence=heuristic_result.confidence,
            severity="low",
            action_taken="escalated",
            raw_input=text[:2000],
        )
        verdict.events.append(event)

    return verdict


async def run_classifier_pipeline(
    text: str,
    user,
    heuristic_result: HeuristicResult,
    thread_id: str | None = None,
    org_id: int | None = None,
) -> GuardrailVerdict:
    """Layers 1+2: classifier + reviewer pipeline.

    No timeout — runs to completion and always creates GuardrailEvent records.
    Designed to run concurrently with the LLM stream.
    """
    verdict = GuardrailVerdict()
    verdict.heuristic_result = heuristic_result

    # --- Layer 1: Cheap model classifier ---
    try:
        from guardrails.classifier import classify_message

        classifier_result = await classify_message(
            text=text,
            user_id=user.pk,
            org_id=org_id,
            conversation_id=thread_id,
        )
        verdict.classifier_result = classifier_result
    except Exception:
        logger.exception("guardrail: classifier error for user_id=%s, defaulting to allow", user.pk)
        return verdict

    if not classifier_result.is_suspicious:
        if heuristic_result.is_suspicious:
            await _create_event_async(
                user=user,
                org_id=org_id,
                thread_id=thread_id,
                trigger_source="user_message",
                check_type="classifier",
                tags=classifier_result.concern_tags,
                confidence=classifier_result.confidence,
                severity="low",
                action_taken="dismissed",
                raw_input=text[:2000],
                reviewer_output=classifier_result.reasoning,
            )
        return verdict

    # Classifier flagged — log escalation event
    classifier_event = await _create_event_async(
        user=user,
        org_id=org_id,
        thread_id=thread_id,
        trigger_source="user_message",
        check_type="classifier",
        tags=classifier_result.concern_tags,
        confidence=classifier_result.confidence,
        severity="medium",
        action_taken="escalated",
        raw_input=text[:2000],
        reviewer_output=classifier_result.reasoning,
    )
    verdict.events.append(classifier_event)

    # --- Layer 2: Top model reviewer ---
    try:
        from guardrails.reviewer import review_flagged_message

        reviewer_decision = await review_flagged_message(
            text=text,
            classifier_result=classifier_result,
            user_id=user.pk,
            org_id=org_id,
            conversation_id=thread_id,
        )
        verdict.reviewer_decision = reviewer_decision
    except Exception:
        logger.exception("guardrail: reviewer error for user_id=%s, defaulting to block", user.pk)
        verdict.action = "block"
        verdict.message = "Your message has been flagged for review. Please contact your system administrator."
        return verdict

    # Apply reviewer decision
    verdict.action = reviewer_decision.action
    verdict.message = reviewer_decision.user_message

    review_event = await _create_event_async(
        user=user,
        org_id=org_id,
        thread_id=thread_id,
        trigger_source="user_message",
        check_type="llm_review",
        tags=classifier_result.concern_tags,
        confidence=classifier_result.confidence,
        severity=reviewer_decision.severity,
        action_taken=_map_action(reviewer_decision.action),
        raw_input=text[:2000],
        reviewer_output=reviewer_decision.reasoning,
        related_event=classifier_event,
    )
    verdict.events.append(review_event)

    # Handle suspension
    if reviewer_decision.action == "suspend":
        await _suspend_user_async(user, org_id, reviewer_decision.reasoning)

    logger.info(
        "guardrail: review complete user_id=%s action=%s severity=%s",
        user.pk, reviewer_decision.action, reviewer_decision.severity,
    )

    return verdict


async def check_user_message(
    text: str,
    user,
    thread_id: str | None = None,
    org_id: int | None = None,
) -> GuardrailVerdict:
    """Run the full guardrail pipeline sequentially (backward compat).

    Layer 0: Heuristic scan (sync, ~0ms)
    Layer 1+2: Classifier + reviewer

    Returns a GuardrailVerdict with the final action and any created events.
    """
    heuristic_verdict = await check_heuristics(text, user, thread_id, org_id)
    if heuristic_verdict.action == "block":
        return heuristic_verdict

    pipeline_verdict = await run_classifier_pipeline(
        text, user, heuristic_verdict.heuristic_result, thread_id, org_id,
    )
    # Merge heuristic events into pipeline verdict
    pipeline_verdict.events = heuristic_verdict.events + pipeline_verdict.events
    return pipeline_verdict


def _map_action(action: str) -> str:
    """Map reviewer action to GuardrailEvent.ActionTaken value."""
    mapping = {
        "dismiss": "dismissed",
        "warn": "warned",
        "block": "blocked",
        "suspend": "suspended",
    }
    return mapping.get(action, "logged")


def _create_event_sync(
    user,
    org_id: int | None,
    thread_id: str | None,
    trigger_source: str,
    check_type: str,
    tags: list[str],
    confidence: float,
    severity: str,
    action_taken: str,
    raw_input: str,
    reviewer_output: str | None = None,
    related_event=None,
    llm_call_log=None,
):
    """Create a GuardrailEvent synchronously."""
    from guardrails.models import GuardrailEvent

    return GuardrailEvent.objects.create(
        user=user,
        organization_id=org_id,
        thread_id=thread_id,
        llm_call_log=llm_call_log,
        trigger_source=trigger_source,
        check_type=check_type,
        tags=tags,
        confidence=confidence,
        severity=severity,
        action_taken=action_taken,
        raw_input=raw_input,
        reviewer_output=reviewer_output,
        related_event=related_event,
    )


async def _create_event_async(**kwargs):
    """Create a GuardrailEvent from async context."""
    from asgiref.sync import sync_to_async
    return await sync_to_async(_create_event_sync)(**kwargs)


def _suspend_user_sync(user, org_id: int | None, reason: str):
    """Suspend the user's membership in the organization (sync)."""
    from accounts.models import Membership

    if org_id:
        Membership.objects.filter(
            user=user, org_id=org_id,
        ).update(
            is_suspended=True,
            suspended_at=timezone.now(),
            suspended_reason=reason[:2000],
        )
        logger.warning(
            "guardrail: suspended user_id=%s org_id=%s reason=%s",
            user.pk, org_id, reason[:200],
        )


async def _suspend_user_async(user, org_id: int | None, reason: str):
    """Suspend the user's membership from async context."""
    from asgiref.sync import sync_to_async
    await sync_to_async(_suspend_user_sync)(user, org_id, reason)
