"""Guardrail orchestrator: ties heuristic, classifier, and reviewer layers together.

Entry point for the chat consumer to check user messages before streaming.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from django.utils import timezone

from guardrails.schemas import ClassifierResult, HeuristicResult, ReviewerDecision

logger = logging.getLogger(__name__)

# Timeout for the cheap model classifier (seconds)
_CLASSIFIER_TIMEOUT = 5.0


@dataclass
class GuardrailVerdict:
    """Result of the full guardrail check pipeline."""

    action: str = "allow"  # allow | warn | block | suspend
    message: str = ""
    heuristic_result: HeuristicResult | None = None
    classifier_result: ClassifierResult | None = None
    reviewer_decision: ReviewerDecision | None = None
    events: list = field(default_factory=list)  # list of created GuardrailEvent records


async def check_user_message(
    text: str,
    user,  # accounts.models.User
    thread_id: str | None = None,
    org_id: int | None = None,
) -> GuardrailVerdict:
    """Run the full guardrail pipeline on a user message.

    Layer 0: Heuristic scan (sync, ~0ms)
    Layer 1: Cheap model classifier (async)
    Layer 2: Top model reviewer (only on escalation from Layer 1)

    Returns a GuardrailVerdict with the final action and any created events.
    """
    from guardrails.heuristics import heuristic_scan

    verdict = GuardrailVerdict()

    # --- Layer 0: Heuristic pre-filter ---
    heuristic_result = heuristic_scan(text)
    verdict.heuristic_result = heuristic_result

    if heuristic_result.should_block:
        # High-confidence heuristic match → block immediately
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
        # Log but don't block — escalate to classifier
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

    # --- Layer 1: Cheap model classifier ---
    try:
        from guardrails.classifier import classify_message

        classifier_result = await asyncio.wait_for(
            classify_message(
                text=text,
                user_id=user.pk,
                org_id=org_id,
                conversation_id=thread_id,
            ),
            timeout=_CLASSIFIER_TIMEOUT,
        )
        verdict.classifier_result = classifier_result
    except asyncio.TimeoutError:
        logger.warning("guardrail: classifier timed out for user_id=%s, defaulting to allow", user.pk)
        return verdict
    except Exception:
        logger.exception("guardrail: classifier error for user_id=%s, defaulting to allow", user.pk)
        return verdict

    if not classifier_result.is_suspicious:
        # Classifier says it's clean
        if heuristic_result.is_suspicious:
            # Log dismissed heuristic match for tuning
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
        verdict.message = "Your message could not be verified. Please try rephrasing."
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
