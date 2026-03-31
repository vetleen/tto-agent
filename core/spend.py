"""Monthly spend tracking and budget enforcement."""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

from django.db.models import Sum
from django.utils import timezone

logger = logging.getLogger(__name__)


def get_month_boundaries(today: date | None = None):
    """Return (query_start_aware, query_end_aware, next_month_date) for the current calendar month."""
    if today is None:
        today = timezone.now().date()
    month_start = today.replace(day=1)
    if month_start.month == 12:
        next_month_start = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month_start = month_start.replace(month=month_start.month + 1)
    query_start = timezone.make_aware(
        timezone.datetime.combine(month_start, timezone.datetime.min.time())
    )
    query_end = timezone.make_aware(
        timezone.datetime.combine(next_month_start, timezone.datetime.min.time())
    )
    return query_start, query_end, next_month_start


def get_user_monthly_spend(user, query_start, query_end) -> Decimal:
    """Sum cost_usd for a single user in the given time window."""
    from llm.models import LLMCallLog

    result = LLMCallLog.objects.filter(
        user=user,
        created_at__gte=query_start,
        created_at__lt=query_end,
    ).aggregate(total=Sum("cost_usd"))
    return result["total"] or Decimal("0")


def get_org_monthly_spend(org_id, query_start, query_end) -> Decimal:
    """Sum cost_usd for all users in an org in the given time window."""
    from accounts.models import Membership
    from llm.models import LLMCallLog

    user_ids = list(
        Membership.objects.filter(org_id=org_id).values_list("user_id", flat=True)
    )
    if not user_ids:
        return Decimal("0")
    result = LLMCallLog.objects.filter(
        user_id__in=user_ids,
        created_at__gte=query_start,
        created_at__lt=query_end,
    ).aggregate(total=Sum("cost_usd"))
    return result["total"] or Decimal("0")


def get_budget_status(user) -> dict | None:
    """Return budget info for the given user, or None if no budget is configured.

    Returns a dict with keys: user_spend, user_budget, org_spend, org_budget,
    effective_spend, effective_budget, percentage, exceeded, exceeded_reason, reset_date.
    """
    from accounts.models import Membership

    membership = (
        Membership.objects.filter(user=user)
        .select_related("org")
        .first()
    )
    if not membership or not membership.org:
        return None

    org_prefs = membership.org.preferences or {}
    user_budget = Decimal(str(org_prefs.get("monthly_budget_per_user", 0) or 0))
    org_budget = Decimal(str(org_prefs.get("monthly_budget_org", 0) or 0))

    if user_budget == 0 and org_budget == 0:
        return None

    query_start, query_end, next_month_start = get_month_boundaries()

    user_spend = get_user_monthly_spend(user, query_start, query_end)

    org_spend = None
    if org_budget > 0:
        org_spend = get_org_monthly_spend(membership.org_id, query_start, query_end)

    # Determine exceeded state
    exceeded = False
    exceeded_reason = None
    if user_budget > 0 and user_spend >= user_budget:
        exceeded = True
        exceeded_reason = "user"
    if org_budget > 0 and org_spend is not None and org_spend >= org_budget:
        exceeded = True
        if exceeded_reason is None:
            exceeded_reason = "org"

    # For the progress bar, pick the tighter constraint (higher %)
    if user_budget > 0 and org_budget > 0 and org_spend is not None:
        user_pct = float(user_spend / user_budget * 100)
        org_pct = float(org_spend / org_budget * 100)
        if user_pct >= org_pct:
            effective_spend, effective_budget = user_spend, user_budget
        else:
            effective_spend, effective_budget = org_spend, org_budget
    elif user_budget > 0:
        effective_spend, effective_budget = user_spend, user_budget
    else:
        effective_spend, effective_budget = org_spend, org_budget

    percentage = int(min(float(effective_spend / effective_budget * 100), 100)) if effective_budget > 0 else 0

    return {
        "user_spend": user_spend,
        "user_budget": user_budget,
        "org_spend": org_spend,
        "org_budget": org_budget,
        "effective_spend": effective_spend,
        "effective_budget": effective_budget,
        "percentage": percentage,
        "exceeded": exceeded,
        "exceeded_reason": exceeded_reason,
        "reset_date": f"{next_month_start.strftime('%B')} {next_month_start.day}, {next_month_start.year}",
    }
