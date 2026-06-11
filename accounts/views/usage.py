"""Usage pages: per-user and per-organization LLM spend breakdowns."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from django.contrib.auth.decorators import login_required
from django.db.models import Count, Sum
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_GET

from accounts.models import Membership
from accounts.views._helpers import org_admin_required


def _parse_date(value):
    """Parse a YYYY-MM-DD string, return a date or None."""
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class UsageWindow:
    """Resolved reporting window plus the month-navigation context."""

    start_date: date
    end_date: date  # inclusive display bound
    query_start: object  # aware datetime, inclusive
    query_end: object  # aware datetime, exclusive
    custom_range: bool
    display_month: date | None
    prev_month: date | None
    next_month: date | None


def resolve_usage_window(request) -> UsageWindow:
    """Resolve ?start/?end into a reporting window.

    Both params parse -> custom range (swapped if reversed, end inclusive).
    Otherwise month mode: ?start anchors the month, default current month;
    next-month navigation is suppressed for future months.
    """
    today = timezone.now().date()
    parsed_start = _parse_date(request.GET.get("start"))
    parsed_end = _parse_date(request.GET.get("end"))

    def _aware_midnight(d):
        return timezone.make_aware(
            timezone.datetime.combine(d, timezone.datetime.min.time())
        )

    if parsed_start and parsed_end:
        start_date, end_date = parsed_start, parsed_end
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        return UsageWindow(
            start_date=start_date,
            end_date=end_date,
            query_start=_aware_midnight(start_date),
            # Inclusive end date -> exclusive bound one day later.
            query_end=_aware_midnight(end_date + timedelta(days=1)),
            custom_range=True,
            display_month=None,
            prev_month=None,
            next_month=None,
        )

    start_date = (parsed_start or today).replace(day=1)
    if start_date.month == 12:
        next_month_first = start_date.replace(year=start_date.year + 1, month=1, day=1)
    else:
        next_month_first = start_date.replace(month=start_date.month + 1, day=1)
    return UsageWindow(
        start_date=start_date,
        end_date=next_month_first - timedelta(days=1),
        query_start=_aware_midnight(start_date),
        query_end=_aware_midnight(next_month_first),
        custom_range=False,
        display_month=start_date,
        prev_month=(start_date - timedelta(days=1)).replace(day=1),
        # Only show next if not in the future
        next_month=next_month_first if next_month_first <= today.replace(day=1) else None,
    )


def aggregate_usage_totals(qs) -> dict:
    """Aggregate cost/calls/tokens over an LLMCallLog queryset."""
    totals = qs.aggregate(
        total_cost=Sum("cost_usd"),
        total_calls=Count("id"),
        total_input_tokens=Sum("input_tokens"),
        total_output_tokens=Sum("output_tokens"),
    )
    totals["total_cost"] = totals["total_cost"] or Decimal("0")
    totals["total_input_tokens"] = totals["total_input_tokens"] or 0
    totals["total_output_tokens"] = totals["total_output_tokens"] or 0
    return totals


def _window_context(window: UsageWindow) -> dict:
    today = timezone.now().date()
    return {
        "start_date": window.start_date,
        "end_date": window.end_date,
        "custom_range": window.custom_range,
        "display_month": window.display_month,
        "prev_month": window.prev_month,
        "next_month": window.next_month,
        "today": today,
        "current_year": today.year,
    }


@login_required
@require_GET
def usage_page(request):
    from llm.models import LLMCallLog

    window = resolve_usage_window(request)
    qs = LLMCallLog.objects.filter(
        user=request.user,
        created_at__gte=window.query_start,
        created_at__lt=window.query_end,
    )

    model_breakdown = (
        qs.values("model")
        .annotate(
            cost=Sum("cost_usd"),
            calls=Count("id"),
            input_tokens=Sum("input_tokens"),
            output_tokens=Sum("output_tokens"),
        )
        .order_by("-cost")
    )

    return render(request, "accounts/usage.html", {
        **_window_context(window),
        "totals": aggregate_usage_totals(qs),
        "model_breakdown": model_breakdown,
    })


@login_required
@require_GET
@org_admin_required
def org_usage_page(request):
    from llm.models import LLMCallLog

    org = request.org_membership.org
    window = resolve_usage_window(request)

    user_ids = list(Membership.objects.filter(org=org).values_list("user_id", flat=True))
    qs = LLMCallLog.objects.filter(
        user_id__in=user_ids,
        created_at__gte=window.query_start,
        created_at__lt=window.query_end,
    )

    user_breakdown = (
        qs.values("user_id", "user__email", "user__first_name", "user__last_name")
        .annotate(
            cost=Sum("cost_usd"),
            calls=Count("id"),
            input_tokens=Sum("input_tokens"),
            output_tokens=Sum("output_tokens"),
        )
        .order_by("-cost")
    )

    return render(request, "accounts/org_usage.html", {
        **_window_context(window),
        "org": org,
        "totals": aggregate_usage_totals(qs),
        "user_breakdown": user_breakdown,
    })
