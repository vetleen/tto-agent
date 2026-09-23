"""Usage pages: per-user and per-organization LLM spend breakdowns."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from django.contrib.auth.decorators import login_required
from django.db.models import Count, Sum
from django.db.models.functions import Coalesce
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_GET

from accounts.models import Membership
from accounts.views._helpers import org_admin_required


def _parse_date(value):
    """Parse a YYYY-MM-DD string, return a date or None.

    Dates outside a sane reporting range are treated as unparseable so the
    window falls back to the default month rather than crashing later date
    arithmetic (e.g. 9999-12-31 + 1 day -> OverflowError, or replace(year=10000)).
    """
    try:
        parsed = date.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if not (2000 <= parsed.year <= 2100):
        return None
    return parsed


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


_PROVIDER_LABELS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google_genai": "Google",
}


def _provider_label(model: str) -> str:
    """Human label for the provider behind a logged model string."""
    from llm.core.model_factory import detect_provider

    provider = detect_provider(model)
    if provider:
        return _PROVIDER_LABELS.get(provider, provider.replace("_", " ").title())
    if "/" in model:
        return model.split("/", 1)[0].title()
    return "Other"


# Stable color slot per provider (``wf-series-N`` in input.css) so a
# provider keeps its color across periods; anything else shares the last slot.
_PROVIDER_SERIES = {"Google": 1, "Anthropic": 2, "OpenAI": 3}
_OTHER_SERIES = 4


def spend_share(cost, total) -> dict:
    """Share of *total* spend as a bar width (percent) and a rounded label."""
    if not total or not cost:
        return {"pct": 0.0, "label": "0%"}
    pct = float(cost / total * 100)
    label = "<1%" if pct < 1 else f"{round(pct)}%"
    return {"pct": round(pct, 1), "label": label}


def build_provider_breakdown(qs, total_cost) -> dict:
    """Group per-model spend under its provider.

    Returns ``{"providers": [...], "models": [...]}``: providers sorted by
    total cost descending (each with its models, likewise sorted), and a flat
    list of every model sorted by cost across providers. Every entry carries
    its ``share`` of *total_cost* and its provider's ``series`` color slot.
    Model strings that normalise to the same display name (e.g.
    ``anthropic/claude-x`` vs bare ``claude-x``) are merged.
    """
    from llm.display import get_display_name

    rows = qs.values("model").annotate(
        cost=Coalesce(Sum("cost_usd"), Decimal("0")),
        calls=Count("id"),
        input_tokens=Coalesce(Sum("input_tokens"), 0),
        output_tokens=Coalesce(Sum("output_tokens"), 0),
    )

    providers: dict[str, dict] = {}
    for row in rows:
        label = _provider_label(row["model"])
        series = _PROVIDER_SERIES.get(label, _OTHER_SERIES)
        provider = providers.setdefault(label, {
            "name": label, "series": series, "cost": Decimal("0"), "calls": 0,
            "input_tokens": 0, "output_tokens": 0, "models": {},
        })
        display = get_display_name(row["model"])
        model = provider["models"].setdefault(display, {
            "name": display, "provider": label, "series": series,
            "cost": Decimal("0"), "calls": 0, "input_tokens": 0, "output_tokens": 0,
        })
        for bucket in (provider, model):
            bucket["cost"] += row["cost"]
            bucket["calls"] += row["calls"]
            bucket["input_tokens"] += row["input_tokens"]
            bucket["output_tokens"] += row["output_tokens"]

    def by_cost(entry):
        return (-entry["cost"], entry["name"])

    sorted_providers = sorted(providers.values(), key=by_cost)
    for provider in sorted_providers:
        provider["share"] = spend_share(provider["cost"], total_cost)
        provider["models"] = sorted(provider["models"].values(), key=by_cost)
        for model in provider["models"]:
            model["share"] = spend_share(model["cost"], total_cost)
    models = sorted(
        (m for p in sorted_providers for m in p["models"]), key=by_cost
    )
    return {"providers": sorted_providers, "models": models}


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
            # Coalesce so NULL-cost groups sort as 0 (Postgres orders NULLS
            # first on DESC, floating them above real spend) and render "$0".
            cost=Coalesce(Sum("cost_usd"), Decimal("0")),
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
            # Coalesce so NULL-cost groups sort as 0 rather than floating to the
            # top of the table on Postgres (NULLS FIRST on DESC).
            cost=Coalesce(Sum("cost_usd"), Decimal("0")),
            calls=Count("id"),
            input_tokens=Sum("input_tokens"),
            output_tokens=Sum("output_tokens"),
        )
        .order_by("-cost")
    )
    totals = aggregate_usage_totals(qs)
    user_breakdown = [
        {**row, "share": spend_share(row["cost"], totals["total_cost"])}
        for row in user_breakdown
    ]

    return render(request, "accounts/org_usage.html", {
        **_window_context(window),
        "org": org,
        "totals": totals,
        "user_breakdown": user_breakdown,
        "provider_breakdown": build_provider_breakdown(qs, totals["total_cost"]),
    })
