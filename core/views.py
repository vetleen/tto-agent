from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import timedelta
from uuid import UUID

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from chat.models import ChatThread
from documents.models import DataRoom
from meetings.models import Meeting

from .retention import RETENTION_PERIODS

logger = logging.getLogger(__name__)


# ── Error handlers ──────────────────────────────────────────────────


def error_400(request, exception):
    return render(request, "errors/400.html", status=400)


def error_403(request, exception):
    return render(request, "errors/403.html", status=403)


def error_404(request, exception):
    return render(request, "errors/404.html", status=404)


def error_500(request):
    return render(request, "errors/500.html", status=500)


# ── Inbox / up-for-deletion queue ───────────────────────────────────

# Items whose retain_until falls within this window (or are already overdue but
# not yet purged by enforce_retention) surface in the inbox.
INBOX_WINDOW = timedelta(days=30)
INBOX_PAGE_SIZE = 50

# Date-sort options for the `sort` query param ("soonest" = ascending retain_until).
INBOX_SORTS = ("soonest", "latest")

# Registry driving both the listing query and the renew action so the three
# user-facing models are handled uniformly. id_field is the lookup column whose
# value is embedded in each row's composite "type:id" key (all three are UUIDs).
INBOX_TYPES = {
    "chat": {
        "model": ChatThread,
        "retention_key": "chat.ChatThread",
        "id_field": "id",
        "name_field": "title",
        "label": "Chat",
        "label_plural": "Chats",
    },
    "dataroom": {
        "model": DataRoom,
        "retention_key": "documents.DataRoom",
        "id_field": "uuid",
        "name_field": "name",
        "label": "Data Room",
        "label_plural": "Data Rooms",
    },
    "meeting": {
        "model": Meeting,
        "retention_key": "meetings.Meeting",
        "id_field": "uuid",
        "name_field": "name",
        "label": "Meeting",
        "label_plural": "Meetings",
    },
}


def _open_url(type_key, obj_id):
    """Build the link that opens the underlying object."""
    if type_key == "chat":
        return f"{reverse('chat_home')}?thread={obj_id}"
    if type_key == "dataroom":
        return reverse("data_room_documents", kwargs={"data_room_id": obj_id})
    if type_key == "meeting":
        return reverse("meeting_detail", kwargs={"meeting_uuid": obj_id})
    return "#"


def _relative_future(retain_until, now):
    """Return (countdown_label, absolute_date_str, is_overdue) in Europe/Oslo time."""
    local = timezone.localtime(retain_until)
    # Avoid %-d (not portable to Windows); build the date string explicitly.
    absolute = f"{local.strftime('%b')} {local.day}, {local.year}"
    if retain_until <= now:
        return "Today", absolute, True
    days = (local.date() - timezone.localtime(now).date()).days
    if days == 0:
        return "Today", absolute, False
    if days == 1:
        return "Tomorrow", absolute, False
    return f"in {days} days", absolute, False


@login_required
@require_http_methods(["GET"])
def inbox(request):
    """List the user's items closest to deletion, filtered/sorted per query params."""
    now = timezone.now()
    cutoff = now + INBOX_WINDOW
    show_archived = request.GET.get("show_archived") == "1"

    type_filter = request.GET.get("type", "all")
    if type_filter not in INBOX_TYPES:
        type_filter = "all"

    sort = request.GET.get("sort", "soonest")
    if sort not in INBOX_SORTS:
        sort = "soonest"

    def _base_qs(cfg):
        qs = cfg["model"].objects.filter(
            created_by=request.user,
            retain_until__isnull=False,
            retain_until__lte=cutoff,
        )
        if not show_archived:
            qs = qs.filter(is_archived=False)
        return qs

    # Per-type counts power the filter tabs; computed for every type regardless
    # of the active filter so each tab shows its own count.
    type_counts = {key: _base_qs(cfg).count() for key, cfg in INBOX_TYPES.items()}
    active_types = (
        INBOX_TYPES if type_filter == "all" else {type_filter: INBOX_TYPES[type_filter]}
    )

    items = []
    for type_key, cfg in active_types.items():
        qs = _base_qs(cfg).only(
            cfg["id_field"], cfg["name_field"], "retain_until", "is_archived"
        )
        for obj in qs:
            obj_id = str(getattr(obj, cfg["id_field"]))
            label, absolute, overdue = _relative_future(obj.retain_until, now)
            items.append({
                "key": f"{type_key}:{obj_id}",
                "type_label": cfg["label"],
                "name": getattr(obj, cfg["name_field"]) or "(untitled)",
                "retain_until": obj.retain_until,
                "countdown": label,
                "absolute": absolute,
                "is_overdue": overdue,
                "is_archived": obj.is_archived,
                "open_url": _open_url(type_key, obj_id),
            })

    # Ascending = soonest first (overdue floats to top); "latest" reverses it.
    items.sort(key=lambda d: d["retain_until"], reverse=(sort == "latest"))
    page_obj = Paginator(items, INBOX_PAGE_SIZE).get_page(request.GET.get("page"))

    # Company name for the retention-policy subtitle. A user has at most one
    # membership (unique constraint), so .first() is unambiguous.
    membership = request.user.organization_memberships.select_related("org").first()
    org_name = membership.org.name if membership else ""

    type_tabs = [{"key": "all", "label": "All", "count": sum(type_counts.values())}]
    type_tabs += [
        {"key": key, "label": cfg["label_plural"], "count": type_counts[key]}
        for key, cfg in INBOX_TYPES.items()
    ]

    return render(
        request,
        "core/inbox.html",
        {
            "page_obj": page_obj,
            "show_archived": show_archived,
            "org_name": org_name,
            "type_filter": type_filter,
            "sort": sort,
            "next_sort": "latest" if sort == "soonest" else "soonest",
            "type_tabs": type_tabs,
        },
    )


@login_required
@require_POST
def inbox_renew(request):
    """Reset retain_until to a full retention period for the given items.

    Body: {"items": ["chat:<id>", "dataroom:<uuid>", ...]}. Uses a direct
    .update() (mirroring the retention signals) so updated_at/auto_now is not
    bumped and post_save signals do not fire. Ownership is enforced by the
    created_by filter, so another user's id simply matches zero rows.
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    raw = body.get("items") if isinstance(body, dict) else None
    if not isinstance(raw, list) or not raw:
        return JsonResponse({"error": "items must be a non-empty list"}, status=400)

    grouped = defaultdict(list)
    for entry in raw:
        if not isinstance(entry, str) or ":" not in entry:
            continue
        type_key, _, obj_id = entry.partition(":")
        cfg = INBOX_TYPES.get(type_key)
        if cfg is None or not obj_id:
            continue
        try:
            UUID(obj_id)  # all three ids are UUIDs; guard the ORM against junk
        except (ValueError, AttributeError, TypeError):
            continue
        grouped[type_key].append(obj_id)

    if not grouped:
        return JsonResponse({"error": "No valid items"}, status=400)

    now = timezone.now()
    renewed = 0
    for type_key, ids in grouped.items():
        cfg = INBOX_TYPES[type_key]
        renewed += cfg["model"].objects.filter(
            **{f"{cfg['id_field']}__in": ids, "created_by": request.user},
        ).update(retain_until=now + RETENTION_PERIODS[cfg["retention_key"]])

    logger.info("Inbox renew by user %s: %d item(s) renewed.", request.user.id, renewed)
    return JsonResponse({"renewed": renewed})
