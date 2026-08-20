from django.conf import settings as django_settings

from .models import Membership, UserSettings, get_membership


def nav_context(request):
    """Shared navbar context: assistant name, theme, org-admin flag, budget status."""
    context = {
        "assistant_name": django_settings.ASSISTANT_NAME,
    }
    # `request.user` is absent when a request is rejected before
    # AuthenticationMiddleware runs (e.g. DisallowedHost); the error handlers
    # still render templates through this context processor, so guard it the
    # way Django's own auth context processor does.
    if hasattr(request, "user") and request.user.is_authenticated:
        # Read-only: the post_save signal creates UserSettings for every new
        # user, so a missing row just means defaults (no write per request).
        settings = UserSettings.objects.filter(user=request.user).first()
        prefs = (settings.preferences if settings else None) or {}
        context["theme"] = prefs.get("theme") or (settings.theme if settings else UserSettings.Theme.SYSTEM)
        # Org-admin flag (for the org settings link); shares the per-request
        # memoized membership with the suspension middleware and resolvers.
        membership = get_membership(request.user)
        context["user_is_org_admin"] = bool(
            membership and membership.role == Membership.Role.ADMIN
        )
        # Budget status for navbar progress bar (cached; display-only).
        from core.spend import get_cached_budget_status

        context["budget_status"] = get_cached_budget_status(request.user)

        # Unread-loop count for the navbar "Loops" badge: loops that produced a
        # result the owner hasn't opened yet.
        from django.db.models import F, Q

        from chat.models import Loop

        context["loops_unread_count"] = (
            Loop.objects.filter(created_by=request.user, last_result_at__isnull=False)
            .filter(Q(last_seen_at__isnull=True) | Q(last_result_at__gt=F("last_seen_at")))
            .count()
        )
    return context
