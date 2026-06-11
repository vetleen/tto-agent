"""Shared helpers for accounts views."""
from __future__ import annotations

from functools import wraps

from django.http import HttpResponseForbidden, JsonResponse

from accounts.models import Membership, get_membership


def get_admin_membership(user):
    """Return the user's admin membership (with org) or None.

    Uses the per-request memoized membership; a user has at most one
    membership (unique_membership_per_user), so filtering by role in Python
    is equivalent to filtering in SQL.
    """
    membership = get_membership(user)
    if membership and membership.role == Membership.Role.ADMIN:
        return membership
    return None


def org_admin_required(view_func):
    """Gate a view to org admins; attaches the membership as request.org_membership.

    Must run after login_required (it assumes an authenticated user). The 403
    is content-negotiated like accounts.views.auth.rate_limited: browser page
    loads (Accept: text/html) get the plain HTML 403, fetch() callers get JSON
    their error handlers surface via data.error.
    """

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        membership = get_admin_membership(request.user)
        if not membership:
            if "text/html" in (request.headers.get("Accept") or ""):
                return HttpResponseForbidden("Admin access required.")
            return JsonResponse({"error": "Admin access required."}, status=403)
        request.org_membership = membership
        return view_func(request, *args, **kwargs)

    return wrapper
