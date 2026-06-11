import logging
import threading
import uuid


_thread_locals = threading.local()


def get_request_id() -> str:
    """Return the current request ID, or '-' if none is set."""
    return getattr(_thread_locals, "request_id", "-")


class RequestIDMiddleware:
    """Capture or generate a request ID for every HTTP request.

    Reads X-Request-ID from the incoming request (set by Heroku router),
    or generates a UUID4 if absent. Stores it in thread-local storage
    for the logging filter and echoes it back in the response header.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request_id = request.META.get("HTTP_X_REQUEST_ID") or str(uuid.uuid4())
        _thread_locals.request_id = request_id
        request.request_id = request_id
        try:
            import sentry_sdk as _sentry_sdk
            _sentry_sdk.set_tag("request_id", request_id)
        except ImportError:
            pass
        response = self.get_response(request)
        response["X-Request-ID"] = request_id
        _thread_locals.request_id = "-"
        return response


class SuspensionMiddleware:
    """Redirect suspended non-staff users to the suspended page.

    Runs once per HTTP request. Django ``is_staff``/superusers (platform
    operators) are exempt; org-level admins are not. Anonymous users pass
    through (handled by the normal login flow). Only the suspended page and
    logout are reachable while suspended, so the user can still leave.
    """

    # Only what a suspended user must still reach. NOT the whole /accounts/
    # prefix — settings/profile/org/usage live under /accounts/ and must be gated.
    EXEMPT_PATHS = ("/accounts/suspended/", "/accounts/logout/")
    EXEMPT_PREFIXES = ("/static/", "/media/")

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, "user", None)
        if (
            user is not None
            and user.is_authenticated
            and not user.is_staff
            and request.path not in self.EXEMPT_PATHS
            and not request.path.startswith(self.EXEMPT_PREFIXES)
        ):
            from accounts.models import Membership

            if Membership.objects.filter(user=user, is_suspended=True).exists():
                from django.shortcuts import redirect

                return redirect("accounts:suspended")
        return self.get_response(request)


class RequireOrgMiddleware:
    """Redirect authenticated users with no organization to the no-org page.

    Every active user must belong to exactly one organization: ``Membership`` has a
    unique constraint on ``user`` and org-scoped features (preferences, budget,
    guardrails, suspension) all resolve the org from it. A user with zero
    memberships has no functional account — those resolutions silently no-op — so
    gate them to an explanatory page until an administrator adds them to an org.

    Mirrors ``SuspensionMiddleware``: runs once per HTTP request, exempts Django
    staff/superusers (platform operators may have no org), and lets anonymous users
    through to the normal login flow. Must run AFTER ``SuspensionMiddleware`` — a
    suspended user is shown the suspended page; a no-org user can't be suspended
    (suspension lives on a membership they don't have).
    """

    EXEMPT_PATHS = ("/accounts/no-org/", "/accounts/logout/")
    EXEMPT_PREFIXES = ("/static/", "/media/")

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        from django.conf import settings

        user = getattr(request, "user", None)
        if (
            getattr(settings, "REQUIRE_ORG_MEMBERSHIP", True)
            and user is not None
            and user.is_authenticated
            and not user.is_staff
            and request.path not in self.EXEMPT_PATHS
            and not request.path.startswith(self.EXEMPT_PREFIXES)
        ):
            from accounts.models import Membership

            if not Membership.objects.filter(user=user).exists():
                from django.shortcuts import redirect

                return redirect("accounts:no_org")
        return self.get_response(request)


class RequestIDFilter(logging.Filter):
    """Inject request_id into every log record."""

    def filter(self, record):
        record.request_id = get_request_id()
        return True
