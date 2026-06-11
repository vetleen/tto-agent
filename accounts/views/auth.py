from django.conf import settings
from django.contrib.auth import get_user_model, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView as BaseLoginView
from django.contrib.auth.views import PasswordResetView as BasePasswordResetView
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
from django_ratelimit.decorators import ratelimit

from ..forms import SignUpForm
from ..verification import (
    can_resend_verification,
    send_verification_email,
    verify_token,
)

User = get_user_model()


# Two stacked throttles: per-IP (fast attacks from one source) and per-username
# (distributed credential stuffing against one account; the key falls back to the
# client IP when the field is empty — see core.ratelimit.login_username_or_ip).
# Accepted tradeoff: username keying lets an attacker deliberately exhaust a known
# account's bucket for the window; 30/h plus the generic 429 page (no reason
# disclosed) and automatic expiry is the mitigation.
@method_decorator(
    [
        ratelimit(key="ip", rate="5/m", method="POST", block=True),
        ratelimit(key="core.ratelimit.login_username_or_ip", rate="30/h", method="POST", block=True),
    ],
    name="post",
)
class LoginView(BaseLoginView):
    # The login page is a fixed light/paper composition; force the light theme
    # so the shared form/semantic tokens resolve to their light values.
    extra_context = {"force_light": True}

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(settings.LOGIN_REDIRECT_URL)
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        # "Keep me signed in": persistent session when checked, otherwise the
        # session expires when the browser closes.
        if not form.cleaned_data.get("remember_me"):
            self.request.session.set_expiry(0)
        else:
            self.request.session.set_expiry(None)
        return super().form_valid(form)


@method_decorator(ratelimit(key="ip", rate="3/h", method="POST", block=True), name="post")
class PasswordResetView(BasePasswordResetView):
    extra_email_context = {"assistant_name": settings.ASSISTANT_NAME}


def rate_limited(request, exception=None):
    # Browser form posts (login, admin login) send Accept: text/html and get the
    # branded page. fetch() callers — the JSON settings endpoints and the
    # multipart feedback widget (Accept: */*) — get JSON their error handlers
    # surface via data.error.
    if "text/html" in (request.headers.get("Accept") or ""):
        return render(request, "registration/rate_limited.html", status=429)
    return JsonResponse(
        {"error": "Too many requests. Please wait a few minutes and try again."},
        status=429,
    )


def index(request):
    if request.user.is_authenticated:
        return redirect(settings.LOGIN_REDIRECT_URL)
    return render(request, "index.html", {"landing_page": True})


@login_required
def suspended(request):
    from ..models import get_user_org

    org = get_user_org(request.user)
    return render(
        request,
        "registration/suspended.html",
        {"org_name": org.name if org else None},
        status=403,
    )


@login_required
def no_org(request):
    """Shown to authenticated users with no organization (see RequireOrgMiddleware)."""
    return render(request, "registration/no_org.html", status=403)


# SECURITY — the signup / email-verification flow below is intentionally DISABLED
# (no routes in accounts/urls.py). Before re-enabling public signup, add a login
# gate so unverified / inactive users cannot authenticate:
#   1. Override CustomAuthenticationForm.confirm_login_allowed() (accounts/forms.py)
#      to reject users when settings.EMAIL_VERIFICATION_REQUIRED and not
#      user.email_verified. (Django's base form already rejects is_active=False.)
#   2. Route verify_required / verify_email / resend_verification / signup.
#   3. Add a data migration backfilling email_verified=True for existing accounts
#      so the new gate does not lock anyone out (the field defaults to False).
#   4. Un-skip accounts/tests/test_verification.py (esp.
#      test_login_blocked_when_not_verified_redirects_to_verify_required).
#   5. Hash verification tokens at rest (EmailVerificationToken stores them in
#      plaintext and surfaces them read-only in the Django admin).
#   6. Convert verify_email() to a POST-confirm step: it currently logs the user
#      in on a GET with the token in the URL, which leaks into server/proxy logs
#      and browser history.
#   7. Add a view-level @ratelimit to resend_verification — the per-user
#      backoff in can_resend_verification only throttles known accounts; the
#      view itself accepts arbitrary POSTed emails.
#   8. Fix the account-enumeration oracle in resend_verification: unknown
#      email -> redirect to login, existing unverified email -> rendered
#      "sent" page. Responses must be indistinguishable.
# Note: verify_email() below calls login() directly, bypassing the form gate.
# verify_token() already rejects inactive users (accounts/verification.py); it
# must additionally respect the email_verified gate once routed.
def signup(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            if getattr(settings, "EMAIL_VERIFICATION_REQUIRED", True):
                send_verification_email(request, user, is_resend=False)
                request.session["verification_pending_email"] = user.email
                return redirect("accounts:verify_email_sent")
            login(request, user)
            return redirect("index")
    else:
        form = SignUpForm()
    return render(request, "registration/signup.html", {"form": form})


def verify_email_sent(request):
    email = request.session.get("verification_pending_email", "")
    return render(
        request,
        "registration/verify_email_sent.html",
        {"email": email},
    )


def verify_email(request, token):
    user, error = verify_token(token)
    if error:
        return render(
            request,
            "registration/verify_email_error.html",
            {"error": error},
        )
    login(request, user, backend="django.contrib.auth.backends.ModelBackend")
    if "verification_pending_email" in request.session:
        del request.session["verification_pending_email"]
    return redirect(settings.LOGIN_REDIRECT_URL)


@require_http_methods(["GET", "POST"])
def resend_verification(request):
    email = request.session.get("verification_pending_email") or (
        request.POST.get("email") if request.method == "POST" else None
    )
    if not email:
        return redirect("accounts:login")
    try:
        user = User.objects.get(email=email)
    except User.DoesNotExist:
        return redirect("accounts:login")
    if user.email_verified:
        return redirect("accounts:login")
    allowed, wait_seconds = can_resend_verification(user)
    if not allowed:
        return render(
            request,
            "registration/verify_email_sent.html",
            {
                "email": email,
                "resend_wait_seconds": wait_seconds,
                "resend_rate_limited": True,
            },
        )
    if request.method == "POST":
        token = send_verification_email(request, user, is_resend=True)
        if token is None:
            # Raced double-submit: another request consumed the resend slot
            # between our check and the locked re-check inside.
            allowed, wait_seconds = can_resend_verification(user)
            return render(
                request,
                "registration/verify_email_sent.html",
                {
                    "email": email,
                    "resend_wait_seconds": wait_seconds,
                    "resend_rate_limited": True,
                },
            )
        return render(
            request,
            "registration/verify_email_sent.html",
            {"email": email, "resend_sent": True},
        )
    return redirect("accounts:verify_email_sent")


def verify_required(request):
    """Shown when login is blocked because email is not verified."""
    email = request.session.get("verification_pending_email") or request.GET.get("email", "")
    return render(
        request,
        "registration/verify_required.html",
        {"email": email},
    )
