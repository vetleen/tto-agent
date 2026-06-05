from django.conf import settings
from django.contrib.auth import get_user_model, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView as BaseLoginView
from django.contrib.auth.views import PasswordResetView as BasePasswordResetView
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


@method_decorator(ratelimit(key="ip", rate="5/m", method="POST", block=True), name="post")
class LoginView(BaseLoginView):
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(settings.LOGIN_REDIRECT_URL)
        return super().dispatch(request, *args, **kwargs)


@method_decorator(ratelimit(key="ip", rate="3/h", method="POST", block=True), name="post")
class PasswordResetView(BasePasswordResetView):
    pass


def rate_limited(request, exception=None):
    return render(request, "registration/rate_limited.html", status=429)


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
# Note: verify_email() below calls login() directly, bypassing the form gate — it
# must perform the same email_verified / is_active check once routed.
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
        send_verification_email(request, user, is_resend=True)
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
