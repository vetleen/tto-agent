from django.conf import settings
from django.contrib.auth import get_user_model, login
from django.contrib.auth.views import LoginView as BaseLoginView
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from ..forms import SignUpForm
from ..verification import (
    can_resend_verification,
    send_verification_email,
    verify_token,
)

User = get_user_model()


class LoginView(BaseLoginView):
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(settings.LOGIN_REDIRECT_URL)
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        user = form.get_user()
        if getattr(settings, "EMAIL_VERIFICATION_REQUIRED", True) and not user.email_verified:
            self.request.session["verification_pending_email"] = user.email
            return redirect("accounts:verify_required")
        return super().form_valid(form)


def index(request):
    return render(request, "index.html", {"landing_page": True})


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


def delete_account(request):
    if not request.user.is_authenticated:
        return redirect("accounts:login")

    if request.method == "POST":
        request.user.delete()
        return redirect("index")

    return render(request, "registration/delete_account.html")
