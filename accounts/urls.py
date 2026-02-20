from django.contrib.auth import views as auth_views
from django.urls import include, path, reverse_lazy

from .forms import (
    CustomAuthenticationForm,
    CustomPasswordChangeForm,
    CustomPasswordResetForm,
    CustomSetPasswordForm,
)
from .views.auth import (
    LoginView,
    delete_account,
    resend_verification,
    signup,
    verify_email,
    verify_email_sent,
    verify_required,
)
from .views.settings import theme_update

app_name = "accounts"

urlpatterns = [
    path(
        "login/",
        LoginView.as_view(authentication_form=CustomAuthenticationForm),
        name="login",
    ),
    path("signup/", signup, name="signup"),
    path("verify-email/sent/", verify_email_sent, name="verify_email_sent"),
    path("verify-email/<str:token>/", verify_email, name="verify_email"),
    path("verify-email/resend/", resend_verification, name="resend_verification"),
    path("verify-required/", verify_required, name="verify_required"),
    path("settings/theme/", theme_update, name="theme_update"),
    path("delete/", delete_account, name="account_delete"),
    path(
        "password_change/",
        auth_views.PasswordChangeView.as_view(
            form_class=CustomPasswordChangeForm,
            success_url=reverse_lazy("accounts:password_change_done"),
        ),
        name="password_change",
    ),
    path(
        "password_reset/",
        auth_views.PasswordResetView.as_view(
            form_class=CustomPasswordResetForm,
            success_url=reverse_lazy("accounts:password_reset_done"),
        ),
        name="password_reset",
    ),
    path(
        "reset/<uidb64>/<token>/",
        auth_views.PasswordResetConfirmView.as_view(
            form_class=CustomSetPasswordForm,
            success_url=reverse_lazy("accounts:password_reset_complete"),
        ),
        name="password_reset_confirm",
    ),
    path("", include("django.contrib.auth.urls")),
]
