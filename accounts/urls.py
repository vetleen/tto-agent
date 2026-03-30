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
from .views.settings import (
    org_allowed_models_update,
    org_allowed_transcription_models_update,
    org_max_context_update,
    org_models_update,
    org_settings_page,
    org_skills_update,
    org_subagents_update,
    org_tools_update,
    org_transcription_model_update,
    preferences_max_context_update,
    preferences_models_update,
    preferences_transcription_model_update,
    settings_page,
    theme_update,
    usage_page,
)

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
    path("settings/", settings_page, name="settings"),
    path("settings/theme/", theme_update, name="theme_update"),
    path("settings/models/", preferences_models_update, name="preferences_models_update"),
    path("settings/max-context/", preferences_max_context_update, name="preferences_max_context_update"),
    path("settings/transcription-model/", preferences_transcription_model_update, name="preferences_transcription_model_update"),
    path("usage/", usage_page, name="usage"),
    path("org/settings/", org_settings_page, name="org_settings"),
    path("org/settings/allowed-models/", org_allowed_models_update, name="org_allowed_models_update"),
    path("org/settings/models/", org_models_update, name="org_models_update"),
    path("org/settings/tools/", org_tools_update, name="org_tools_update"),
    path("org/settings/skills/", org_skills_update, name="org_skills_update"),
    path("org/settings/subagents/", org_subagents_update, name="org_subagents_update"),
    path("org/settings/max-context/", org_max_context_update, name="org_max_context_update"),
    path("org/settings/allowed-transcription-models/", org_allowed_transcription_models_update, name="org_allowed_transcription_models_update"),
    path("org/settings/transcription-model/", org_transcription_model_update, name="org_transcription_model_update"),
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
