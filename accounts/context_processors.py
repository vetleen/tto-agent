from django.conf import settings as django_settings

from .models import Membership, UserSettings


def nav_context(request):
    """Shared navbar context: assistant name, theme, org-admin flag, budget status."""
    context = {
        "assistant_name": django_settings.ASSISTANT_NAME,
    }
    if request.user.is_authenticated:
        settings, _ = UserSettings.objects.get_or_create(user=request.user)
        # Read from preferences JSON first, fall back to CharField
        prefs = settings.preferences or {}
        context["theme"] = prefs.get("theme") or settings.theme
        # Check if user is an org admin (for showing org settings link)
        context["user_is_org_admin"] = Membership.objects.filter(
            user=request.user,
            role=Membership.Role.ADMIN,
        ).exists()
        # Budget status for navbar progress bar
        from core.spend import get_budget_status

        context["budget_status"] = get_budget_status(request.user)
    return context
