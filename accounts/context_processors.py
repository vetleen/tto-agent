from .models import Membership, UserSettings


def theme(request):
    """Add current user theme and org admin flag to context."""
    context = {}
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
    return context
