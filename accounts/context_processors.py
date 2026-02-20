from .models import UserSettings


def theme(request):
    """Add current user theme to context for logged-in users."""
    context = {}
    if request.user.is_authenticated:
        settings, _ = UserSettings.objects.get_or_create(user=request.user)
        context["theme"] = settings.theme
    return context
