from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from accounts.models import UserSettings


@login_required
@require_POST
def theme_update(request):
    theme_value = (request.POST.get("theme") or "").strip().lower()
    if theme_value not in (UserSettings.Theme.LIGHT, UserSettings.Theme.DARK):
        return JsonResponse({"error": "Invalid theme"}, status=400)
    settings, _ = UserSettings.objects.get_or_create(user=request.user)
    settings.theme = theme_value
    settings.save()
    return JsonResponse({"theme": settings.theme})
