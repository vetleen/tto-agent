"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django_ratelimit.decorators import ratelimit

from accounts.views.auth import index

# /admin/login/ gets the same throttle as the main login form — staff accounts
# must not be the easiest ones to brute-force. Rebinding the bound method on the
# AdminSite instance wraps the view before admin.site.urls (below) snapshots it;
# GETs and the ?next= flow are untouched, and a blocked POST raises Ratelimited,
# which RatelimitMiddleware routes to RATELIMIT_VIEW (429 page).
admin.site.login = ratelimit(key="ip", rate="5/m", method="POST", block=True)(admin.site.login)

urlpatterns = [
    path("", index, name="index"),
    path("accounts/", include(("accounts.urls", "accounts"), namespace="accounts")),
    path("chat/", include("chat.urls")),
    path("skills/", include("agent_skills.urls")),
    path("data-rooms/", include("documents.urls")),
    path("meetings/", include("meetings.urls")),
    path("inbox/", include("core.urls")),
    path("admin/", admin.site.urls),
    path("api/feedback/", include("feedback.urls")),
]

handler400 = "core.views.error_400"
handler403 = "core.views.error_403"
handler404 = "core.views.error_404"
handler500 = "core.views.error_500"

if settings.DEBUG:
    urlpatterns = [
        path("__debug__/", include("debug_toolbar.urls")),
        path("__reload__/", include("django_browser_reload.urls")),
        *urlpatterns,
    ]
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
