from django.urls import path

from . import views

urlpatterns = [
    path("", views.inbox, name="inbox"),
    path("renew/", views.inbox_renew, name="inbox_renew"),
]
