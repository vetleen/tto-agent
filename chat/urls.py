from django.urls import path

from . import views

urlpatterns = [
    path("", views.chat_home, name="chat_home"),
    path("threads/<uuid:thread_id>/delete/", views.thread_delete, name="thread_delete"),
    path("threads/<uuid:thread_id>/archive/", views.thread_archive, name="thread_archive"),
    path("api/data-rooms/", views.data_rooms_for_user, name="chat_data_rooms_api"),
]
