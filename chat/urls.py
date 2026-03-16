from django.urls import path

from . import views

urlpatterns = [
    path("", views.chat_home, name="chat_home"),
    path("threads/<uuid:thread_id>/delete/", views.thread_delete, name="thread_delete"),
    path("threads/<uuid:thread_id>/archive/", views.thread_archive, name="thread_archive"),
    path("threads/<uuid:thread_id>/canvas/export/", views.canvas_export, name="canvas_export"),
    path("threads/<uuid:thread_id>/canvas/<int:canvas_id>/export/", views.canvas_export, name="canvas_export_by_id"),
    path("threads/<uuid:thread_id>/canvas/import/", views.canvas_import, name="canvas_import"),
    path("threads/<uuid:thread_id>/canvas/<int:canvas_id>/import/", views.canvas_import, name="canvas_import_by_id"),
    path("api/threads/create/", views.thread_create, name="thread_create"),
    path("api/threads/<uuid:thread_id>/canvas/save-to-data-room/", views.canvas_save_to_data_room, name="canvas_save_to_data_room"),
    path("api/threads/<uuid:thread_id>/canvas/<int:canvas_id>/save-to-data-room/", views.canvas_save_to_data_room, name="canvas_save_to_data_room_by_id"),
    path("api/skills/", views.skills_for_user, name="chat_skills_api"),
    path("api/data-rooms/", views.data_rooms_for_user, name="chat_data_rooms_api"),
]
