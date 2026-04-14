from django.urls import path

from . import views

urlpatterns = [
    path("", views.meeting_list, name="meeting_list"),
    path("create/", views.meeting_create, name="meeting_create"),
    path("<uuid:meeting_uuid>/", views.meeting_detail, name="meeting_detail"),
    path("<uuid:meeting_uuid>/rename/", views.meeting_rename, name="meeting_rename"),
    path("<uuid:meeting_uuid>/archive/", views.meeting_archive, name="meeting_archive"),
    path("<uuid:meeting_uuid>/delete/", views.meeting_delete, name="meeting_delete"),
    path("<uuid:meeting_uuid>/metadata/", views.meeting_update_metadata, name="meeting_update_metadata"),
    path(
        "<uuid:meeting_uuid>/transcription-progress/",
        views.meeting_transcription_progress,
        name="meeting_transcription_progress",
    ),
    path(
        "<uuid:meeting_uuid>/cancel-transcription/",
        views.meeting_cancel_transcription,
        name="meeting_cancel_transcription",
    ),
    path("<uuid:meeting_uuid>/link-data-room/", views.meeting_link_data_room, name="meeting_link_data_room"),
    path(
        "<uuid:meeting_uuid>/unlink-data-room/<uuid:data_room_uuid>/",
        views.meeting_unlink_data_room,
        name="meeting_unlink_data_room",
    ),
    path("<uuid:meeting_uuid>/upload/", views.meeting_upload, name="meeting_upload"),
    path("<uuid:meeting_uuid>/upload-transcript/", views.meeting_upload_transcript, name="meeting_upload_transcript"),
    path("<uuid:meeting_uuid>/upload-audio/", views.meeting_upload_audio, name="meeting_upload_audio"),
    path("<uuid:meeting_uuid>/save-to-data-room/", views.meeting_save_to_data_room, name="meeting_save_to_data_room"),
    path("<uuid:meeting_uuid>/attachments/upload/", views.meeting_upload_attachment, name="meeting_upload_attachment"),
    path(
        "<uuid:meeting_uuid>/attachments/<uuid:attachment_id>/delete/",
        views.meeting_delete_attachment,
        name="meeting_delete_attachment",
    ),
    path(
        "<uuid:meeting_uuid>/artifacts/<uuid:artifact_id>/delete/",
        views.meeting_delete_artifact,
        name="meeting_delete_artifact",
    ),
    path(
        "<uuid:meeting_uuid>/create-minutes-thread/",
        views.meeting_create_minutes_thread,
        name="meeting_create_minutes_thread",
    ),
]
