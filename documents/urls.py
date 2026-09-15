from django.urls import path

from . import views

urlpatterns = [
    path("", views.data_room_list, name="data_room_list"),
    path("<uuid:data_room_id>/delete/", views.data_room_delete, name="data_room_delete"),
    path("<uuid:data_room_id>/delete-check/", views.data_room_delete_check, name="data_room_delete_check"),
    path("<uuid:data_room_id>/rename/", views.data_room_rename, name="data_room_rename"),
    path("<uuid:data_room_id>/archive/", views.data_room_archive, name="data_room_archive"),
    path("<uuid:data_room_id>/documents/delete-check/", views.document_delete_check, name="document_delete_check"),
    path("<uuid:data_room_id>/documents/bulk-delete/", views.document_bulk_delete, name="document_bulk_delete"),
    path("<uuid:data_room_id>/documents/bulk-archive/", views.document_bulk_archive, name="document_bulk_archive"),
    path("<uuid:data_room_id>/documents/status/", views.document_status, name="document_status"),
    path("<uuid:data_room_id>/documents/upload/", views.document_upload, name="document_upload"),
    path("<uuid:data_room_id>/documents/duplicate-check/", views.document_duplicate_check, name="document_duplicate_check"),
    path(
        "<uuid:data_room_id>/documents/<int:document_id>/delete/",
        views.document_delete,
        name="document_delete",
    ),
    path(
        "<uuid:data_room_id>/documents/<int:document_id>/rename/",
        views.document_rename,
        name="document_rename",
    ),
    path(
        "<uuid:data_room_id>/documents/<int:document_id>/archive/",
        views.document_archive,
        name="document_archive",
    ),
    path(
        "<uuid:data_room_id>/documents/<int:document_id>/chunks/",
        views.document_chunks,
        name="document_chunks",
    ),
    path(
        "<uuid:data_room_id>/documents/<int:document_id>/file/",
        views.document_file,
        name="document_file",
    ),
    path(
        "<uuid:data_room_id>/documents/<int:document_id>/edit-source/",
        views.document_edit_source,
        name="document_edit_source",
    ),
    path(
        "<uuid:data_room_id>/documents/<int:document_id>/save/",
        views.document_save,
        name="document_save",
    ),
    path(
        "<uuid:data_room_id>/documents/<int:document_id>/versions/<int:version_id>/verdict/",
        views.document_version_verdict,
        name="document_version_verdict",
    ),
    path(
        "<uuid:data_room_id>/documents/<int:document_id>/rescan/",
        views.document_rescan,
        name="document_rescan",
    ),
    path("<uuid:data_room_id>/generate-description/", views.data_room_generate_description, name="data_room_generate_description"),
    path("<uuid:data_room_id>/description/", views.data_room_update_description, name="data_room_update_description"),
    path("<uuid:data_room_id>/documents/", views.data_room_documents, name="data_room_documents"),
]
