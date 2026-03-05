from django.urls import path

from . import views

urlpatterns = [
    path("", views.data_room_list, name="data_room_list"),
    path("<uuid:data_room_id>/delete/", views.data_room_delete, name="data_room_delete"),
    path("<uuid:data_room_id>/rename/", views.data_room_rename, name="data_room_rename"),
    path("<uuid:data_room_id>/archive/", views.data_room_archive, name="data_room_archive"),
    path("<uuid:data_room_id>/documents/bulk-delete/", views.document_bulk_delete, name="document_bulk_delete"),
    path("<uuid:data_room_id>/documents/bulk-archive/", views.document_bulk_archive, name="document_bulk_archive"),
    path("<uuid:data_room_id>/documents/status/", views.document_status, name="document_status"),
    path("<uuid:data_room_id>/documents/upload/", views.document_upload, name="document_upload"),
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
    path("<uuid:data_room_id>/documents/", views.data_room_documents, name="data_room_documents"),
]
