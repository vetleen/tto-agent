from django.urls import path

from . import views

urlpatterns = [
    path("", views.project_list, name="project_list"),
    path("<uuid:project_id>/", views.project_detail_redirect, name="project_detail"),
    path("<uuid:project_id>/chat/", views.project_chat, name="project_chat"),
    path("<uuid:project_id>/delete/", views.project_delete, name="project_delete"),
    path("<uuid:project_id>/rename/", views.project_rename, name="project_rename"),
    path("<uuid:project_id>/documents/upload/", views.document_upload, name="document_upload"),
    path(
        "<uuid:project_id>/documents/<int:document_id>/delete/",
        views.document_delete,
        name="document_delete",
    ),
    path(
        "<uuid:project_id>/documents/<int:document_id>/rename/",
        views.document_rename,
        name="document_rename",
    ),
    path(
        "<uuid:project_id>/documents/<int:document_id>/chunks/",
        views.document_chunks,
        name="document_chunks",
    ),
    path("<uuid:project_id>/documents/", views.project_documents, name="project_documents"),
]
