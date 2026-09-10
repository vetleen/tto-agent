from django.contrib import admin
from .models import (
    DataRoom,
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentTag,
    DataRoomDocumentVersion,
    PIIReviewEvent,
)


@admin.register(DataRoom)
class DataRoomAdmin(admin.ModelAdmin):
    list_display = ("uuid", "name", "slug", "created_by", "created_at", "updated_at")
    list_filter = ("created_at",)
    search_fields = ("name", "slug")
    raw_id_fields = ("created_by",)
    readonly_fields = ("uuid",)


class DataRoomDocumentTagInline(admin.TabularInline):
    model = DataRoomDocumentTag
    extra = 0
    readonly_fields = ("key", "value", "created_at")
    fields = ("key", "value", "created_at")


class DataRoomDocumentChunkInline(admin.TabularInline):
    model = DataRoomDocumentChunk
    extra = 0
    max_num = 20
    readonly_fields = ("chunk_index", "token_count", "created_at")
    fields = ("chunk_index", "heading", "text", "token_count", "source_page_start", "source_page_end", "created_at")
    ordering = ("chunk_index",)
    show_change_link = True


class DataRoomDocumentVersionInline(admin.TabularInline):
    model = DataRoomDocumentVersion
    extra = 0
    readonly_fields = ("version_index", "origin", "status", "is_searchable", "is_quarantined", "created_at")
    fields = ("version_index", "origin", "status", "is_searchable", "is_quarantined", "created_at")
    ordering = ("version_index",)
    show_change_link = True


@admin.register(DataRoomDocument)
class DataRoomDocumentAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "name", "data_room", "status", "token_count", "uploaded_by", "uploaded_at", "processed_at", "file_metadata_date", "document_date")
    list_filter = ("status", "uploaded_at")
    search_fields = ("original_filename", "name")
    raw_id_fields = ("data_room", "uploaded_by", "current_version", "active_searchable_version")
    inlines = [DataRoomDocumentVersionInline]
    readonly_fields = ("uploaded_at", "processed_at", "updated_at", "token_count", "file_metadata_date", "document_date")


@admin.register(DataRoomDocumentVersion)
class DataRoomDocumentVersionAdmin(admin.ModelAdmin):
    list_display = ("document", "version_index", "origin", "status", "parser_type", "is_searchable", "is_quarantined", "token_count", "created_by", "created_at", "processed_at")
    list_filter = ("origin", "status", "parser_type", "is_searchable", "is_quarantined")
    search_fields = ("document__original_filename", "document__name")
    raw_id_fields = ("document", "created_by")
    inlines = [DataRoomDocumentTagInline, DataRoomDocumentChunkInline]
    readonly_fields = ("created_at", "processed_at", "updated_at", "processing_metadata_pretty")
    # Derived artefact (e.g. the spreadsheet manifest: mesh geometry + paid
    # vision results) — display it, never let it be hand-edited.
    exclude = ("processing_metadata",)

    @admin.display(description="Processing metadata")
    def processing_metadata_pretty(self, obj):
        import json

        from django.utils.html import format_html

        if not obj.processing_metadata:
            return "—"
        return format_html(
            '<pre style="max-height: 400px; max-width: 80ch; overflow: auto; margin: 0;">{}</pre>',
            json.dumps(obj.processing_metadata, indent=2, ensure_ascii=False),
        )


@admin.register(DataRoomDocumentTag)
class DataRoomDocumentTagAdmin(admin.ModelAdmin):
    list_display = ("version", "key", "value", "created_at")
    list_filter = ("key",)
    search_fields = ("key", "value", "version__document__original_filename")
    raw_id_fields = ("version",)


@admin.register(DataRoomDocumentChunk)
class DataRoomDocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("version", "chunk_index", "heading", "token_count", "created_at")
    list_filter = ("version__document__data_room",)
    search_fields = ("text", "heading")
    raw_id_fields = ("version",)
    ordering = ("version", "chunk_index")


@admin.register(PIIReviewEvent)
class PIIReviewEventAdmin(admin.ModelAdmin):
    """Read-only review queue for PII reviewer hits and near-hits (calibration)."""

    list_display = (
        "created_at", "action", "article_9", "article_10", "confidence",
        "document_title", "org_id",
    )
    list_filter = ("action", "article_9", "article_10")
    search_fields = ("document_title", "findings", "reasoning", "excerpt")
    raw_id_fields = ("document", "data_room")
    readonly_fields = (
        "id", "created_at", "document", "data_room", "version_id", "user_id",
        "org_id", "window_index", "document_title", "candidate_categories",
        "action", "article_9", "article_10", "confidence", "reasoning",
        "findings", "excerpt", "retain_until",
    )
    ordering = ("-created_at",)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser
