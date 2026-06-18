from django.contrib import admin
from .models import (
    DataRoom,
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentTag,
    DataRoomDocumentVersion,
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
    list_display = ("document", "version_index", "origin", "status", "is_searchable", "is_quarantined", "token_count", "created_by", "created_at", "processed_at")
    list_filter = ("origin", "status", "is_searchable", "is_quarantined")
    search_fields = ("document__original_filename", "document__name")
    raw_id_fields = ("document", "created_by")
    inlines = [DataRoomDocumentTagInline, DataRoomDocumentChunkInline]
    readonly_fields = ("created_at", "processed_at", "updated_at")


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
