from django.contrib import admin
from .models import DataRoom, DataRoomDocument, DataRoomDocumentChunk, DataRoomDocumentTag


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


@admin.register(DataRoomDocument)
class DataRoomDocumentAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "data_room", "status", "token_count", "uploaded_by", "uploaded_at", "processed_at", "file_metadata_date", "document_date")
    list_filter = ("status", "uploaded_at")
    search_fields = ("original_filename",)
    raw_id_fields = ("data_room", "uploaded_by")
    inlines = [DataRoomDocumentTagInline, DataRoomDocumentChunkInline]
    readonly_fields = ("uploaded_at", "processed_at", "updated_at", "token_count", "file_metadata_date", "document_date")


@admin.register(DataRoomDocumentTag)
class DataRoomDocumentTagAdmin(admin.ModelAdmin):
    list_display = ("document", "key", "value", "created_at")
    list_filter = ("key",)
    search_fields = ("key", "value", "document__original_filename")
    raw_id_fields = ("document",)


@admin.register(DataRoomDocumentChunk)
class DataRoomDocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "chunk_index", "heading", "token_count", "created_at")
    list_filter = ("document__data_room",)
    search_fields = ("text", "heading")
    raw_id_fields = ("document",)
    ordering = ("document", "chunk_index")
