from django.contrib import admin
from .models import Project, ProjectDocument, ProjectDocumentChunk


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("uuid", "name", "slug", "created_by", "created_at", "updated_at")
    list_filter = ("created_at",)
    search_fields = ("name", "slug")
    raw_id_fields = ("created_by",)
    readonly_fields = ("uuid",)


class ProjectDocumentChunkInline(admin.TabularInline):
    model = ProjectDocumentChunk
    extra = 0
    readonly_fields = ("chunk_index", "token_count", "created_at")
    fields = ("chunk_index", "heading", "text", "token_count", "source_page_start", "source_page_end", "created_at")
    ordering = ("chunk_index",)
    show_change_link = True


@admin.register(ProjectDocument)
class ProjectDocumentAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "project", "status", "token_count", "uploaded_by", "uploaded_at", "processed_at")
    list_filter = ("status", "uploaded_at")
    search_fields = ("original_filename",)
    raw_id_fields = ("project", "uploaded_by")
    inlines = [ProjectDocumentChunkInline]
    readonly_fields = ("uploaded_at", "processed_at", "created_at", "updated_at", "token_count")


@admin.register(ProjectDocumentChunk)
class ProjectDocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "chunk_index", "heading", "token_count", "created_at")
    list_filter = ("document__project",)
    search_fields = ("text", "heading")
    raw_id_fields = ("document",)
    ordering = ("document", "chunk_index")
