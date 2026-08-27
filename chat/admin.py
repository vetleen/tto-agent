from django.contrib import admin
from django.db.models import CharField, OuterRef, Subquery, Sum
from django.db.models.functions import Cast

from llm.admin import _pretty_json_html
from llm.models import LLMCallLog

from .models import (
    Asset,
    CanvasCheckpoint,
    ChatCanvas,
    ChatMessage,
    ChatThread,
    ChatThreadDataRoom,
    SlideComment,
    SlideRender,
    SlideRenderRun,
    SlideSet,
    SlideSetCheckpoint,
    SubAgentRun,
    ThreadChunkUsage,
    ThreadTask,
)


class SubAgentRunInline(admin.TabularInline):
    model = SubAgentRun
    extra = 0
    readonly_fields = ("id", "status", "prompt", "model_tier", "model_used", "tokens_used", "cost_usd", "created_at", "completed_at")
    fields = ("status", "prompt", "model_tier", "model_used", "tokens_used", "cost_usd", "created_at", "completed_at")


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ("id", "role", "content", "tool_call_id", "is_redacted", "created_at")
    fields = ("role", "content", "tool_call_id", "is_redacted", "created_at")


class ChatThreadDataRoomInline(admin.TabularInline):
    model = ChatThreadDataRoom
    extra = 0
    readonly_fields = ("attached_at",)
    raw_id_fields = ("data_room",)


class ThreadTaskInline(admin.TabularInline):
    model = ThreadTask
    extra = 0
    readonly_fields = ("id", "created_at", "updated_at")
    fields = ("order", "title", "status", "created_at", "updated_at")


class ThreadChunkUsageInline(admin.TabularInline):
    model = ThreadChunkUsage
    extra = 0
    readonly_fields = ("chunk", "document", "created_at")
    raw_id_fields = ("chunk", "document")


@admin.register(ChatThread)
class ChatThreadAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "created_by", "cost_usd", "created_at", "updated_at")
    list_filter = ("created_at",)
    search_fields = ("title",)
    readonly_fields = ("id", "created_at", "updated_at")
    inlines = [ChatThreadDataRoomInline, ThreadTaskInline, SubAgentRunInline, ThreadChunkUsageInline, ChatMessageInline]

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        cost_subquery = (
            LLMCallLog.objects.filter(conversation_id=Cast(OuterRef("id"), output_field=CharField()))
            .values("conversation_id")
            .annotate(total=Sum("cost_usd"))
            .values("total")
        )
        return qs.annotate(_cost_usd=Subquery(cost_subquery))

    @admin.display(description="Cost (USD)", ordering="_cost_usd")
    def cost_usd(self, obj):
        if obj._cost_usd is None:
            return "-"
        return f"${obj._cost_usd:.4f}"


class CanvasCheckpointInline(admin.TabularInline):
    model = CanvasCheckpoint
    extra = 0
    readonly_fields = ("source", "description", "order", "created_at")


@admin.register(ChatCanvas)
class ChatCanvasAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "title", "updated_at")
    inlines = [CanvasCheckpointInline]


class SlideSetCheckpointInline(admin.TabularInline):
    model = SlideSetCheckpoint
    extra = 0
    readonly_fields = ("source", "description", "order", "created_at")


class SlideCommentInline(admin.TabularInline):
    model = SlideComment
    extra = 0
    readonly_fields = ("slide_id", "author", "text", "status", "created_at")
    fields = ("slide_id", "author", "text", "status", "created_at")


@admin.register(SlideSet)
class SlideSetAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "title", "base_template", "is_active", "deleted_at", "updated_at")
    list_filter = ("is_active", "base_template", "created_at")
    search_fields = ("title", "id")
    readonly_fields = ("content_pretty", "created_at", "updated_at", "last_activated_at")
    # Raw ``content`` is excluded from the form; ``content_pretty`` shows the deck
    # JSON indented (the field is huge and hand-editing it in admin is unsafe).
    fields = (
        "thread", "title", "base_template", "is_active", "deleted_at",
        "content_pretty", "last_activated_at", "created_at", "updated_at",
    )
    raw_id_fields = ("thread",)
    list_select_related = ("thread",)
    inlines = [SlideSetCheckpointInline, SlideCommentInline]

    @admin.display(description="Content")
    def content_pretty(self, obj):
        return _pretty_json_html(obj.content)


@admin.register(SlideRenderRun)
class SlideRenderRunAdmin(admin.ModelAdmin):
    list_display = ("short_id", "slide_set", "purpose", "status", "created_at", "finished_at")
    list_filter = ("purpose", "status", "created_at")
    search_fields = ("id", "slide_set__title")
    readonly_fields = ("id", "created_at", "started_at", "finished_at")
    list_select_related = ("slide_set",)
    ordering = ["-created_at"]

    @admin.display(description="ID")
    def short_id(self, obj):
        return str(obj.id)[:8]


@admin.register(SlideRender)
class SlideRenderAdmin(admin.ModelAdmin):
    list_display = ("id", "slide_set", "slide_id", "content_hash", "width", "height", "rendered_at")
    search_fields = ("slide_id", "content_hash", "slide_set__title")
    readonly_fields = ("rendered_at",)
    list_select_related = ("slide_set",)
    raw_id_fields = ("asset",)


@admin.register(SubAgentRun)
class SubAgentRunAdmin(admin.ModelAdmin):
    list_display = ("short_id", "thread", "user", "status", "model_tier", "model_used", "short_prompt", "tokens_used", "cost_display", "has_result", "created_at")
    list_filter = ("status", "model_tier", "created_at")
    search_fields = ("prompt", "result", "id")
    readonly_fields = ("id", "created_at", "completed_at")
    list_select_related = ("thread", "user")
    ordering = ["-created_at"]

    @admin.display(description="ID")
    def short_id(self, obj):
        return str(obj.id)[:8]

    @admin.display(description="Prompt")
    def short_prompt(self, obj):
        return obj.prompt[:80] if obj.prompt else ""

    @admin.display(description="Cost", ordering="cost_usd")
    def cost_display(self, obj):
        return f"${obj.cost_usd:.4f}" if obj.cost_usd else "-"

    @admin.display(description="Result?", boolean=True)
    def has_result(self, obj):
        return bool(obj.result)


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "role", "short_content", "is_redacted", "created_at")
    list_filter = ("role", "is_redacted", "created_at")
    readonly_fields = ("id", "created_at")

    @admin.display(description="Content")
    def short_content(self, obj):
        return obj.content[:100] if obj.content else ""


@admin.register(Asset)
class AssetAdmin(admin.ModelAdmin):
    """Inspector for machine-created assets (embedded document images,
    spreadsheet tiles, generated/web images, file references). Everything but
    the two human-readable text fields is read-only, and rows can't be added
    by hand — the exactly-one-owner constraint and blob naming conventions
    belong to the code paths that mint them."""

    list_display = ("short_id", "thumbnail", "kind", "content_type", "owner",
                    "width", "height", "size_display", "created_by", "created_at")
    list_filter = ("kind", "content_type", "created_at")
    search_fields = ("description", "alt_text", "sha256", "version__document__original_filename")
    raw_id_fields = ("version", "canvas", "message", "thread", "slide_set", "created_by")
    readonly_fields = ("id", "preview", "kind", "content_type", "blob", "size_bytes",
                       "width", "height", "sha256", "source_url", "source_page_url",
                       "version", "canvas", "message", "thread", "slide_set",
                       "created_by", "created_at")
    fields = ("id", "preview", "kind", "content_type", "blob", "size_bytes",
              "width", "height", "sha256", "description", "alt_text",
              "source_url", "source_page_url",
              "version", "canvas", "message", "thread", "slide_set",
              "created_by", "created_at")
    list_select_related = ("version__document", "created_by")
    ordering = ["-created_at"]
    list_per_page = 50

    def has_add_permission(self, request):
        return False

    @admin.display(description="ID")
    def short_id(self, obj):
        return str(obj.id)[:8]

    @admin.display(description="Owner")
    def owner(self, obj):
        if obj.version_id:
            doc = obj.version.document
            return f"v{obj.version.version_index} of {doc.original_filename[:40]}"
        if obj.canvas_id:
            return f"canvas {obj.canvas_id}"
        if obj.message_id:
            return f"message {obj.message_id}"
        if obj.thread_id:
            return f"thread {str(obj.thread_id)[:8]}"
        if obj.slide_set_id:
            return f"slide set {str(obj.slide_set_id)[:8]}"
        return "—"

    @admin.display(description="Size")
    def size_display(self, obj):
        if not obj.size_bytes:
            return "—"
        if obj.size_bytes >= 1024 * 1024:
            return f"{obj.size_bytes / (1024 * 1024):.1f} MB"
        return f"{obj.size_bytes / 1024:.0f} KB"

    def _image_url(self, obj) -> str:
        """Best-effort browser-viewable URL for the asset's image bytes: the
        blob itself, else a reference asset's data-room source. Storage URL
        generation is local (signed for S3); the bytes are only fetched by the
        admin user's browser. Orphaned files (see the 2026-08 bucket swap) just
        render as a broken image."""
        try:
            if obj.blob:
                return obj.blob.url
            if obj.version_id:
                from chat.assets import image_asset_source

                source, content_type = image_asset_source(obj)
                if source and (content_type or "").startswith("image/"):
                    return source.url
        except Exception:  # noqa: BLE001 — cosmetic only, never break the page
            pass
        return ""

    @admin.display(description="Preview")
    def thumbnail(self, obj):
        from django.utils.html import format_html

        url = self._image_url(obj)
        if url and (obj.content_type or "").startswith("image/"):
            return format_html(
                '<img src="{}" loading="lazy" style="height: 40px; max-width: 120px; object-fit: contain;" />',
                url,
            )
        return "reference" if (obj.version_id and not obj.blob) else "—"

    @admin.display(description="Preview")
    def preview(self, obj):
        from django.utils.html import format_html

        url = self._image_url(obj)
        if url and (obj.content_type or "").startswith("image/"):
            return format_html(
                '<img src="{}" style="max-height: 320px; max-width: 640px; object-fit: contain; '
                'border: 1px solid #ccc;" />',
                url,
            )
        if obj.version_id and not obj.blob:
            return "Blob-less reference — bytes resolve from the data-room version on serve."
        return "—"
