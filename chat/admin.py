from django.contrib import admin
from django.db.models import CharField, OuterRef, Subquery, Sum
from django.db.models.functions import Cast

from llm.models import LLMCallLog

from .models import CanvasCheckpoint, ChatCanvas, ChatMessage, ChatThread, ChatThreadDataRoom, ThreadTask


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


@admin.register(ChatThread)
class ChatThreadAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "created_by", "cost_usd", "created_at", "updated_at")
    list_filter = ("created_at",)
    search_fields = ("title",)
    readonly_fields = ("id", "created_at", "updated_at")
    inlines = [ChatThreadDataRoomInline, ThreadTaskInline, ChatMessageInline]

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


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "role", "short_content", "is_redacted", "created_at")
    list_filter = ("role", "is_redacted", "created_at")
    readonly_fields = ("id", "created_at")

    @admin.display(description="Content")
    def short_content(self, obj):
        return obj.content[:100] if obj.content else ""
