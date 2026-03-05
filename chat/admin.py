from django.contrib import admin

from .models import ChatMessage, ChatThread, ChatThreadDataRoom


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ("id", "role", "content", "tool_call_id", "created_at")
    fields = ("role", "content", "tool_call_id", "created_at")


class ChatThreadDataRoomInline(admin.TabularInline):
    model = ChatThreadDataRoom
    extra = 0
    readonly_fields = ("attached_at",)
    raw_id_fields = ("data_room",)


@admin.register(ChatThread)
class ChatThreadAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "created_by", "created_at", "updated_at")
    list_filter = ("created_at",)
    search_fields = ("title",)
    readonly_fields = ("id", "created_at", "updated_at")
    inlines = [ChatThreadDataRoomInline, ChatMessageInline]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "role", "short_content", "created_at")
    list_filter = ("role", "created_at")
    readonly_fields = ("id", "created_at")

    @admin.display(description="Content")
    def short_content(self, obj):
        return obj.content[:100] if obj.content else ""
