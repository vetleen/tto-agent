from django.contrib import admin

from .models import ChatMessage, ChatThread


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ("id", "created_at", "token_count")
    fields = ("id", "role", "status", "content", "error", "token_count", "llm_call_log", "created_at")
    ordering = ("created_at", "id")


@admin.register(ChatThread)
class ChatThreadAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "user", "is_archived", "created_at", "updated_at", "last_message_at")
    list_filter = ("is_archived", "created_at", "updated_at", "last_message_at")
    search_fields = ("title", "user__email", "user__username", "id")
    readonly_fields = ("id", "created_at", "updated_at")
    inlines = [ChatMessageInline]
