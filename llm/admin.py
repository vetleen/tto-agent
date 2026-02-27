from django.contrib import admin

from llm.models import LLMCallLog


@admin.register(LLMCallLog)
class LLMCallLogAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "model",
        "user",
        "status",
        "is_stream",
        "total_tokens",
        "duration_ms",
        "created_at",
    ]
    list_filter = ["status", "model", "is_stream"]
    search_fields = ["run_id", "user__email", "error_type"]
    readonly_fields = [
        "id",
        "created_at",
        "duration_ms",
        "user",
        "run_id",
        "model",
        "is_stream",
        "prompt",
        "raw_output",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "status",
        "error_type",
        "error_message",
    ]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
