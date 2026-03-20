from django.contrib import admin

from guardrails.models import GuardrailEvent


@admin.register(GuardrailEvent)
class GuardrailEventAdmin(admin.ModelAdmin):
    list_display = (
        "created_at", "user", "trigger_source", "check_type",
        "severity", "action_taken", "confidence",
    )
    list_filter = ("trigger_source", "check_type", "severity", "action_taken", "organization")
    search_fields = ("user__email", "raw_input")
    raw_id_fields = ("user", "organization", "thread", "llm_call_log", "related_event")
    readonly_fields = (
        "id", "created_at", "user", "organization", "thread", "llm_call_log",
        "trigger_source", "check_type", "tags", "confidence", "severity",
        "action_taken", "raw_input", "reviewer_output", "related_event",
    )
    ordering = ("-created_at",)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
