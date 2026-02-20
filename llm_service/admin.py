from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from .models import LLMCallLog


@admin.register(LLMCallLog)
class LLMCallLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "created_at",
        "model",
        "status",
        "user_message_preview",
        "total_tokens",
        "cost_usd",
        "duration_ms",
        "user",
        "request_id",
    )
    list_filter = ("status", "model", "is_stream")
    search_fields = ("request_id", "model", "error_message", "user_message_preview", "prompt_preview")
    readonly_fields = (
        "id",
        "created_at",
        "model",
        "is_stream",
        "request_kwargs",
        "prompt_hash",
        "prompt_preview",
        "user_message_preview",
        "provider_response_id",
        "response_model",
        "response_preview",
        "response_hash",
        "raw_response_payload",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "cost_source",
        "status",
        "error_type",
        "error_message",
        "http_status",
        "retry_count",
        "provider_request_id",
        "metadata",
        "request_id",
        "duration_ms",
        "user",
    )
    fieldsets = (
        (None, {"fields": ("id", "created_at", "duration_ms", "status", "user", "request_id", "metadata")}),
        ("Request", {"fields": ("model", "is_stream", "request_kwargs", "prompt_hash", "prompt_preview", "user_message_preview")}),
        ("Response", {"fields": ("provider_response_id", "response_model", "response_preview", "response_hash")}),
        ("Raw LLM response", {"fields": ("raw_response_payload",), "description": "Full raw response from LiteLLM (JSON). For streams this is an array of chunk objects."}),
        ("Usage / cost", {"fields": ("input_tokens", "output_tokens", "total_tokens", "cost_usd", "cost_source")}),
        ("Errors", {"fields": ("error_type", "error_message", "http_status", "retry_count", "provider_request_id")}),
    )
    ordering = ["-created_at"]
    date_hierarchy = "created_at"

    def raw_response_payload_formatted(self, obj):
        """Display raw_response_payload in detail view as preformatted, scrollable block."""
        if not obj.raw_response_payload:
            return "—"
        return mark_safe(
            '<pre style="max-height: 60em; overflow: auto; white-space: pre-wrap; word-break: break-all;">'
            + obj.raw_response_payload.replace("<", "&lt;").replace(">", "&gt;")
            + "</pre>"
        )

    raw_response_payload_formatted.short_description = "Raw LLM response (full)"

    def get_readonly_fields(self, request, obj=None):
        """Show formatted raw response in change view instead of raw text field."""
        fields = list(super().get_readonly_fields(request, obj=obj))
        if obj is not None and "raw_response_payload" in fields:
            fields = [f for f in fields if f != "raw_response_payload"]
            fields.append("raw_response_payload_formatted")
        return fields

    def get_fieldsets(self, request, obj=None):
        """Use formatted field in change view."""
        fieldsets = super().get_fieldsets(request, obj=obj)
        if fieldsets is None:
            fieldsets = self.fieldsets
        if obj is not None and fieldsets:
            fieldsets = list(fieldsets)
            for i, fieldset in enumerate(fieldsets):
                if not isinstance(fieldset, (list, tuple)) or len(fieldset) != 2:
                    continue
                name, opts = fieldset
                if not isinstance(opts, dict):
                    continue
                fields = opts.get("fields")
                if name and "Raw LLM response" in (name or "") and fields and "raw_response_payload" in fields:
                    opts = dict(opts)
                    opts["fields"] = ("raw_response_payload_formatted",)
                    fieldsets[i] = (name, opts)
                    break
        return fieldsets
