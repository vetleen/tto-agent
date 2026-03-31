import json

from django.contrib import admin
from django.utils.html import format_html

from llm.models import LLMCallLog


def _pretty_json_html(value, fallback="(empty)"):
    """Render a JSON-serialisable value or raw string as an indented <pre> block."""
    if value is None or value == "":
        return fallback

    if isinstance(value, (dict, list)):
        formatted = json.dumps(value, indent=2, ensure_ascii=False)
    else:
        try:
            parsed = json.loads(value)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            formatted = str(value)

    return format_html(
        '<pre style="max-height:400px;overflow:auto;background:#f8f9fa;'
        'padding:12px;border-radius:4px;font-size:13px;line-height:1.4;'
        'white-space:pre-wrap;word-break:break-word;">{}</pre>',
        formatted,
    )


@admin.register(LLMCallLog)
class LLMCallLogAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "prompt_preview",
        "user_prompt_preview",
        "model",
        "user",
        "cost_usd",
        "status",
        "is_stream",
        "has_tools",
        "total_tokens",
        "duration_ms",
        "created_at",
    ]
    list_filter = [
        "status",
        "model",
        "is_stream",
        ("user", admin.RelatedOnlyFieldListFilter),
        ("created_at", admin.DateFieldListFilter),
    ]
    date_hierarchy = "created_at"
    search_fields = ["run_id", "trace_id", "conversation_id", "user__email", "error_type"]
    readonly_fields = [
        "id",
        "created_at",
        "duration_ms",
        "user",
        "run_id",
        "trace_id",
        "conversation_id",
        "model",
        "is_stream",
        "formatted_prompt",
        "formatted_tools",
        "formatted_raw_output",
        "formatted_response_metadata",
        "stop_reason",
        "provider_model_id",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "cost_usd",
        "status",
        "error_type",
        "error_message",
    ]
    fieldsets = (
        ("Identity", {
            "fields": ("id", "created_at", "duration_ms", "user", "run_id"),
        }),
        ("Tracing", {
            "fields": ("trace_id", "conversation_id"),
        }),
        ("Request", {
            "fields": ("model", "is_stream", "formatted_prompt", "formatted_tools"),
        }),
        ("Response", {
            "fields": ("formatted_raw_output", "formatted_response_metadata", "stop_reason", "provider_model_id"),
        }),
        ("Usage", {
            "fields": ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens", "cost_usd"),
        }),
        ("Status", {
            "fields": ("status", "error_type", "error_message"),
        }),
    )

    @admin.display(description="Prompt")
    def prompt_preview(self, obj):
        messages = obj.prompt or []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle structured content blocks (e.g. Anthropic format)
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content = block.get("text", "")
                        break
                else:
                    content = ""
            if content:
                return content[:100] + ("…" if len(content) > 100 else "")
        return ""

    @admin.display(description="User Prompt")
    def user_prompt_preview(self, obj):
        messages = obj.prompt or []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content = block.get("text", "")
                        break
                else:
                    content = ""
            if content:
                return content[:100] + ("…" if len(content) > 100 else "")
        return ""

    @admin.display(description="Prompt (formatted)")
    def formatted_prompt(self, obj):
        return _pretty_json_html(obj.prompt)

    @admin.display(description="Tools (formatted)")
    def formatted_tools(self, obj):
        return _pretty_json_html(obj.tools)

    @admin.display(boolean=True, description="Tools?")
    def has_tools(self, obj):
        return bool(obj.tools)

    @admin.display(description="Raw Output (formatted)")
    def formatted_raw_output(self, obj):
        return _pretty_json_html(obj.raw_output)

    @admin.display(description="Response Metadata (formatted)")
    def formatted_response_metadata(self, obj):
        return _pretty_json_html(obj.response_metadata)

    def changelist_view(self, request, extra_context=None):
        response = super().changelist_view(request, extra_context=extra_context)
        if hasattr(response, "context_data"):
            try:
                from django.db.models import Count, Sum

                qs = response.context_data["cl"].queryset
                response.context_data["summary"] = qs.aggregate(
                    total_cost=Sum("cost_usd"),
                    total_calls=Count("id"),
                )
            except (KeyError, AttributeError):
                pass
        return response

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
