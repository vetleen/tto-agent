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
        "formatted_prompt",
        "formatted_raw_prompt",
        "formatted_raw_output",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "status",
        "error_type",
        "error_message",
    ]
    fieldsets = (
        ("Identity", {
            "fields": ("id", "created_at", "duration_ms", "user", "run_id"),
        }),
        ("Request", {
            "fields": ("model", "is_stream", "formatted_prompt", "formatted_raw_prompt"),
        }),
        ("Response", {
            "fields": ("formatted_raw_output",),
        }),
        ("Usage", {
            "fields": ("input_tokens", "output_tokens", "total_tokens", "cost_usd"),
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

    @admin.display(description="Prompt (formatted)")
    def formatted_prompt(self, obj):
        return _pretty_json_html(obj.prompt)

    @admin.display(description="Raw Prompt (formatted)")
    def formatted_raw_prompt(self, obj):
        return _pretty_json_html(obj.raw_prompt)

    @admin.display(description="Raw Output (formatted)")
    def formatted_raw_output(self, obj):
        return _pretty_json_html(obj.raw_output)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
