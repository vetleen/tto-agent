from django.contrib import admin
from django.utils.html import format_html

from .models import Feedback


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "url", "short_text", "created_at")
    list_filter = ("created_at",)
    search_fields = ("text", "url", "user__email")
    raw_id_fields = ("user",)
    readonly_fields = (
        "user",
        "url",
        "user_agent",
        "viewport",
        "text",
        "console_errors",
        "screenshot_preview",
        "created_at",
    )

    @admin.display(description="Text")
    def short_text(self, obj):
        return obj.text[:100] if obj.text else ""

    @admin.display(description="Screenshot")
    def screenshot_preview(self, obj):
        if obj.screenshot:
            return format_html(
                '<img src="{}" style="max-width:400px; max-height:300px;" />',
                obj.screenshot.url,
            )
        return "-"
