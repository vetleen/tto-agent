from django.contrib import admin

from agent_skills.models import AgentSkill, SkillResource


class SkillResourceInline(admin.TabularInline):
    model = SkillResource
    extra = 0
    fields = ("name", "kind", "file_type", "status", "is_quarantined", "token_count")
    readonly_fields = ("token_count",)


@admin.register(AgentSkill)
class AgentSkillAdmin(admin.ModelAdmin):
    list_display = (
        "name", "slug", "level", "audience", "organization", "created_by",
        "is_active", "scan_state", "standing_token_count",
    )
    list_filter = ("level", "audience", "scan_state", "is_active")
    search_fields = ("name", "slug", "description")
    raw_id_fields = ("organization", "created_by", "parent")
    fieldsets = (
        ("Identity", {"fields": ("name", "slug", "emoji", "description")}),
        ("Ownership", {"fields": ("level", "audience", "organization", "created_by", "parent")}),
        ("Configuration", {"fields": ("instructions", "tool_names", "is_active")}),
        # standing_token_count is what the chat_skill_attach token budget
        # (SKILL_ATTACH_TOKEN_BUDGET, ~40k) sums across attached skills.
        ("Scan / approval", {
            "fields": (
                "scan_state", "approved_content_hash", "scan_detail",
                "standing_token_count",
            )
        }),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )
    readonly_fields = (
        "created_at", "updated_at", "approved_content_hash", "standing_token_count",
    )
    inlines = [SkillResourceInline]
