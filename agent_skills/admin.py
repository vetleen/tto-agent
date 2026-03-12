from django.contrib import admin

from agent_skills.models import AgentSkill, SkillTemplate


class SkillTemplateInline(admin.TabularInline):
    model = SkillTemplate
    extra = 0


@admin.register(AgentSkill)
class AgentSkillAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "level", "organization", "created_by", "is_active")
    list_filter = ("level", "is_active")
    search_fields = ("name", "slug", "description")
    raw_id_fields = ("organization", "created_by", "parent")
    fieldsets = (
        ("Identity", {"fields": ("name", "slug", "description")}),
        ("Ownership", {"fields": ("level", "organization", "created_by", "parent")}),
        ("Configuration", {"fields": ("instructions", "tool_names", "is_active")}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )
    readonly_fields = ("created_at", "updated_at")
    inlines = [SkillTemplateInline]
