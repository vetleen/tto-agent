from django.contrib import admin, messages

from agent_skills.models import AgentSkill, SkillResource


class SkillResourceInline(admin.TabularInline):
    model = SkillResource
    extra = 0
    fields = ("name", "kind", "file_type", "status", "is_quarantined", "token_count")
    readonly_fields = ("token_count",)


class DeletedSkillFilter(admin.SimpleListFilter):
    """Filter by soft-delete state (``deleted_at`` null/not-null).

    Clearer than the datetime ``list_filter`` default, which offers date ranges
    rather than the live/deleted split an admin actually wants here.
    """

    title = "deleted"
    parameter_name = "deleted"

    def lookups(self, request, model_admin):
        return (("yes", "Deleted"), ("no", "Live"))

    def queryset(self, request, queryset):
        if self.value() == "yes":
            return queryset.filter(deleted_at__isnull=False)
        if self.value() == "no":
            return queryset.filter(deleted_at__isnull=True)
        return queryset


@admin.register(AgentSkill)
class AgentSkillAdmin(admin.ModelAdmin):
    list_display = (
        "name", "slug", "level", "audience", "organization", "created_by",
        "is_active", "deleted_at", "scan_state", "standing_token_count",
    )
    list_filter = ("level", "audience", "scan_state", "is_active", DeletedSkillFilter)
    search_fields = ("name", "slug", "description")
    raw_id_fields = ("organization", "created_by", "parent")
    actions = ["restore_skills"]
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
        # deleted_at is read-only: restore only via the "Restore selected skills"
        # action, which also re-dedupes the slug and re-activates the skill.
        # Clearing it by hand would skip both and can trip the unique constraint.
        ("Lifecycle", {"fields": ("deleted_at",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )
    readonly_fields = (
        "created_at", "updated_at", "approved_content_hash", "standing_token_count",
        "deleted_at",
    )
    inlines = [SkillResourceInline]

    @admin.action(description="Restore selected skills (undo soft-delete)")
    def restore_skills(self, request, queryset):
        from agent_skills.services import restore_skill

        restored = 0
        for skill in queryset.filter(deleted_at__isnull=False):
            restore_skill(skill)
            restored += 1
        if restored:
            self.message_user(
                request, f"Restored {restored} skill(s).", messages.SUCCESS
            )
        else:
            self.message_user(
                request,
                "No soft-deleted skills in the selection.",
                messages.WARNING,
            )
