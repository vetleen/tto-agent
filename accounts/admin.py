from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from django.utils import timezone

from .models import EmailVerificationToken, Membership, Organization, Scope, User


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    model = User
    ordering = ("email",)
    list_display = ("email", "is_staff", "is_active", "email_verified")
    search_fields = ("email",)
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Email verification", {"fields": ("email_verified",)}),
        ("Permissions", {"fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions")}),
        ("Important dates", {"fields": ("last_login", "date_joined")}),
    )


@admin.register(EmailVerificationToken)
class EmailVerificationTokenAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at", "is_expired")
    list_filter = ("created_at",)
    search_fields = ("user__email",)
    raw_id_fields = ("user",)
    readonly_fields = ("token", "created_at")

    def is_expired(self, obj):
        from django.conf import settings as django_settings
        timeout = getattr(django_settings, "EMAIL_VERIFICATION_TIMEOUT", 86400)
        return (timezone.now() - obj.created_at).total_seconds() > timeout
    is_expired.boolean = True
    is_expired.short_description = "Expired"
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("email", "password1", "password2", "is_staff", "is_superuser"),
            },
        ),
    )


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    search_fields = ("name", "slug")
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Scope)
class ScopeAdmin(admin.ModelAdmin):
    list_display = ("code", "name")
    search_fields = ("code", "name")


@admin.register(Membership)
class MembershipAdmin(admin.ModelAdmin):
    list_display = ("user", "org", "role")
    list_filter = ("role", "org")
    search_fields = ("user__email", "org__name")
    raw_id_fields = ("user", "org")
    filter_horizontal = ("scopes",)
