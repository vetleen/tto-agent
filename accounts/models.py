from __future__ import annotations

from typing import Any

from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin
from django.db import models
from django.utils import timezone


class UserManager(BaseUserManager):
    def _create_user(self, email: str, password: str | None, **extra_fields: Any) -> "User":
        if not email:
            raise ValueError("The email address must be set.")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()
        user.save(using=self._db)
        return user

    def create_user(self, email: str, password: str | None = None, **extra_fields: Any) -> "User":
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email: str, password: str | None = None, **extra_fields: Any) -> "User":
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")
        return self._create_user(email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(auto_now_add=True)

    # Profile
    first_name = models.CharField(max_length=150, blank=True, default="")
    last_name = models.CharField(max_length=150, blank=True, default="")
    title = models.CharField(max_length=150, blank=True, default="")
    description = models.TextField(blank=True, default="", max_length=5000)
    # Personal personality override for the assistant ("SOUL"). Blank = inherit
    # the org-wide soul, then the system default. See accounts.agent_customization.
    soul = models.TextField(blank=True, default="", max_length=5000)

    # Email verification
    email_verified = models.BooleanField(default=False)
    last_verification_email_sent_at = models.DateTimeField(null=True, blank=True)
    verification_resend_count = models.PositiveIntegerField(default=0)
    verification_resend_window_start = models.DateTimeField(null=True, blank=True)

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS: list[str] = []

    def __str__(self) -> str:
        return self.email


class EmailVerificationToken(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="email_verification_tokens",
    )
    token = models.CharField(max_length=64, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"Verification for {self.user.email}"


class UserSettings(models.Model):
    class Theme(models.TextChoices):
        LIGHT = "light", "Light"
        DARK = "dark", "Dark"

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="settings",
        primary_key=True,
    )
    theme = models.CharField(
        max_length=10,
        choices=Theme.choices,
        default=Theme.LIGHT,
    )
    preferences = models.JSONField(default=dict, blank=True)

    def __str__(self) -> str:
        return f"Settings for {self.user}"


class Organization(models.Model):
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100, unique=True)
    description = models.TextField(blank=True, default="", max_length=5000)
    # Org-wide personality baseline for the assistant ("SOUL"). Blank = inherit
    # the system default. Members may override it with their own User.soul when
    # the org allows it (preferences["allow_user_soul"]).
    soul = models.TextField(blank=True, default="", max_length=5000)
    preferences = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ("name",)

    def __str__(self) -> str:
        return self.name


class Scope(models.Model):
    """Reusable scope for membership permissions (e.g. billing, settings)."""
    code = models.SlugField(max_length=64, unique=True)
    name = models.CharField(max_length=128)

    class Meta:
        ordering = ("code",)

    def __str__(self) -> str:
        return self.name or self.code


class Membership(models.Model):
    class Role(models.TextChoices):
        ADMIN = "admin", "Admin"
        MEMBER = "member", "Member"
        VIEWER = "viewer", "Viewer"

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="organization_memberships",
    )
    org = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="memberships",
    )
    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.MEMBER,
    )
    scopes = models.ManyToManyField(
        Scope,
        related_name="memberships",
        blank=True,
    )

    # Suspension (per-org; admins manage their own members). Lifecycle fields
    # are maintained by save() below — note that queryset .update() bypasses
    # save(), so suspension writers must use save()/suspend()/unsuspend().
    is_suspended = models.BooleanField(default=False)
    suspended_at = models.DateTimeField(null=True, blank=True)
    suspended_reason = models.TextField(blank=True, default="")

    class Meta:
        ordering = ("org", "user")
        constraints = [
            models.UniqueConstraint(fields=["user"], name="unique_membership_per_user"),
        ]

    # Kept (rather than the constraint's violation_error_message) so the form
    # error stays keyed on the "user" field — admin inlines and tests rely on it.
    def validate_unique(self, exclude=None):
        super().validate_unique(exclude=exclude)
        if self.user_id is not None:
            qs = Membership.objects.filter(user_id=self.user_id)
            if self.pk:
                qs = qs.exclude(pk=self.pk)
            if qs.exists():
                from django.core.exceptions import ValidationError
                raise ValidationError(
                    {"user": "This user already belongs to an organization."}
                )

    def save(self, **kwargs):
        """Keep the suspension bookkeeping consistent for every writer.

        Suspending without a timestamp stamps now; unsuspending clears the
        timestamp and reason. Explicitly-set values are never overwritten, and
        update_fields is extended so the bookkeeping always persists.
        """
        extra: set[str] = set()
        if self.is_suspended and self.suspended_at is None:
            self.suspended_at = timezone.now()
            extra = {"suspended_at"}
        elif not self.is_suspended and (self.suspended_at is not None or self.suspended_reason):
            self.suspended_at = None
            self.suspended_reason = ""
            extra = {"suspended_at", "suspended_reason"}
        update_fields = kwargs.get("update_fields")
        if extra and update_fields is not None:
            kwargs["update_fields"] = set(update_fields) | extra
        super().save(**kwargs)

    def suspend(self, reason: str = "") -> None:
        """Suspend this membership; save() stamps suspended_at."""
        self.is_suspended = True
        self.suspended_reason = (reason or "")[:2000]
        self.save(update_fields=["is_suspended", "suspended_reason"])

    def unsuspend(self) -> None:
        """Lift the suspension; save() clears the timestamp and reason."""
        self.is_suspended = False
        self.save(update_fields=["is_suspended"])

    def __str__(self) -> str:
        return f"{self.user} in {self.org} ({self.role})"


def get_user_org(user):
    """Return the user's single Organization, or None."""
    if not user or not user.is_authenticated:
        return None
    m = Membership.objects.filter(user=user).select_related("org").first()
    return m.org if m else None
