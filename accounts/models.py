from __future__ import annotations

from typing import Any

from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin
from django.db import models
from django.db.models.functions import Lower
from django.utils import timezone


class UserManager(BaseUserManager):
    def _create_user(self, email: str, password: str | None, **extra_fields: Any) -> "User":
        if not email:
            raise ValueError("The email address must be set.")
        # normalize_email only lowercases the domain; store the whole address
        # lowercase so equality and the Lower(email) unique constraint agree.
        # (The admin add form bypasses this manager — the constraint plus the
        # iexact lookup below still keep such accounts unique and loginable.)
        email = self.normalize_email(email).lower()
        user = self.model(email=email, **extra_fields)
        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()
        user.save(using=self._db)
        return user

    def get_by_natural_key(self, username: str | None) -> "User":
        # Case-insensitive login lookup, matching PasswordResetForm's iexact
        # behavior (and the Lower(email) unique constraint).
        return self.get(**{f"{self.model.USERNAME_FIELD}__iexact": username})

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

    # Optional profile picture. Two files are kept: a small re-encoded thumbnail
    # shown in the nav and chat (profile_picture), and the re-encoded full-size
    # upload retained for future, higher-resolution uses (profile_picture_original).
    # Uploads are validated and re-encoded in accounts.avatars (no PII/guardrails
    # scan — the bytes are never shown to the assistant).
    profile_picture = models.ImageField(
        upload_to="user_avatars/%Y/%m/", blank=True, max_length=500
    )
    profile_picture_original = models.ImageField(
        upload_to="user_avatars/originals/%Y/%m/", blank=True, max_length=500
    )

    # Email verification
    email_verified = models.BooleanField(default=False)
    last_verification_email_sent_at = models.DateTimeField(null=True, blank=True)
    verification_resend_count = models.PositiveIntegerField(default=0)
    verification_resend_window_start = models.DateTimeField(null=True, blank=True)

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS: list[str] = []

    class Meta:
        constraints = [
            models.UniqueConstraint(
                Lower("email"),
                name="accounts_user_email_ci_unique",
                violation_error_message="A user with this email already exists.",
            ),
        ]

    def __str__(self) -> str:
        return self.email


class EmailVerificationToken(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="email_verification_tokens",
    )
    token = models.CharField(max_length=64, unique=True)
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
    # Legacy: preferences["theme"] is the canonical store (backfilled in
    # migration 0007); this column is dual-written for deploy safety and can
    # be dropped once no running code reads it.
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


_MEMBERSHIP_CACHE_ATTR = "_cached_membership"


def get_membership(user):
    """Return the user's single Membership (with org), memoized on the user.

    Middleware, the navbar context processor, budget checks, and the
    preference resolvers all need the same membership row; caching it on the
    user instance collapses those into one query per request (request.user is
    a fresh instance each request, so the cache is naturally request-scoped).
    Long-lived holders of a user instance (WebSocket consumers) must call
    invalidate_membership_cache() before re-reading org state.
    """
    if not user or not user.is_authenticated:
        return None
    if not hasattr(user, _MEMBERSHIP_CACHE_ATTR):
        membership = (
            Membership.objects.filter(user=user).select_related("org").first()
        )
        setattr(user, _MEMBERSHIP_CACHE_ATTR, membership)
    return getattr(user, _MEMBERSHIP_CACHE_ATTR)


def invalidate_membership_cache(user) -> None:
    """Drop the memoized membership so the next get_membership() re-queries."""
    if hasattr(user, _MEMBERSHIP_CACHE_ATTR):
        delattr(user, _MEMBERSHIP_CACHE_ATTR)


def get_user_org(user):
    """Return the user's single Organization, or None."""
    m = get_membership(user)
    return m.org if m else None
