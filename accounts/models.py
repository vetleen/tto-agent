from __future__ import annotations

from typing import Any

from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin
from django.db import models


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
    # Preferred LLM model for chat (e.g. moonshot/kimi-k2.5). Blank = use app default.
    chat_model = models.CharField(max_length=128, blank=True, default="")

    def __str__(self) -> str:
        return f"Settings for {self.user}"


class Organization(models.Model):
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100, unique=True)

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

    class Meta:
        unique_together = [("user", "org")]
        ordering = ("org", "user")

    def __str__(self) -> str:
        return f"{self.user} in {self.org} ({self.role})"
