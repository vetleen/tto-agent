from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models


class AgentSkill(models.Model):
    class Level(models.TextChoices):
        SYSTEM = "system", "System"
        ORG = "org", "Organization"
        USER = "user", "User"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    slug = models.SlugField(max_length=64)
    name = models.CharField(max_length=255)
    description = models.TextField(max_length=1024, blank=True)
    instructions = models.TextField()
    tool_names = models.JSONField(default=list, blank=True)
    level = models.CharField(max_length=10, choices=Level.choices)
    organization = models.ForeignKey(
        "accounts.Organization",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="agent_skills",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="agent_skills",
    )
    parent = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(
                fields=["slug"],
                condition=models.Q(level="system"),
                name="unique_system_skill_slug",
            ),
            models.UniqueConstraint(
                fields=["slug", "organization"],
                condition=models.Q(level="org"),
                name="unique_org_skill_slug",
            ),
            models.UniqueConstraint(
                fields=["slug", "created_by"],
                condition=models.Q(level="user"),
                name="unique_user_skill_slug",
            ),
            models.CheckConstraint(
                condition=~models.Q(level="system")
                | models.Q(organization__isnull=True, created_by__isnull=True),
                name="system_skill_no_owner",
            ),
            models.CheckConstraint(
                condition=~models.Q(level="org")
                | models.Q(organization__isnull=False),
                name="org_skill_has_org",
            ),
            models.CheckConstraint(
                condition=~models.Q(level="user")
                | models.Q(created_by__isnull=False),
                name="user_skill_has_creator",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.get_level_display()})"
