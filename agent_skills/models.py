from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models

# Upper bound on a skill's instructions length. Enforced at the model
# (max_length, so ModelForms and the Django admin validate it), in the UI
# (the textarea's maxlength), and as a server-side truncation backstop on
# every write path. Equal to chat.services.CANVAS_MAX_CHARS so the
# canvas->instructions path can never exceed it.
MAX_INSTRUCTIONS_CHARS = 75_000

# Upper bound on a resource's (formerly "template") content length.
# SkillResource.content is an uncapped TextField (no migration needed); this is
# enforced as a truncation backstop on every write path (form save, import,
# canvas save) and when a resource's content is returned into the LLM context.
MAX_RESOURCE_CHARS = MAX_INSTRUCTIONS_CHARS
# Back-compat alias — callers migrating from the template-era name.
MAX_TEMPLATE_CHARS = MAX_RESOURCE_CHARS

# Hard cap on how many skills may be attached to a single chat thread.
# NOTE: being replaced by a token budget (settings.SKILL_ATTACH_TOKEN_BUDGET);
# kept until the chat_skill_attach / consumer / UI call sites are migrated.
MAX_THREAD_SKILLS = 5


def skill_resource_upload_path(instance: "SkillResource", filename: str) -> str:
    """Storage path for a resource's original (native) file — PDF/image bytes."""
    return f"skill_resources/{instance.skill_id}/{uuid.uuid4()}/{filename}"


class AgentSkill(models.Model):
    class Level(models.TextChoices):
        SYSTEM = "system", "System"
        ORG = "org", "Organization"
        USER = "user", "User"

    class Audience(models.TextChoices):
        # Who may use the skill. ``MAIN`` skills attach to a chat thread for the
        # main assistant; ``SUBAGENT`` skills are "specializations" the
        # orchestrator gives a sub-agent on spawn; ``SHARED`` skills surface in
        # both places (seed-only — there is no UI to author a shared skill).
        MAIN = "main", "Main agent"
        SUBAGENT = "subagent", "Sub-agent"
        SHARED = "shared", "Shared"

    class ScanState(models.TextChoices):
        # Approval gate for user/org skills (system skills are auto-approved).
        # A skill is *effectively approved* only when scan_state == APPROVED AND
        # its current content hash equals approved_content_hash (see
        # services.skill_is_approved). Any edit changes the hash and silently
        # un-approves the skill without a write here.
        UNSCANNED = "unscanned", "Not scanned"
        PENDING = "pending", "Scanning"
        APPROVED = "approved", "Approved"
        BLOCKED = "blocked", "Blocked"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    slug = models.SlugField(max_length=64)
    name = models.CharField(max_length=255)
    emoji = models.CharField(max_length=16, blank=True, default="")
    description = models.TextField(max_length=1024, blank=True)
    instructions = models.TextField(max_length=MAX_INSTRUCTIONS_CHARS)
    tool_names = models.JSONField(default=list, blank=True)
    audience = models.CharField(
        max_length=10, choices=Audience.choices, default=Audience.MAIN
    )
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

    # --- scan / approval gate ---
    scan_state = models.CharField(
        max_length=10, choices=ScanState.choices, default=ScanState.UNSCANNED
    )
    approved_content_hash = models.CharField(max_length=64, blank=True, default="")
    scan_detail = models.TextField(blank=True, default="")
    # Cached standing prompt cost (instructions + resource manifest), recomputed
    # on save; consulted by the chat_skill_attach token-budget check.
    standing_token_count = models.PositiveIntegerField(default=0)

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


class SkillResource(models.Model):
    """A file/document bundled with a skill and disclosed to the agent on demand.

    Generalizes the former ``SkillTemplate`` (name + content). Every existing
    template row migrates to ``kind=TEMPLATE``. Resources are read *whole, by
    name* (no chunking/embedding); PDFs and images additionally keep their
    native bytes so they can be shown to the model inline.
    """

    class Kind(models.TextChoices):
        # Semantic role. Only TEMPLATE is behaviorally special (canvas-loadable
        # via skill_resource_load / skill_template_load). REFERENCE is read-only.
        REFERENCE = "reference", "Reference"
        TEMPLATE = "template", "Template"

    class FileType(models.TextChoices):
        # Storage/representation axis (orthogonal to Kind). Drives the row icon
        # and whether "view" delivers text or an inline native asset.
        TEXT = "text", "Text"
        PDF = "pdf", "PDF"
        IMAGE = "image", "Image"

    class Status(models.TextChoices):
        # Mirrors the Data Room document lifecycle so the existing status-marker
        # JS + polling render directly.
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        SCANNING = "scanning", "Scanning"
        READY = "ready", "Ready"
        SCAN_FAILED = "scan_failed", "Scan failed"
        QUARANTINED = "quarantined", "Quarantined"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    skill = models.ForeignKey(
        AgentSkill, on_delete=models.CASCADE, related_name="templates"
    )
    name = models.CharField(max_length=255)
    kind = models.CharField(
        max_length=16, choices=Kind.choices, default=Kind.REFERENCE
    )
    file_type = models.CharField(
        max_length=16, choices=FileType.choices, default=FileType.TEXT
    )
    # Extracted markdown (text kinds) or a PDF's extracted-text fallback; empty
    # for images.
    content = models.TextField(blank=True, default="")
    # Native bytes for PDF/image resources (null for typed text).
    original_file = models.FileField(
        upload_to=skill_resource_upload_path, max_length=255, null=True, blank=True
    )
    original_filename = models.CharField(max_length=255, blank=True, default="")
    media_type = models.CharField(max_length=100, blank=True, default="")

    content_sha256 = models.CharField(max_length=64, blank=True, default="")
    token_count = models.PositiveIntegerField(default=0)

    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.PENDING
    )
    is_quarantined = models.BooleanField(default=False)
    quarantine_reason = models.CharField(max_length=255, blank=True, default="")
    quarantine_detail = models.TextField(blank=True, default="")
    # Only-True PII category flags (e.g. {"pii_special_category": True}); the
    # gated ones (Art.9/criminal) also set is_quarantined.
    pii_categories = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(
                fields=["skill", "name"], name="unique_resource_per_skill"
            ),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.get_kind_display()} for {self.skill.name})"


# Back-compat alias for callers still importing the template-era name. Points at
# the same model/table; removed once all references migrate to SkillResource.
SkillTemplate = SkillResource
