import hashlib

import agent_skills.models
from django.db import migrations, models


def _resource_content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _skill_content_hash(instructions: str, description: str, resources) -> str:
    parts = [instructions or "", description or ""]
    for res in resources:
        parts.append(
            "\x1f".join([res.name or "", res.kind or "", res.content_sha256 or ""])
        )
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def grandfather_existing(apps, schema_editor):
    """Every existing SkillTemplate row is a text template that was already in
    use; mark it ready and stamp its content hash. Existing user/org skills were
    usable before the approval gate existed, so grandfather them as approved so
    the Phase-3 attach gate never blocks them (system skills are always approved
    in code, so they need no stamp)."""
    AgentSkill = apps.get_model("agent_skills", "AgentSkill")
    SkillResource = apps.get_model("agent_skills", "SkillResource")

    for res in SkillResource.objects.all():
        res.kind = "template"
        res.file_type = "text"
        res.status = "ready"
        res.content_sha256 = _resource_content_hash(res.content)
        res.save(update_fields=["kind", "file_type", "status", "content_sha256"])

    for skill in AgentSkill.objects.exclude(level="system"):
        resources = SkillResource.objects.filter(skill=skill).order_by("name")
        skill.scan_state = "approved"
        skill.approved_content_hash = _skill_content_hash(
            skill.instructions, skill.description, resources
        )
        skill.save(update_fields=["scan_state", "approved_content_hash"])


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("agent_skills", "0005_agentskill_audience"),
    ]

    operations = [
        migrations.RenameModel(old_name="SkillTemplate", new_name="SkillResource"),
        migrations.RemoveConstraint(
            model_name="skillresource",
            name="unique_template_per_skill",
        ),
        migrations.AddConstraint(
            model_name="skillresource",
            constraint=models.UniqueConstraint(
                fields=("skill", "name"), name="unique_resource_per_skill"
            ),
        ),
        # --- new SkillResource fields ---
        migrations.AddField(
            model_name="skillresource",
            name="kind",
            field=models.CharField(
                choices=[("reference", "Reference"), ("template", "Template")],
                default="reference",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="file_type",
            field=models.CharField(
                choices=[("text", "Text"), ("pdf", "PDF"), ("image", "Image")],
                default="text",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="original_file",
            field=models.FileField(
                blank=True,
                null=True,
                upload_to=agent_skills.models.skill_resource_upload_path,
            ),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="original_filename",
            field=models.CharField(blank=True, default="", max_length=255),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="media_type",
            field=models.CharField(blank=True, default="", max_length=100),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="content_sha256",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="token_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("processing", "Processing"),
                    ("scanning", "Scanning"),
                    ("ready", "Ready"),
                    ("scan_failed", "Scan failed"),
                    ("quarantined", "Quarantined"),
                ],
                default="pending",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="is_quarantined",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="quarantine_reason",
            field=models.CharField(blank=True, default="", max_length=255),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="quarantine_detail",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="pii_categories",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="skillresource",
            name="error",
            field=models.TextField(blank=True, default=""),
        ),
        # --- new AgentSkill approval-gate fields ---
        migrations.AddField(
            model_name="agentskill",
            name="scan_state",
            field=models.CharField(
                choices=[
                    ("unscanned", "Not scanned"),
                    ("pending", "Scanning"),
                    ("approved", "Approved"),
                    ("blocked", "Blocked"),
                ],
                default="unscanned",
                max_length=10,
            ),
        ),
        migrations.AddField(
            model_name="agentskill",
            name="approved_content_hash",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="agentskill",
            name="scan_detail",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="agentskill",
            name="standing_token_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.RunPython(grandfather_existing, noop_reverse),
    ]
