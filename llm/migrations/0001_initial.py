import uuid
from decimal import Decimal

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="LLMCallLog",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("duration_ms", models.PositiveIntegerField(blank=True, null=True)),
                (
                    "run_id",
                    models.CharField(blank=True, db_index=True, max_length=255),
                ),
                ("model", models.CharField(max_length=255)),
                ("is_stream", models.BooleanField(default=False)),
                ("prompt", models.JSONField()),
                ("raw_output", models.TextField(blank=True)),
                ("input_tokens", models.PositiveIntegerField(blank=True, null=True)),
                ("output_tokens", models.PositiveIntegerField(blank=True, null=True)),
                ("total_tokens", models.PositiveIntegerField(blank=True, null=True)),
                (
                    "cost_usd",
                    models.DecimalField(
                        blank=True, decimal_places=8, max_digits=12, null=True
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[("success", "Success"), ("error", "Error")],
                        db_index=True,
                        default="success",
                        max_length=32,
                    ),
                ),
                (
                    "error_type",
                    models.CharField(blank=True, max_length=255, null=True),
                ),
                ("error_message", models.TextField(blank=True)),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="llm_call_logs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "LLM Call Log",
                "verbose_name_plural": "LLM Call Logs",
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="llmcalllog",
            index=models.Index(
                fields=["created_at"], name="llm_calllog_created_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="llmcalllog",
            index=models.Index(
                fields=["user", "created_at"], name="llm_calllog_user_created_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="llmcalllog",
            index=models.Index(
                fields=["model", "created_at"], name="llm_calllog_model_created_idx"
            ),
        ),
    ]
