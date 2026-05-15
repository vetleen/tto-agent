from django.conf import settings
from django.db import models
from django.utils import timezone

from core.retention import RETENTION_PERIODS


class Feedback(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="feedback_submissions",
    )
    url = models.URLField(max_length=2000, blank=True, default="")
    user_agent = models.TextField(blank=True, default="")
    viewport = models.CharField(max_length=50, blank=True, default="")
    text = models.TextField()
    screenshot = models.ImageField(
        upload_to="feedback/%Y/%m/",
        blank=True,
        max_length=500,
    )
    console_errors = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    retain_until = models.DateTimeField(null=True, blank=True, db_index=True)

    def save(self, *args, **kwargs):
        if self._state.adding:
            self.retain_until = timezone.now() + RETENTION_PERIODS["feedback.Feedback"]
        super().save(*args, **kwargs)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Feedback #{self.pk} by {self.user} at {self.created_at:%Y-%m-%d %H:%M}"
