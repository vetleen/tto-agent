"""Tests for feedback file cleanup signal."""
from __future__ import annotations

import tempfile
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from feedback.models import Feedback

User = get_user_model()


class FeedbackScreenshotCleanupTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._override = override_settings(MEDIA_ROOT=cls._tmpdir.name)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._override.disable()
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def setUp(self):
        self.user = User.objects.create_user(email="fb@example.com", password="testpass")

    def test_delete_feedback_removes_screenshot(self):
        img = SimpleUploadedFile("shot.png", b"\x89PNG\r\n", content_type="image/png")
        fb = Feedback.objects.create(user=self.user, text="test", screenshot=img)
        stored_path = Path(fb.screenshot.path)
        self.assertTrue(stored_path.exists())

        fb.delete()
        self.assertFalse(stored_path.exists())

    def test_delete_without_screenshot_does_not_raise(self):
        fb = Feedback.objects.create(user=self.user, text="no screenshot")
        fb.delete()
        self.assertFalse(Feedback.objects.filter(pk=fb.pk).exists())

    def test_retain_until_set_on_creation(self):
        from datetime import timedelta

        from django.utils import timezone

        before = timezone.now()
        fb = Feedback.objects.create(user=self.user, text="retention check")
        self.assertIsNotNone(fb.retain_until)
        self.assertGreaterEqual(fb.retain_until, before + timedelta(days=89))
