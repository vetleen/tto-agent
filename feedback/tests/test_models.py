from django.contrib.auth import get_user_model
from django.test import TestCase

from feedback.models import Feedback


class FeedbackModelTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            email="tester@example.com",
            password="test-pass-123",
        )

    def test_create_feedback(self):
        fb = Feedback.objects.create(
            user=self.user,
            url="http://localhost/chat/",
            text="Great feature!",
            user_agent="Mozilla/5.0",
            viewport="1920x1080",
        )
        self.assertIsNotNone(fb.pk)
        self.assertIsNotNone(fb.created_at)
        self.assertEqual(fb.user, self.user)

    def test_str(self):
        fb = Feedback.objects.create(user=self.user, text="Test")
        self.assertIn("Feedback #", str(fb))
        self.assertIn("tester@example.com", str(fb))

    def test_ordering(self):
        fb1 = Feedback.objects.create(user=self.user, text="First")
        fb2 = Feedback.objects.create(user=self.user, text="Second")
        items = list(Feedback.objects.all())
        self.assertEqual(items[0], fb2)
        self.assertEqual(items[1], fb1)

    def test_user_deletion_preserves_feedback(self):
        fb = Feedback.objects.create(user=self.user, text="Keep this")
        self.user.delete()
        fb.refresh_from_db()
        self.assertIsNone(fb.user)
        self.assertEqual(fb.text, "Keep this")

    def test_default_values(self):
        fb = Feedback.objects.create(user=self.user, text="Minimal")
        self.assertEqual(fb.url, "")
        self.assertEqual(fb.user_agent, "")
        self.assertEqual(fb.viewport, "")
        self.assertEqual(fb.console_errors, [])
        self.assertFalse(fb.screenshot)
