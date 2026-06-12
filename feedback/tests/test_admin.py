from django.contrib.admin.sites import site
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from feedback.models import Feedback


class FeedbackAdminTests(TestCase):
    def test_feedback_registered_in_admin(self):
        self.assertIn(Feedback, site._registry)


class FeedbackAdminImmutabilityTests(TestCase):
    """Feedback is immutable user testimony — no adding or editing, even for
    superusers. Delete stays allowed (GDPR/cleanup)."""

    def setUp(self):
        self.admin_user = get_user_model().objects.create_superuser(
            email="admin@example.com",
            password="admin-pass-123",
        )
        self.client.force_login(self.admin_user)
        self.feedback = Feedback.objects.create(
            user=self.admin_user,
            text="The dashboard is great",
        )

    def test_add_permission_denied(self):
        response = self.client.get(reverse("admin:feedback_feedback_add"))
        self.assertEqual(response.status_code, 403)

    def test_change_post_denied(self):
        response = self.client.post(
            reverse("admin:feedback_feedback_change", args=[self.feedback.pk]),
            {"text": "tampered"},
        )
        self.assertEqual(response.status_code, 403)
        self.feedback.refresh_from_db()
        self.assertEqual(self.feedback.text, "The dashboard is great")

    def test_change_page_renders_read_only(self):
        # View permission remains, so the change URL renders Django's
        # read-only "view" page (200) with no Save button.
        response = self.client.get(
            reverse("admin:feedback_feedback_change", args=[self.feedback.pk])
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "The dashboard is great")
        self.assertNotContains(response, 'name="_save"')

    def test_delete_still_allowed(self):
        response = self.client.get(
            reverse("admin:feedback_feedback_delete", args=[self.feedback.pk])
        )
        self.assertEqual(response.status_code, 200)
