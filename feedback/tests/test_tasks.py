"""Tests for the feedback admin-notification task.

The task function is called directly (synchronously) — enqueueing is covered
by the view tests in test_views.py.
"""
from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings

from feedback.models import Feedback
from feedback.tasks import notify_admin_feedback_task


@override_settings(
    EMAIL_SENDING_ENABLED=True,
    ADMINS=[("Admin", "admin@example.com")],
    DEFAULT_FROM_EMAIL="noreply@example.com",
)
class NotifyAdminFeedbackTaskTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            email="emailer@example.com",
            password="test-pass-123",
        )
        self.feedback = Feedback.objects.create(
            user=self.user,
            text="Click here please",
            url="https://example.com/page",
        )
        mail.outbox = []

    def test_sends_email_to_admins(self):
        notify_admin_feedback_task(self.feedback.pk)
        self.assertEqual(len(mail.outbox), 1)
        message = mail.outbox[0]
        self.assertEqual(message.to, ["admin@example.com"])
        self.assertEqual(message.from_email, "noreply@example.com")
        self.assertIn(f"#{self.feedback.pk}", message.subject)
        self.assertIn(f"user #{self.user.pk}", message.subject)

    def test_email_marks_fields_as_user_supplied(self):
        notify_admin_feedback_task(self.feedback.pk)
        self.assertEqual(len(mail.outbox), 1)
        body = mail.outbox[0].body
        self.assertIn("user-supplied", body)
        self.assertIn("Click here please", body)
        self.assertIn("https://example.com/page", body)

    def test_noop_when_email_sending_disabled(self):
        with override_settings(EMAIL_SENDING_ENABLED=False):
            notify_admin_feedback_task(self.feedback.pk)
        self.assertEqual(mail.outbox, [])

    def test_noop_when_no_admins(self):
        with override_settings(ADMINS=[]):
            notify_admin_feedback_task(self.feedback.pk)
        self.assertEqual(mail.outbox, [])

    def test_noop_when_feedback_deleted(self):
        # Retention cleanup or user cascade can delete the row between
        # enqueue and run — the task must no-op, not raise.
        pk = self.feedback.pk
        self.feedback.delete()
        notify_admin_feedback_task(pk)
        self.assertEqual(mail.outbox, [])
