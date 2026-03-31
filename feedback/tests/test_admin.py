from django.contrib.admin.sites import site
from django.test import TestCase

from feedback.models import Feedback


class FeedbackAdminTests(TestCase):
    def test_feedback_registered_in_admin(self):
        self.assertIn(Feedback, site._registry)
