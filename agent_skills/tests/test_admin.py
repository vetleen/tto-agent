"""Tests for the AgentSkill admin — the soft-delete restore action."""

from django.contrib.admin.sites import AdminSite
from django.contrib.auth import get_user_model
from django.contrib.messages.storage.fallback import FallbackStorage
from django.test import RequestFactory, TestCase
from django.utils import timezone

from agent_skills.admin import AgentSkillAdmin
from agent_skills.models import AgentSkill

User = get_user_model()


def _request_with_messages():
    request = RequestFactory().post("/admin/")
    setattr(request, "session", {})
    setattr(request, "_messages", FallbackStorage(request))
    return request


class RestoreSkillsActionTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="adm@example.com", password="pass")
        self.admin = AgentSkillAdmin(AgentSkill, AdminSite())
        self.deleted = AgentSkill.objects.create(
            slug="gone", name="Gone", instructions="i",
            level="user", created_by=self.user,
            deleted_at=timezone.now(), is_active=False,
        )
        self.live = AgentSkill.objects.create(
            slug="here", name="Here", instructions="i",
            level="user", created_by=self.user,
        )

    def test_restore_action_undeletes_and_reactivates(self):
        self.admin.restore_skills(_request_with_messages(), AgentSkill.objects.all())
        self.deleted.refresh_from_db()
        self.assertIsNone(self.deleted.deleted_at)
        self.assertTrue(self.deleted.is_active)
        # The already-live skill is untouched.
        self.live.refresh_from_db()
        self.assertIsNone(self.live.deleted_at)
        self.assertEqual(self.live.slug, "here")

    def test_restore_rededupes_when_slug_taken(self):
        # A live skill already holds the slug the deleted one wants back.
        AgentSkill.objects.create(
            slug="gone", name="Reused", instructions="i",
            level="user", created_by=self.user,
        )
        self.admin.restore_skills(_request_with_messages(), AgentSkill.objects.all())
        self.deleted.refresh_from_db()
        self.assertIsNone(self.deleted.deleted_at)
        self.assertTrue(self.deleted.is_active)
        self.assertNotEqual(self.deleted.slug, "gone")
