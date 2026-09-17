"""Tests for the rescan_skills rollout command."""

from io import StringIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from accounts.models import Membership, Organization
from agent_skills import resources as svc
from agent_skills.models import AgentSkill

User = get_user_model()


class RescanSkillsCommandTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="cmd@example.com", password="pw")
        self.org = Organization.objects.create(name="O", slug="o")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.ADMIN)
        self.raw = AgentSkill.objects.create(
            slug="raw", name="Raw", instructions="i", level="user", created_by=self.user,
        )
        self.org_raw = AgentSkill.objects.create(
            slug="org-raw", name="Org Raw", instructions="i", level="org",
            organization=self.org,
        )
        self.blocked = AgentSkill.objects.create(
            slug="blocked", name="Blocked", instructions="i", level="user",
            created_by=self.user, scan_state=AgentSkill.ScanState.BLOCKED,
        )
        approved = AgentSkill.objects.create(
            slug="ok", name="Ok", instructions="i", level="user", created_by=self.user,
        )
        approved.scan_state = AgentSkill.ScanState.APPROVED
        approved.approved_content_hash = svc.compute_skill_content_hash(approved)
        approved.save(update_fields=["scan_state", "approved_content_hash"])
        AgentSkill.objects.create(slug="sys", name="Sys", instructions="i", level="system")

    def _run(self, *args):
        out = StringIO()
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            call_command("rescan_skills", *args, stdout=out)
        return out.getvalue(), dispatch

    def test_dry_run_lists_without_changing(self):
        output, dispatch = self._run("--dry-run")
        self.assertIn("raw", output)
        self.assertIn("org-raw", output)
        self.assertNotIn("blocked", output)
        self.assertNotIn("/ok", output)
        self.assertNotIn("sys", output)
        dispatch.assert_not_called()
        self.raw.refresh_from_db()
        self.assertEqual(self.raw.scan_state, AgentSkill.ScanState.UNSCANNED)

    def test_queues_unapproved_skills(self):
        output, dispatch = self._run()
        self.assertEqual(dispatch.call_count, 2)
        self.raw.refresh_from_db()
        self.org_raw.refresh_from_db()
        self.blocked.refresh_from_db()
        self.assertEqual(self.raw.scan_state, AgentSkill.ScanState.PENDING)
        self.assertEqual(self.org_raw.scan_state, AgentSkill.ScanState.PENDING)
        self.assertEqual(self.blocked.scan_state, AgentSkill.ScanState.BLOCKED)
        self.assertIn("Processed 2", output)

    def test_state_blocked_only(self):
        _, dispatch = self._run("--state", "blocked")
        dispatch.assert_called_once()
        self.blocked.refresh_from_db()
        self.assertEqual(self.blocked.scan_state, AgentSkill.ScanState.PENDING)

    def test_limit(self):
        _, dispatch = self._run("--limit", "1")
        dispatch.assert_called_once()
