"""Tests for skill-resource ingest, scanning, and the approval gate."""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Organization
from agent_skills import resources as svc
from agent_skills.models import AgentSkill, SkillResource

User = get_user_model()

CLEAN_GUARDRAIL = ("allow", "", [])
CLEAN_PII = ({}, False, "", "")


class DetectFileTypeTests(TestCase):
    def test_maps_extensions_to_buckets(self):
        self.assertEqual(svc.detect_file_type("a.png"), SkillResource.FileType.IMAGE)
        self.assertEqual(svc.detect_file_type("a.jpg"), SkillResource.FileType.IMAGE)
        self.assertEqual(svc.detect_file_type("a.pdf"), SkillResource.FileType.PDF)
        self.assertEqual(svc.detect_file_type("a.docx"), SkillResource.FileType.TEXT)
        self.assertEqual(svc.detect_file_type("a.xlsx"), SkillResource.FileType.TEXT)
        self.assertEqual(svc.detect_file_type("a.txt"), SkillResource.FileType.TEXT)
        self.assertEqual(svc.detect_file_type("a.md"), SkillResource.FileType.TEXT)

    def test_rejects_audio_and_unknown(self):
        with self.assertRaises(svc.UnsupportedResourceType):
            svc.detect_file_type("a.mp3")
        with self.assertRaises(svc.UnsupportedResourceType):
            svc.detect_file_type("a.exe")


class HashAndApprovalTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="h@example.com", password="pw")
        self.skill = AgentSkill.objects.create(
            slug="s", name="S", instructions="do the thing", description="d",
            level="user", created_by=self.user,
        )

    def test_content_hash_deterministic_and_changes(self):
        h1 = svc.compute_skill_content_hash(self.skill)
        self.assertEqual(h1, svc.compute_skill_content_hash(self.skill))
        self.skill.instructions = "do a different thing"
        self.assertNotEqual(h1, svc.compute_skill_content_hash(self.skill))

    def test_hash_reflects_resources(self):
        h1 = svc.compute_skill_content_hash(self.skill)
        svc.create_text_resource(
            self.skill, name="R", content="body", user=self.user,
        )
        self.assertNotEqual(h1, svc.compute_skill_content_hash(self.skill))

    def test_system_skill_always_approved(self):
        sys_skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="i", level="system",
        )
        self.assertTrue(svc.skill_is_approved(sys_skill))

    def test_user_skill_needs_matching_hash(self):
        # Strict path: pretend the org has scanning configured.
        with patch.object(svc, "_scanning_configured", return_value=True):
            self.assertFalse(svc.skill_is_approved(self.skill))
            with patch.object(svc, "_scan_text_guardrail", return_value=CLEAN_GUARDRAIL), \
                    patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
                self.assertTrue(svc.scan_and_approve_skill(self.skill, self.user))
            self.skill.refresh_from_db()
            self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.APPROVED)
            self.assertTrue(svc.skill_is_approved(self.skill))
            # Editing the content invalidates approval without any write.
            self.skill.instructions = "changed"
            self.assertFalse(svc.skill_is_approved(self.skill))

    def test_unscanned_approved_when_no_scanning_configured(self):
        # Nothing to scan (no models / no membership) -> trivially usable.
        with patch.object(svc, "_scanning_configured", return_value=False):
            self.assertTrue(svc.skill_is_approved(self.skill))

    def test_blocked_never_approved(self):
        self.skill.scan_state = AgentSkill.ScanState.BLOCKED
        with patch.object(svc, "_scanning_configured", return_value=False):
            self.assertFalse(svc.skill_is_approved(self.skill))

    def test_standing_tokens_recomputed(self):
        self.assertEqual(self.skill.standing_token_count, 0)
        total = svc.recompute_standing_tokens(self.skill)
        self.assertGreater(total, 0)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.standing_token_count, total)


class CreateAndExtractTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="c@example.com", password="pw")
        self.skill = AgentSkill.objects.create(
            slug="s", name="S", instructions="i", level="user", created_by=self.user,
        )

    def test_create_text_resource_not_scanned_on_save(self):
        res = svc.create_text_resource(
            self.skill, name="Notes", content="Hello there", user=self.user,
        )
        self.assertEqual(res.file_type, SkillResource.FileType.TEXT)
        self.assertEqual(res.status, SkillResource.Status.READY)
        self.assertEqual(res.original_filename, "")
        self.assertGreater(res.token_count, 0)
        self.assertEqual(
            res.content_sha256, svc.resource_content_hash("Hello there"),
        )

    def test_extract_text_from_txt(self):
        text = svc.extract_text(b"hello world", "txt")
        self.assertIn("hello world", text)

    def test_ingest_text_file_scans_clean(self):
        with patch.object(svc, "_scan_text_guardrail", return_value=CLEAN_GUARDRAIL), \
                patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
            res = svc.ingest_file(
                self.skill, data=b"some reference notes", filename="notes.txt",
                user=self.user,
            )
        self.assertEqual(res.file_type, SkillResource.FileType.TEXT)
        self.assertEqual(res.original_filename, "notes.txt")
        self.assertEqual(res.status, SkillResource.Status.READY)
        self.assertIn("some reference notes", res.content)
        self.assertFalse(res.is_quarantined)

    def test_unique_name_deduplicates(self):
        svc.create_text_resource(self.skill, name="dup.txt", content="a", user=self.user)
        second = svc._unique_name(self.skill, "dup.txt")
        self.assertNotEqual(second, "dup.txt")


class ScanResourceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="s@example.com", password="pw")
        self.skill = AgentSkill.objects.create(
            slug="s", name="S", instructions="i", level="user", created_by=self.user,
        )
        self.res = svc.create_text_resource(
            self.skill, name="R", content="body text", user=self.user,
        )

    def test_clean_marks_ready(self):
        with patch.object(svc, "_scan_text_guardrail", return_value=CLEAN_GUARDRAIL), \
                patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
            svc.scan_resource(self.res, self.user)
        self.res.refresh_from_db()
        self.assertEqual(self.res.status, SkillResource.Status.READY)
        self.assertFalse(self.res.is_quarantined)

    def test_pii_gated_quarantines(self):
        pii = ({"pii_special_category": True}, True,
               "Contains GDPR Article 9 (special category) personal data.", "found")
        with patch.object(svc, "_scan_text_guardrail", return_value=CLEAN_GUARDRAIL), \
                patch.object(svc, "_scan_text_pii", return_value=pii):
            svc.scan_resource(self.res, self.user)
        self.res.refresh_from_db()
        self.assertEqual(self.res.status, SkillResource.Status.QUARANTINED)
        self.assertTrue(self.res.is_quarantined)
        self.assertIn("Article 9", self.res.quarantine_reason)
        self.assertEqual(self.res.pii_categories, {"pii_special_category": True})

    def test_guardrail_quarantines(self):
        guard = ("quarantine", "adversarial", ["delimiter_injection"])
        with patch.object(svc, "_scan_text_guardrail", return_value=guard), \
                patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
            svc.scan_resource(self.res, self.user)
        self.res.refresh_from_db()
        self.assertEqual(self.res.status, SkillResource.Status.QUARANTINED)
        self.assertTrue(self.res.is_quarantined)

    def test_scan_failure_fails_closed(self):
        with patch.object(svc, "_scan_text_guardrail", side_effect=RuntimeError("llm down")):
            svc.scan_resource(self.res, self.user)
        self.res.refresh_from_db()
        self.assertEqual(self.res.status, SkillResource.Status.SCAN_FAILED)
        self.assertFalse(self.res.is_quarantined)


class ApprovalGateTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="a@example.com", password="pw")
        self.skill = AgentSkill.objects.create(
            slug="s", name="S", instructions="do the thing", description="d",
            level="user", created_by=self.user,
        )

    def test_clean_skill_approved(self):
        with patch.object(svc, "_scan_text_guardrail", return_value=CLEAN_GUARDRAIL), \
                patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
            ok = svc.scan_and_approve_skill(self.skill, self.user)
        self.assertTrue(ok)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.APPROVED)
        self.assertTrue(self.skill.approved_content_hash)

    def test_system_skill_auto_approved(self):
        sys_skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="i", level="system",
        )
        with patch.object(svc, "_scan_text_guardrail") as g:
            self.assertTrue(svc.scan_and_approve_skill(sys_skill, self.user))
            g.assert_not_called()

    def test_quarantined_resource_blocks(self):
        SkillResource.objects.create(
            skill=self.skill, name="bad.pdf", kind="reference",
            file_type="pdf", original_filename="bad.pdf",
            is_quarantined=True, status="quarantined",
        )
        with patch.object(svc, "_scan_text_guardrail", return_value=CLEAN_GUARDRAIL), \
                patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
            ok = svc.scan_and_approve_skill(self.skill, self.user)
        self.assertFalse(ok)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.BLOCKED)
        self.assertIn("bad.pdf", self.skill.scan_detail)

    def test_adversarial_text_blocks(self):
        with patch.object(svc, "_scan_text_guardrail",
                          return_value=("quarantine", "adversarial", [])), \
                patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
            ok = svc.scan_and_approve_skill(self.skill, self.user)
        self.assertFalse(ok)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.BLOCKED)
