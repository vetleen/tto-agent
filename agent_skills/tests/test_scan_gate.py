"""Tests for the scan-on-save approval gate.

Covers ``request_skill_rescan`` (the hook every writer calls), the async
scan's snapshot/stamp discipline (``scan_and_approve_skill``), the per-blob
verdict cache, ``bulk_skill_approval``, and the views that render the verdict
(toggle, scan-status endpoint, list rows).
"""

from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import connection
from django.test import TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from accounts.models import Membership, Organization
from agent_skills import resources as svc
from agent_skills.models import AgentSkill, SkillResource

User = get_user_model()

CLEAN_GUARDRAIL = ("allow", "", [])
CLEAN_PII = ({}, False, "", "")
LOCMEM_CACHE = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "skill-scan-gate-tests",
    }
}


def _approve(skill):
    skill.scan_state = AgentSkill.ScanState.APPROVED
    skill.approved_content_hash = svc.compute_skill_content_hash(skill)
    skill.scan_detail = ""
    skill.save(update_fields=["scan_state", "approved_content_hash", "scan_detail"])


def _clean_scans():
    return patch.multiple(
        svc,
        _scan_text_guardrail=lambda *a, **k: CLEAN_GUARDRAIL,
        _scan_text_pii=lambda *a, **k: CLEAN_PII,
    )


class _GateTestBase(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="gate@example.com", password="pw")
        self.org = Organization.objects.create(name="Gate Org", slug="gate-org")
        Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.ADMIN
        )
        self.skill = AgentSkill.objects.create(
            slug="s", name="S", instructions="do the thing", description="d",
            level="user", created_by=self.user,
        )


class RequestRescanTests(_GateTestBase):
    """request_skill_rescan: no-op / inline approve / PENDING + queue."""

    def test_system_skill_is_a_noop(self):
        sys_skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="i", level="system",
        )
        with patch.object(svc, "_dispatch_scan") as dispatch:
            result = svc.request_skill_rescan(sys_skill, self.user)
        self.assertTrue(result["approved"])
        self.assertFalse(result["dispatched"])
        dispatch.assert_not_called()
        sys_skill.refresh_from_db()
        self.assertEqual(sys_skill.scan_state, AgentSkill.ScanState.UNSCANNED)

    def test_noop_while_still_approved(self):
        _approve(self.skill)
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            result = svc.request_skill_rescan(self.skill, self.user)
        self.assertTrue(result["approved"])
        self.assertEqual(result["scan_state"], AgentSkill.ScanState.APPROVED)
        dispatch.assert_not_called()

    def test_noop_when_scanning_not_configured(self):
        with patch.object(svc, "_scanning_configured", return_value=False), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            result = svc.request_skill_rescan(self.skill, self.user)
        self.assertTrue(result["approved"])
        dispatch.assert_not_called()
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.UNSCANNED)

    def test_empty_skill_approved_inline_without_a_scan(self):
        empty = AgentSkill.objects.create(
            slug="empty", name="Empty", instructions="", description="",
            level="user", created_by=self.user,
        )
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                patch.object(svc, "_scan_text_guardrail") as guard, \
                self.captureOnCommitCallbacks(execute=True):
            result = svc.request_skill_rescan(empty, self.user)
        self.assertTrue(result["approved"])
        self.assertFalse(result["dispatched"])
        dispatch.assert_not_called()
        guard.assert_not_called()
        empty.refresh_from_db()
        self.assertEqual(empty.scan_state, AgentSkill.ScanState.APPROVED)
        self.assertEqual(
            empty.approved_content_hash, svc.compute_skill_content_hash(empty)
        )

    def test_content_marks_pending_and_queues_after_commit(self):
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch:
            with self.captureOnCommitCallbacks(execute=True) as callbacks:
                result = svc.request_skill_rescan(self.skill, self.user)
                # Not yet: the queue happens on commit, never mid-transaction.
                dispatch.assert_not_called()
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(result["scan_state"], AgentSkill.ScanState.PENDING)
        self.assertFalse(result["approved"])
        self.assertTrue(result["dispatched"])
        dispatch.assert_called_once_with(str(self.skill.pk), self.user.pk)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.PENDING)
        self.assertEqual(self.skill.scan_detail, "")

    def test_clears_stale_detail_when_requeued(self):
        self.skill.scan_state = AgentSkill.ScanState.BLOCKED
        self.skill.scan_detail = "old reason"
        self.skill.save(update_fields=["scan_state", "scan_detail"])
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan"), \
                self.captureOnCommitCallbacks(execute=True):
            svc.request_skill_rescan(self.skill, self.user)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.PENDING)
        self.assertEqual(self.skill.scan_detail, "")

    def test_in_flight_upload_marks_pending_without_queueing(self):
        SkillResource.objects.create(
            skill=self.skill, name="big.pdf", file_type="pdf",
            original_filename="big.pdf", status=SkillResource.Status.PROCESSING,
        )
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            result = svc.request_skill_rescan(self.skill, self.user)
        self.assertEqual(result["scan_state"], AgentSkill.ScanState.PENDING)
        self.assertFalse(result["dispatched"])
        dispatch.assert_not_called()

    def test_dispatch_false_skips_the_queue(self):
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            result = svc.request_skill_rescan(self.skill, self.user, dispatch=False)
        self.assertEqual(result["scan_state"], AgentSkill.ScanState.PENDING)
        self.assertFalse(result["dispatched"])
        dispatch.assert_not_called()

    def test_failed_publish_reverts_pending_to_unscanned(self):
        self.skill.scan_state = AgentSkill.ScanState.PENDING
        self.skill.save(update_fields=["scan_state"])
        with patch(
            "agent_skills.tasks.scan_and_approve_skill_task.delay",
            side_effect=RuntimeError("broker down"),
        ):
            ok = svc._dispatch_scan(str(self.skill.pk), self.user.pk)
        self.assertFalse(ok)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.UNSCANNED)
        self.assertIn("scheduled", self.skill.scan_detail)

    def test_successful_publish_calls_the_task(self):
        with patch("agent_skills.tasks.scan_and_approve_skill_task.delay") as delay:
            ok = svc._dispatch_scan(str(self.skill.pk), self.user.pk)
        self.assertTrue(ok)
        delay.assert_called_once_with(str(self.skill.pk), self.user.pk)


class ScanOutcomeTests(_GateTestBase):
    """scan_and_approve_skill under async conditions."""

    def test_deferred_while_a_resource_is_in_flight(self):
        SkillResource.objects.create(
            skill=self.skill, name="big.pdf", file_type="pdf",
            original_filename="big.pdf", status=SkillResource.Status.SCANNING,
        )
        with patch.object(svc, "_scan_text_guardrail") as guard:
            ok = svc.scan_and_approve_skill(self.skill, self.user)
        self.assertFalse(ok)
        guard.assert_not_called()
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.PENDING)
        self.assertEqual(self.skill.approved_content_hash, "")

    def test_scan_failed_resource_blocks(self):
        SkillResource.objects.create(
            skill=self.skill, name="broken.docx", file_type="text",
            original_filename="broken.docx", status=SkillResource.Status.SCAN_FAILED,
        )
        with _clean_scans():
            ok = svc.scan_and_approve_skill(self.skill, self.user)
        self.assertFalse(ok)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.BLOCKED)
        self.assertIn("broken.docx", self.skill.scan_detail)

    def test_outcome_not_stamped_when_content_changed_mid_scan(self):
        skill_pk = self.skill.pk

        def mutate_then_allow(*args, **kwargs):
            AgentSkill.objects.filter(pk=skill_pk).update(instructions="changed mid-scan")
            return CLEAN_GUARDRAIL

        with patch.object(svc, "_scan_text_guardrail", side_effect=mutate_then_allow), \
                patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
            ok = svc.scan_and_approve_skill(self.skill, self.user)
        self.assertFalse(ok)
        self.skill.refresh_from_db()
        # Left PENDING for the save's own scan; the stale verdict never landed.
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.PENDING)
        self.assertEqual(self.skill.approved_content_hash, "")

    def test_approved_hash_is_the_scanned_snapshot(self):
        with _clean_scans():
            self.assertTrue(svc.scan_and_approve_skill(self.skill, self.user))
        self.skill.refresh_from_db()
        self.assertEqual(
            self.skill.approved_content_hash, svc.compute_skill_content_hash(self.skill)
        )
        with patch.object(svc, "_scanning_configured", return_value=True):
            self.assertTrue(svc.skill_is_approved(self.skill))

    def test_org_skill_without_creator_scans_as_an_org_admin(self):
        org_skill = AgentSkill.objects.create(
            slug="org-s", name="Org S", instructions="do", level="org",
            organization=self.org,
        )
        seen = {}

        def record(text, user, org_id, label):
            seen["user"] = user
            seen["org_id"] = org_id
            return CLEAN_GUARDRAIL

        with patch.object(svc, "_scan_text_guardrail", side_effect=record), \
                patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
            self.assertTrue(svc.scan_and_approve_skill(org_skill, None))
        self.assertEqual(seen["user"], self.user)
        self.assertEqual(seen["org_id"], self.org.id)

    def test_no_user_at_all_still_scans_and_skips_the_event_log(self):
        lonely_org = Organization.objects.create(name="Lonely", slug="lonely")
        org_skill = AgentSkill.objects.create(
            slug="lonely-s", name="Lonely S", instructions="do", level="org",
            organization=lonely_org,
        )
        with _clean_scans():
            self.assertTrue(svc.scan_and_approve_skill(org_skill, None))
        org_skill.refresh_from_db()
        self.assertEqual(org_skill.scan_state, AgentSkill.ScanState.APPROVED)
        # The audit row needs a user; without one it is skipped, not raised.
        create_fn = MagicMock()
        svc._log_guardrail(create_fn, None, lonely_org.id, "heuristic", [], 0.9, "high", "blocked", "x")
        create_fn.assert_not_called()

    def test_upload_landing_regates_the_skill(self):
        _approve(self.skill)
        with _clean_scans(), patch.object(svc, "_scanning_configured", return_value=True):
            svc.ingest_file(
                self.skill, data=b"reference notes", filename="notes.txt", user=self.user,
            )
            self.skill.refresh_from_db()
            # The new resource changed the hash; the worker path re-approved it.
            self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.APPROVED)
            self.assertTrue(svc.skill_is_approved(self.skill))

    def test_upload_extraction_failure_still_regates(self):
        _approve(self.skill)
        with _clean_scans(), \
                patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "extract_text", side_effect=RuntimeError("boom")):
            res = svc.ingest_file(
                self.skill, data=b"???", filename="broken.txt", user=self.user,
            )
        self.assertEqual(res.status, SkillResource.Status.SCAN_FAILED)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.BLOCKED)
        self.assertIn("broken.txt", self.skill.scan_detail)

    def test_task_skips_duplicate_dispatch_of_an_approved_skill(self):
        from agent_skills.tasks import scan_and_approve_skill_task

        _approve(self.skill)
        with patch.object(svc, "scan_and_approve_skill") as scan:
            scan_and_approve_skill_task(str(self.skill.pk), self.user.pk)
        scan.assert_not_called()

    def test_task_runs_the_scan_for_a_pending_skill(self):
        from agent_skills.tasks import scan_and_approve_skill_task

        self.skill.scan_state = AgentSkill.ScanState.PENDING
        self.skill.save(update_fields=["scan_state"])
        with patch.object(svc, "scan_and_approve_skill") as scan:
            scan_and_approve_skill_task(str(self.skill.pk), self.user.pk)
        scan.assert_called_once()


@override_settings(CACHES=LOCMEM_CACHE)
class BlobVerdictCacheTests(_GateTestBase):
    def setUp(self):
        super().setUp()
        cache.clear()
        self.skill.description = "the description"
        self.skill.save(update_fields=["description"])
        self.fp = patch.object(svc, "_scan_config_fingerprint", return_value="fp1")

    def _scan_with(self, guard):
        with self.fp, patch.object(svc, "_scan_text_guardrail", guard), \
                patch.object(svc, "_scan_text_pii", return_value=CLEAN_PII):
            return svc.scan_and_approve_skill(self.skill, self.user)

    def test_unchanged_blobs_are_not_rescanned(self):
        guard = MagicMock(return_value=CLEAN_GUARDRAIL)
        self.assertTrue(self._scan_with(guard))
        self.assertEqual(guard.call_count, 2)  # instructions + description
        guard.reset_mock()
        self.assertTrue(self._scan_with(guard))
        guard.assert_not_called()

    def test_only_the_changed_blob_is_rescanned(self):
        guard = MagicMock(return_value=CLEAN_GUARDRAIL)
        self._scan_with(guard)
        guard.reset_mock()
        AgentSkill.objects.filter(pk=self.skill.pk).update(description="new description")
        self.assertTrue(self._scan_with(guard))
        self.assertEqual(guard.call_count, 1)
        self.assertEqual(guard.call_args.args[0], "new description")

    def test_a_block_is_never_cached(self):
        def block_instructions(text, *a, **k):
            if text == "do the thing":
                return ("quarantine", "adversarial", [])
            return CLEAN_GUARDRAIL

        guard = MagicMock(side_effect=block_instructions)
        self.assertFalse(self._scan_with(guard))
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.BLOCKED)
        guard.reset_mock()
        self._scan_with(guard)
        # The blocked blob is scanned again; the clean description came from cache.
        self.assertEqual(guard.call_count, 1)
        self.assertEqual(guard.call_args.args[0], "do the thing")

    def test_scan_config_change_invalidates_cached_verdicts(self):
        guard = MagicMock(return_value=CLEAN_GUARDRAIL)
        self._scan_with(guard)
        guard.reset_mock()
        self.fp = patch.object(svc, "_scan_config_fingerprint", return_value="fp2")
        self._scan_with(guard)
        self.assertEqual(guard.call_count, 2)


class BulkApprovalTests(_GateTestBase):
    def test_bulk_matches_single_verdicts_with_bounded_queries(self):
        skills = []
        for i in range(5):
            s = AgentSkill.objects.create(
                slug=f"b{i}", name=f"B{i}", instructions="i", level="user",
                created_by=self.user,
            )
            svc.create_text_resource(s, name="R1", content="x", user=self.user)
            svc.create_text_resource(s, name="r2", content="y", user=self.user)
            if i % 2 == 0:
                _approve(s)
            skills.append(s)
        skills[1].scan_state = AgentSkill.ScanState.BLOCKED
        skills[1].save(update_fields=["scan_state"])

        configured = MagicMock(return_value=True)
        with patch.object(svc, "_scanning_configured_for_org", configured):
            with CaptureQueriesContext(connection) as ctx:
                verdicts = svc.bulk_skill_approval(self.user, skills)
            # One resource prefetch + the (memoized) membership lookup.
            self.assertLessEqual(len(ctx.captured_queries), 2)
            configured.assert_called_once()
            for s in skills:
                self.assertEqual(
                    verdicts[str(s.id)]["approved"],
                    svc.skill_is_approved(s, user=self.user),
                )
        self.assertTrue(verdicts[str(skills[0].id)]["approved"])
        self.assertFalse(verdicts[str(skills[1].id)]["approved"])
        self.assertEqual(verdicts[str(skills[1].id)]["scan_state"], "blocked")
        self.assertFalse(verdicts[str(skills[3].id)]["approved"])

    def test_hash_is_collation_ordered_not_python_sorted(self):
        # Mixed-case names: a codepoint sort would order "Zeta" before "alpha";
        # the DB ordering (Meta.ordering = ["name"]) is what the stored hash uses.
        svc.create_text_resource(self.skill, name="alpha", content="a", user=self.user)
        svc.create_text_resource(self.skill, name="Zeta", content="z", user=self.user)
        _approve(self.skill)
        fresh = AgentSkill.objects.get(pk=self.skill.pk)
        verdict = svc.bulk_skill_approval(self.user, [fresh])[str(fresh.id)]
        with patch.object(svc, "_scanning_configured", return_value=True):
            self.assertTrue(svc.skill_is_approved(fresh))
        self.assertEqual(verdict["scan_state"], "approved")
        self.assertEqual(
            svc.compute_skill_content_hash(fresh),
            svc.compute_skill_content_hash(fresh, list(fresh.templates.all())),
        )

    def test_invalidate_scan_config_cache(self):
        with patch.object(svc, "_scanning_configured_for_org", return_value=True) as cfg:
            svc._scanning_configured(self.skill, self.user)
            svc._scanning_configured(self.skill, self.user)
            self.assertEqual(cfg.call_count, 1)
            svc.invalidate_scan_config_cache(self.user)
            svc._scanning_configured(self.skill, self.user)
            self.assertEqual(cfg.call_count, 2)


@override_settings(ALLOWED_HOSTS=["testserver"])
class ScanGateViewTests(_GateTestBase):
    def setUp(self):
        super().setUp()
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)

    def _toggle(self, skill, enabled):
        return self.client.post(
            reverse("agent_skills_toggle", kwargs={"skill_id": skill.id}),
            {"enabled": "1" if enabled else "0"},
        )

    def test_toggle_enable_queues_scan_and_answers_immediately(self):
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                patch.object(svc, "scan_and_approve_skill") as scan, \
                self.captureOnCommitCallbacks(execute=True):
            resp = self._toggle(self.skill, True)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["now_active"])
        self.assertFalse(data["approved"])
        self.assertEqual(data["scan_state"], "pending")
        scan.assert_not_called()  # never inside the request
        dispatch.assert_called_once_with(str(self.skill.id), self.user.pk)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.PENDING)

    def test_toggle_enable_approved_skill_skips_the_scan(self):
        _approve(self.skill)
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            data = self._toggle(self.skill, True).json()
        self.assertTrue(data["approved"])
        self.assertEqual(data["scan_state"], "approved")
        dispatch.assert_not_called()

    def test_toggle_enable_blocked_skill_requeues(self):
        self.skill.scan_state = AgentSkill.ScanState.BLOCKED
        self.skill.scan_detail = "bad"
        self.skill.save(update_fields=["scan_state", "scan_detail"])
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            data = self._toggle(self.skill, True).json()
        self.assertFalse(data["approved"])
        self.assertEqual(data["scan_state"], "pending")
        dispatch.assert_called_once()

    def test_toggle_disable_never_queues(self):
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            data = self._toggle(self.skill, False).json()
        self.assertFalse(data["now_active"])
        self.assertIn("scan_state", data)
        dispatch.assert_not_called()

    def test_scan_status_endpoint_is_access_gated(self):
        other = User.objects.create_user(email="other@example.com", password="pw")
        theirs = AgentSkill.objects.create(
            slug="theirs", name="Theirs", instructions="i", level="user", created_by=other,
        )
        _approve(self.skill)
        with patch.object(svc, "_scanning_configured", return_value=True):
            resp = self.client.get(
                reverse("agent_skills_scan_status"),
                {"ids": f"{self.skill.id},{theirs.id},not-a-uuid"},
            )
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(set(data["skills"]), {str(self.skill.id)})
        self.assertTrue(data["skills"][str(self.skill.id)]["approved"])
        self.assertEqual(data["skills"][str(self.skill.id)]["scan_state"], "approved")

    def test_resource_status_includes_the_skill_verdict(self):
        self.skill.scan_state = AgentSkill.ScanState.PENDING
        self.skill.save(update_fields=["scan_state"])
        with patch.object(svc, "_scanning_configured", return_value=True):
            resp = self.client.get(
                reverse("agent_skills_resource_status", kwargs={"skill_id": self.skill.id})
            )
        data = resp.json()
        self.assertEqual(data["skill"]["scan_state"], "pending")
        self.assertFalse(data["skill"]["approved"])
        self.assertEqual(data["skill"]["scan_pill"], "pending")

    def test_resource_create_queues_scan(self):
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            resp = self.client.post(
                reverse("agent_skills_resource_create", kwargs={"skill_id": self.skill.id}),
                {"name": "Notes", "content": "Body", "kind": "reference"},
            )
        self.assertEqual(resp.json()["scan_state"], "pending")
        dispatch.assert_called_once()

    def test_form_save_queues_scan_once(self):
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            resp = self.client.post(
                reverse("agent_skills_save", kwargs={"skill_id": self.skill.id}),
                {
                    "action": "save", "name": "S", "description": "d",
                    "instructions": "do the NEW thing", "tool_names_json": "[]",
                },
            )
        self.assertEqual(resp.status_code, 302)
        dispatch.assert_called_once()
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.instructions, "do the NEW thing")
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.PENDING)

    def test_form_save_without_content_change_is_a_noop(self):
        _approve(self.skill)
        with patch.object(svc, "_scanning_configured", return_value=True), \
                patch.object(svc, "_dispatch_scan") as dispatch, \
                self.captureOnCommitCallbacks(execute=True):
            self.client.post(
                reverse("agent_skills_save", kwargs={"skill_id": self.skill.id}),
                {
                    "action": "save", "name": "Renamed only", "description": "d",
                    "instructions": "do the thing", "tool_names_json": "[]",
                },
            )
        dispatch.assert_not_called()
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.name, "Renamed only")
        self.assertEqual(self.skill.scan_state, AgentSkill.ScanState.APPROVED)

    def test_list_rows_carry_effective_state_and_pill(self):
        pending = AgentSkill.objects.create(
            slug="pending", name="Pending", instructions="i", level="user",
            created_by=self.user, scan_state=AgentSkill.ScanState.PENDING,
        )
        blocked = AgentSkill.objects.create(
            slug="blocked", name="Blocked", instructions="i", level="user",
            created_by=self.user, scan_state=AgentSkill.ScanState.BLOCKED,
            scan_detail="flagged",
        )
        _approve(self.skill)
        with patch.object(svc, "_scanning_configured", return_value=True):
            resp = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(resp.status_code, 200)
        rows = {r["skill"].slug: r for r in resp.context["main_user_rows"]}
        self.assertTrue(rows["s"]["is_enabled"])
        self.assertEqual(rows["s"]["scan_pill"], "")
        self.assertTrue(rows["pending"]["is_selected"])
        self.assertFalse(rows["pending"]["is_enabled"])
        self.assertEqual(rows["pending"]["scan_pill"], "pending")
        self.assertFalse(rows["blocked"]["is_enabled"])
        self.assertEqual(rows["blocked"]["scan_pill"], "blocked")
        self.assertEqual(rows["blocked"]["scan_detail"], "flagged")
        # A never-scanned, implicitly-enabled skill: off, with "scan needed".
        raw = AgentSkill.objects.create(
            slug="raw", name="Raw", instructions="i", level="user", created_by=self.user,
        )
        with patch.object(svc, "_scanning_configured", return_value=True):
            resp = self.client.get(reverse("agent_skills_list"))
        rows = {r["skill"].slug: r for r in resp.context["main_user_rows"]}
        self.assertFalse(rows[raw.slug]["is_enabled"])
        self.assertEqual(rows[raw.slug]["scan_pill"], "needed")

    def test_unconfigured_org_pending_skill_is_usable_and_unmarked(self):
        self.skill.scan_state = AgentSkill.ScanState.PENDING
        self.skill.save(update_fields=["scan_state"])
        with patch.object(svc, "_scanning_configured", return_value=False):
            resp = self.client.get(reverse("agent_skills_list"))
        rows = {r["skill"].slug: r for r in resp.context["main_user_rows"]}
        self.assertTrue(rows["s"]["is_enabled"])
        self.assertEqual(rows["s"]["scan_pill"], "")
