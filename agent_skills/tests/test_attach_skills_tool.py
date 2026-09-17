"""Tests for agent_skills.tools.AttachSkillsTool (declarative multi-skill set)."""

import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill
from agent_skills.tools import AttachSkillsTool
from chat.models import ChatThread, ChatThreadSkill
from llm.types import RunContext

User = get_user_model()


def _ctx(user, thread):
    return RunContext.create(user_id=user.pk, conversation_id=str(thread.id))


def _attached_ids(thread):
    """Attached skill ids in attach order (the through-model ordering)."""
    return [
        str(sid)
        for sid in ChatThreadSkill.objects.filter(thread=thread).values_list(
            "skill_id", flat=True
        )
    ]


def _approve(skill):
    """Mark a skill approved so the attach gate lets it through (a real skill
    reaches this state by being enabled, which runs the scan)."""
    from agent_skills.resources import compute_skill_content_hash

    skill.scan_state = AgentSkill.ScanState.APPROVED
    skill.approved_content_hash = compute_skill_content_hash(skill)
    skill.save(update_fields=["scan_state", "approved_content_hash"])


class AttachSkillsToolTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="attach@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="my-skill", name="My Skill", instructions="Do the thing.",
            description="Does the thing.", level="user", created_by=self.user,
        )
        _approve(self.skill)
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.tool = AttachSkillsTool()
        self.tool.context = _ctx(self.user, self.thread)

    def _attach(self, *slugs):
        return json.loads(self.tool._run(skill_slugs=list(slugs)))

    def test_attach_single_slug_updates_thread(self):
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["skills"], [{
            "id": str(self.skill.id), "slug": "my-skill", "name": "My Skill",
            "emoji": self.skill.emoji, "attached_by": "agent",
        }])
        self.assertEqual(result["added"], ["my-skill"])
        self.assertNotIn("removed", result)
        self.assertFalse(result["no_change"])
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])
        row = ChatThreadSkill.objects.get(thread=self.thread, skill=self.skill)
        self.assertEqual(row.attached_by, "agent")

    def test_attach_many_beyond_old_count_cap(self):
        # No count cap any more: attach 7 small skills, all fit the token budget.
        slugs = ["my-skill"]
        ids = [str(self.skill.id)]
        for i in range(1, 7):
            s = AgentSkill.objects.create(
                slug=f"s{i}", name=f"S{i}", instructions="x",
                level="user", created_by=self.user,
            )
            _approve(s)
            slugs.append(s.slug)
            ids.append(str(s.id))
        result = self._attach(*slugs)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([s["id"] for s in result["skills"]], ids)
        self.assertEqual(_attached_ids(self.thread), ids)

    @override_settings(SKILL_ATTACH_TOKEN_BUDGET=1)
    def test_over_token_budget_rejected(self):
        second = AgentSkill.objects.create(
            slug="s2", name="S2", instructions="x", level="user",
            created_by=self.user,
        )
        _approve(second)
        result = self._attach("my-skill", second.slug)
        self.assertEqual(result["status"], "error")
        self.assertIn("budget", result["message"])
        self.assertEqual(_attached_ids(self.thread), [])

    def test_blocked_skill_rejected(self):
        self.skill.scan_state = AgentSkill.ScanState.BLOCKED
        self.skill.scan_detail = "flagged as adversarial"
        self.skill.save(update_fields=["scan_state", "scan_detail"])
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "error")
        self.assertIn("Blocked by the safety scan", result["message"])
        self.assertIn("flagged as adversarial", result["message"])
        self.assertIn("Skills page", result["message"])
        self.assertEqual(_attached_ids(self.thread), [])

    def test_unscanned_rejected_when_scanning_configured(self):
        fresh = AgentSkill.objects.create(
            slug="fresh", name="Fresh", instructions="x",
            level="user", created_by=self.user,
        )  # never enabled -> unscanned
        with patch("agent_skills.resources._scanning_configured", return_value=True):
            result = json.loads(self.tool._run(skill_slugs=[fresh.slug]))
        self.assertEqual(result["status"], "error")
        self.assertIn("Not safety-scanned yet", result["message"])
        self.assertIn("fresh", result["message"])
        self.assertEqual(_attached_ids(self.thread), [])

    @override_settings(SKILL_ATTACH_PENDING_WAIT_SECONDS=0)
    def test_pending_skill_rejected_with_retry_hint(self):
        self.skill.scan_state = AgentSkill.ScanState.PENDING
        self.skill.approved_content_hash = ""
        self.skill.save(update_fields=["scan_state", "approved_content_hash"])
        with patch("agent_skills.resources._scanning_configured", return_value=True):
            result = self._attach("my-skill")
        self.assertEqual(result["status"], "error")
        self.assertIn("Still being safety-scanned", result["message"])
        self.assertIn("try again", result["message"])
        self.assertEqual(_attached_ids(self.thread), [])

    @override_settings(
        SKILL_ATTACH_PENDING_WAIT_SECONDS=2, SKILL_ATTACH_PENDING_POLL_SECONDS=0.05
    )
    def test_pending_skill_that_finishes_during_the_wait_attaches(self):
        from agent_skills.resources import compute_skill_content_hash

        self.skill.scan_state = AgentSkill.ScanState.PENDING
        self.skill.approved_content_hash = ""
        self.skill.save(update_fields=["scan_state", "approved_content_hash"])
        skill_pk, approved_hash = self.skill.pk, compute_skill_content_hash(self.skill)
        calls = {"n": 0}
        real_sleep = __import__("time").sleep

        def finish_scan_then_sleep(seconds):
            # The "worker" lands the verdict while the tool is waiting.
            calls["n"] += 1
            AgentSkill.objects.filter(pk=skill_pk).update(
                scan_state=AgentSkill.ScanState.APPROVED,
                approved_content_hash=approved_hash,
            )
            real_sleep(0)

        with patch("agent_skills.resources._scanning_configured", return_value=True), \
                patch("time.sleep", side_effect=finish_scan_then_sleep):
            result = self._attach("my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(calls["n"], 1)
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])

    def test_gate_runs_before_the_thread_lock(self):
        order = []
        from agent_skills import tools as tools_mod

        def fake_await(skills, user):
            order.append("gate")
            return None

        def fake_lock(pk):
            order.append("lock")

        with patch.object(tools_mod, "_await_skill_approval", side_effect=fake_await), \
                patch("chat.thread_skills.lock_thread", side_effect=fake_lock):
            result = self._attach("my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(order[:2], ["gate", "lock"])

    def test_already_attached_unapproved_not_rejected(self):
        # A re-list of a skill already on the thread is a no-op, not an approval
        # event — even if the skill is unapproved and scanning is configured.
        self.skill.scan_state = AgentSkill.ScanState.UNSCANNED
        self.skill.approved_content_hash = ""
        self.skill.save(update_fields=["scan_state", "approved_content_hash"])
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        with patch("agent_skills.resources._scanning_configured", return_value=True):
            result = self._attach("my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])

    def test_empty_list_is_noop_and_keeps_attached(self):
        """Additive tool: an empty list never detaches (that was the old replace
        semantics, and the reason user-attached skills kept disappearing)."""
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        result = self._attach()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["no_change"])
        self.assertEqual(result["added"], [])
        self.assertEqual(
            [(s["slug"], s["attached_by"]) for s in result["skills"]],
            [("my-skill", "user")],
        )
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])

    def test_empty_list_when_nothing_attached_is_noop(self):
        result = self._attach()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["no_change"])
        self.assertEqual(result["skills"], [])
        self.assertEqual(_attached_ids(self.thread), [])

    def test_attach_is_additive(self):
        """Attaching a second skill keeps the first (previous-then-new order)."""
        second = AgentSkill.objects.create(
            slug="second", name="Second", instructions="x",
            level="user", created_by=self.user,
        )
        _approve(second)
        self._attach("my-skill")
        result = self._attach("second")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["added"], ["second"])
        self.assertEqual([s["slug"] for s in result["skills"]], ["my-skill", "second"])
        self.assertEqual(
            _attached_ids(self.thread), [str(self.skill.id), str(second.id)]
        )

    def test_attach_keeps_user_attached_skill(self):
        """The production bug: a user-attached skill must survive the agent
        attaching another one, and the result says who attached what."""
        other = AgentSkill.objects.create(
            slug="users-pick", name="Users Pick", instructions="x",
            level="user", created_by=self.user,
        )
        ChatThreadSkill.objects.create(thread=self.thread, skill=other)  # UI → "user"
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            [(s["slug"], s["attached_by"]) for s in result["skills"]],
            [("users-pick", "user"), ("my-skill", "agent")],
        )
        self.assertEqual(
            _attached_ids(self.thread), [str(other.id), str(self.skill.id)]
        )
        self.assertEqual(
            ChatThreadSkill.objects.get(thread=self.thread, skill=other).attached_by,
            "user",
        )

    @override_settings(SKILL_ATTACH_TOKEN_BUDGET=1)
    def test_budget_error_names_only_new_skills(self):
        """The budget covers attached + new, but only NEW skills can be refused,
        and the message points at chat_skill_detach rather than at dropping the
        user's skill."""
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        second = AgentSkill.objects.create(
            slug="second", name="Second", instructions="x",
            level="user", created_by=self.user,
        )
        _approve(second)
        result = self._attach("second")
        self.assertEqual(result["status"], "error")
        self.assertIn("second", result["message"])
        self.assertNotIn("my-skill", result["message"])
        self.assertIn("chat_skill_detach", result["message"])
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])

    def test_end_label_for_result(self):
        label = self.tool.end_label_for_result
        # A refusal must never fall back to the success label.
        self.assertEqual(label({"status": "error"}), "Couldn't attach skill")
        self.assertEqual(label({"status": "ok", "added": [], "skills": []}), "No skills attached")
        self.assertEqual(
            label({"status": "ok", "added": [], "skills": [{"slug": "a"}]}),
            "Skill already attached",
        )
        self.assertEqual(
            label({"status": "ok", "added": ["a"], "skills": [{"slug": "a", "name": "Alpha"}]}),
            "Attached skill: Alpha",
        )
        self.assertEqual(
            label({"status": "ok", "added": ["a", "b"], "skills": []}), "Attached 2 skills"
        )

    def test_same_set_is_noop(self):
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["no_change"])
        self.assertEqual([s["id"] for s in result["skills"]], [str(self.skill.id)])

    def test_duplicate_slugs_deduped(self):
        result = self._attach("my-skill", "my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])

    def test_unknown_slug_returns_error_with_available_slugs(self):
        result = self._attach("does-not-exist")
        self.assertEqual(result["status"], "error")
        self.assertIn("my-skill", result["available_slugs"])
        self.assertEqual(_attached_ids(self.thread), [])

    def test_one_unknown_slug_rejects_whole_set(self):
        result = self._attach("my-skill", "nope")
        self.assertEqual(result["status"], "error")
        # Nothing persisted — the set is rejected atomically.
        self.assertEqual(_attached_ids(self.thread), [])

    def test_other_users_user_skill_not_attachable(self):
        other_user = User.objects.create_user(email="other@example.com", password="pass")
        AgentSkill.objects.create(
            slug="private", name="Private", instructions="x",
            level="user", created_by=other_user,
        )
        result = self._attach("private")
        self.assertEqual(result["status"], "error")
        self.assertEqual(_attached_ids(self.thread), [])

    def test_org_disabled_slug_not_attachable(self):
        org = Organization.objects.create(name="Org")
        Membership.objects.create(user=self.user, org=org, role=Membership.Role.ADMIN)
        AgentSkill.objects.create(
            slug="org-skill", name="Org Skill", instructions="x",
            level="org", organization=org,
        )
        org.preferences = {"skills": {"org-skill": {"enabled": False}}}
        org.save(update_fields=["preferences"])
        result = self._attach("org-skill")
        self.assertEqual(result["status"], "error")
        self.assertNotIn("org-skill", result["available_slugs"])

    def test_missing_context_returns_error(self):
        self.tool.context = None
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "error")

    def test_thread_belonging_to_other_user_not_attachable(self):
        other_user = User.objects.create_user(email="other2@example.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other_user, title="t")
        self.tool.context = RunContext.create(
            user_id=self.user.pk, conversation_id=str(other_thread.id),
        )
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "error")

    def test_whitespace_in_slug_stripped(self):
        result = json.loads(self.tool._run(skill_slugs=["  my-skill  "]))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])

    def test_none_slugs_treated_as_empty(self):
        result = json.loads(self.tool._run(skill_slugs=None))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["no_change"])

    # --- Same-turn activation: context slots the pipeline drains mid-loop ---

    def test_attach_populates_added_tool_names(self):
        """Attaching unlocks the skill's (org-filtered) tools for this turn."""
        self.tool.context.skill_tool_map = {"my-skill": ["skill_resource_view"]}
        self._attach("my-skill")
        self.assertEqual(self.tool.context.added_tool_names, ["skill_resource_view"])

    def test_attach_no_change_still_populates_added_tool_names(self):
        """Declarative: an already-attached skill still surfaces its tools (the
        pipeline dedupes), so a redundant re-attach never errors or hides tools."""
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        self.tool.context.skill_tool_map = {"my-skill": ["skill_resource_view"]}
        result = self._attach("my-skill")
        self.assertTrue(result["no_change"])
        self.assertEqual(self.tool.context.added_tool_names, ["skill_resource_view"])

    def test_attach_unmapped_skill_adds_no_tools(self):
        """Fail-closed: a skill absent from skill_tool_map (org-disabled tools /
        no prefs) contributes nothing, never bypassing org filtering."""
        self._attach("my-skill")
        self.assertEqual(self.tool.context.added_tool_names, [])

    def test_attach_new_skill_pushes_instructions(self):
        """A newly-attached skill's instructions are queued for same-turn injection."""
        self._attach("my-skill")
        instr = self.tool.context.pending_skill_instructions
        self.assertEqual(len(instr), 1)
        self.assertIn("Do the thing.", instr[0])
        # Rendered with its slug and origin so the agent can detach it later.
        self.assertIn("Slug: `my-skill`", instr[0])
        self.assertIn("attached by you", instr[0])

    def test_attach_already_attached_skips_instructions(self):
        """Instructions are NOT re-injected for a skill already in this turn's
        system prompt (avoids duplication)."""
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        self._attach("my-skill")
        self.assertEqual(self.tool.context.pending_skill_instructions, [])

    # --- Concurrency: read → diff → write runs under the thread's row lock ---

    def test_diff_sees_rows_committed_before_the_lock(self):
        """WILFRED-8P: two parallel chat_skill_attach calls raced their
        delete+insert and the loser hit the (thread, skill) unique constraint.
        Now the current set is read only after taking the lock, so a writer that
        committed just before us shows up in the diff and only the missing row
        is inserted. Simulated by attaching from the lock hook."""
        second = AgentSkill.objects.create(
            slug="second", name="Second", instructions="x",
            level="user", created_by=self.user,
        )
        _approve(second)

        def concurrent_writer(thread_id):
            # Idempotent: add_thread_skills re-locks inside the same
            # transaction, and the row must still be there for that second call.
            ChatThreadSkill.objects.get_or_create(thread=self.thread, skill=self.skill)

        with patch("chat.thread_skills.lock_thread", side_effect=concurrent_writer):
            result = self._attach("my-skill", "second")

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["no_change"])
        # Only "second" is new relative to what the other writer committed.
        self.assertEqual(result["added"], ["second"])
        self.assertEqual(
            [(s["slug"], s["attached_by"]) for s in result["skills"]],
            [("my-skill", "user"), ("second", "agent")],
        )
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id), str(second.id)])
        # And "my-skill" counts as already attached: no instruction re-injection.
        self.assertEqual(len(self.tool.context.pending_skill_instructions), 1)
        self.assertIn("Second", self.tool.context.pending_skill_instructions[0])

    def test_failed_write_leaves_turn_state_untouched(self):
        """The same-turn effects (tool unlock, instruction injection) run after
        the commit, so a failed write can't leave the live turn out of step with
        what was persisted."""
        self.tool.context.skill_tool_map = {"my-skill": ["skill_resource_view"]}
        with patch(
            "chat.thread_skills.add_thread_skills", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                self.tool._run(skill_slugs=["my-skill"])
        self.assertEqual(self.tool.context.added_tool_names, [])
        self.assertEqual(self.tool.context.pending_skill_instructions, [])
        self.assertEqual(_attached_ids(self.thread), [])

    def test_tool_registered(self):
        from llm.tools.registry import get_tool_registry
        names = get_tool_registry().list_tools()
        self.assertIn("chat_skill_attach", names)
        self.assertIn("chat_skill_detach", names)
