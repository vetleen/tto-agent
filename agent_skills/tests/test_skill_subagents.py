"""Sub-agent specializations made available through skills.

A specialization is ``always_available`` (listed to the orchestrator on every
thread) or listed only while an attached playbook names it in ``subagents``.
Listing only: ``chat_subagent_create`` accepts any enabled specialization.
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Membership, Organization
from agent_skills.models import MAX_SUBAGENTS, AgentSkill
from agent_skills.services import (
    allowed_subagent_slugs,
    clean_subagents,
    export_skill,
    fork_skill,
    import_skill,
    normalize_subagents,
    parse_skill_export,
    skill_subagents,
    specialization_always_available,
    unmet_subagents,
    visible_specializations,
)

User = get_user_model()


def _approve(skill):
    from agent_skills.resources import compute_skill_content_hash

    skill.scan_state = AgentSkill.ScanState.APPROVED
    skill.approved_content_hash = compute_skill_content_hash(skill)
    skill.save(update_fields=["scan_state", "approved_content_hash"])


class _Base(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="subs@example.com", password="pass")
        self.org = Organization.objects.create(name="Org")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.ADMIN)
        self.sys_spec = AgentSkill.objects.create(
            slug="patent-searcher", name="Patent Searcher", instructions="x",
            level="system", audience="subagent", always_available=False,
        )
        self.generic = AgentSkill.objects.create(
            slug="researcher", name="Researcher", instructions="x",
            level="system", audience="subagent",
        )
        self.org_spec = AgentSkill.objects.create(
            slug="org-spec", name="Org Spec", instructions="x", level="org",
            organization=self.org, audience="subagent", always_available=False,
        )
        self.user_spec = AgentSkill.objects.create(
            slug="my-spec", name="My Spec", instructions="x", level="user",
            created_by=self.user, audience="subagent", always_available=False,
        )
        # System seeds are opt-in per org: enable the two system specializations.
        self.org.preferences = {"skills": {
            "patent-searcher": {"enabled": True}, "researcher": {"enabled": True},
        }}
        self.org.save(update_fields=["preferences"])


class NormalizeTests(_Base):
    def test_normalize_drops_self_dupes_and_bad_slugs_and_caps(self):
        raw = ["A", "a", "own", "bad slug", 3] + [f"s{i}" for i in range(20)]
        out = normalize_subagents(raw, own_slug="own")
        self.assertEqual(out[0], "a")
        self.assertNotIn("own", out)
        self.assertNotIn("bad slug", out)
        self.assertEqual(len(out), MAX_SUBAGENTS)
        self.assertEqual(normalize_subagents("patent-searcher"), [])

    def test_only_main_playbooks_carry_subagents(self):
        playbook = AgentSkill(slug="p", audience="main", subagents=["patent-searcher"])
        toolkit = AgentSkill(slug="t", kind="toolkit", subagents=["patent-searcher"])
        spec = AgentSkill(slug="s", audience="subagent", subagents=["patent-searcher"])
        self.assertEqual(skill_subagents(playbook), ["patent-searcher"])
        self.assertEqual(skill_subagents(toolkit), [])
        self.assertEqual(skill_subagents(spec), [])

    def test_shared_specialization_counts_as_always(self):
        self.assertTrue(specialization_always_available(
            AgentSkill(audience="shared", always_available=False)
        ))
        self.assertFalse(specialization_always_available(self.sys_spec))
        self.assertTrue(specialization_always_available(self.generic))


class VisibleSpecializationsTests(_Base):
    def _entries(self):
        from core.preferences import get_preferences

        return get_preferences(self.user).allowed_specializations

    def test_entries_carry_always_available(self):
        by_slug = {e["slug"]: e for e in self._entries()}
        self.assertFalse(by_slug["patent-searcher"]["always_available"])
        self.assertTrue(by_slug["researcher"]["always_available"])

    def test_skills_only_listed_with_a_naming_skill_attached(self):
        entries = self._entries()
        playbook = AgentSkill(slug="prior-art", subagents=["patent-searcher"])
        unrelated = AgentSkill(slug="other")
        self.assertEqual(
            [e["slug"] for e in visible_specializations(entries, [])], ["researcher"],
        )
        self.assertEqual(
            [e["slug"] for e in visible_specializations(entries, [unrelated])],
            ["researcher"],
        )
        self.assertEqual(
            sorted(e["slug"] for e in visible_specializations(entries, [playbook])),
            ["patent-searcher", "researcher"],
        )

    def test_toolkit_cannot_unlock(self):
        entries = self._entries()
        toolkit = AgentSkill(slug="t_tools", kind="toolkit", subagents=["patent-searcher"])
        self.assertNotIn(
            "patent-searcher",
            [e["slug"] for e in visible_specializations(entries, [toolkit])],
        )


class TierRuleTests(_Base):
    def test_system_playbook_names_system_only(self):
        skill = AgentSkill(slug="p", level="system")
        self.assertEqual(
            sorted(allowed_subagent_slugs(skill, self.user)),
            ["patent-searcher", "researcher"],
        )

    def test_org_playbook_names_system_and_org(self):
        skill = AgentSkill(slug="p", level="org", organization=self.org)
        self.assertEqual(
            sorted(allowed_subagent_slugs(skill, self.user)),
            ["org-spec", "patent-searcher", "researcher"],
        )

    def test_user_playbook_names_any_accessible(self):
        skill = AgentSkill(slug="p", level="user", created_by=self.user)
        self.assertEqual(
            sorted(allowed_subagent_slugs(skill, self.user)),
            ["my-spec", "org-spec", "patent-searcher", "researcher"],
        )

    def test_clean_reports_dropped(self):
        skill = AgentSkill(slug="p", level="org", organization=self.org)
        kept, dropped = clean_subagents(
            skill, ["patent-searcher", "my-spec", "nope"], self.user,
        )
        self.assertEqual(kept, ["patent-searcher"])
        self.assertEqual(dropped, ["my-spec", "nope"])

    def test_clean_on_specialization_keeps_nothing(self):
        kept, dropped = clean_subagents(self.user_spec, ["patent-searcher"], self.user)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, ["patent-searcher"])


class UnmetTests(_Base):
    def test_disabled_specialization_is_unavailable_with_its_name(self):
        self.org.preferences["skills"]["patent-searcher"] = {"enabled": False}
        self.org.save(update_fields=["preferences"])
        user = User.objects.get(pk=self.user.pk)  # fresh: prefs are memoized
        _approve(self.org_spec)
        skill = AgentSkill(
            slug="p", level="user", created_by=user,
            subagents=["patent-searcher", "org-spec", "gone"],
        )
        self.assertEqual(unmet_subagents(skill, user), [
            {"slug": "patent-searcher", "name": "Patent Searcher", "reason": "unavailable"},
            {"slug": "gone", "name": "gone", "reason": "unavailable"},
        ])

    def test_unscanned_specialization_is_unapproved(self):
        skill = AgentSkill(
            slug="p", level="user", created_by=self.user, subagents=["my-spec"],
        )
        self.assertEqual(unmet_subagents(skill, self.user), [
            {"slug": "my-spec", "name": "My Spec", "reason": "unapproved"},
        ])


class CopyExportImportTests(_Base):
    def test_fork_carries_both_fields(self):
        playbook = AgentSkill.objects.create(
            slug="prior-art", name="Prior Art", instructions="x", level="org",
            organization=self.org, subagents=["patent-searcher"],
        )
        copy = fork_skill(self.user, playbook, rescan=False)
        self.assertEqual(copy.subagents, ["patent-searcher"])
        spec_copy = fork_skill(self.user, self.org_spec, rescan=False)
        self.assertFalse(spec_copy.always_available)
        self.assertEqual(spec_copy.subagents, [])

    def test_export_import_round_trip(self):
        playbook = AgentSkill.objects.create(
            slug="prior-art", name="Prior Art", instructions="x", level="user",
            created_by=self.user, subagents=["patent-searcher"],
        )
        data = export_skill(playbook)
        self.assertEqual(data["subagents"], ["patent-searcher"])
        [payload] = parse_skill_export(json.dumps({"skills": [data]}))
        imported = import_skill(self.user, payload)
        self.assertEqual(imported.subagents, ["patent-searcher"])

    def test_import_specialization_keeps_availability(self):
        data = export_skill(self.user_spec)
        self.assertFalse(data["always_available"])
        self.assertEqual(data["subagents"], [])
        [payload] = parse_skill_export(json.dumps({"skills": [data]}))
        self.assertFalse(payload["always_available"])
        imported = import_skill(self.user, payload)
        self.assertFalse(imported.always_available)
        # The import keeps it a specialization (it used to land as a main skill).
        self.assertEqual(imported.audience, "subagent")
