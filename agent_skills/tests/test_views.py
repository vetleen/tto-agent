"""View tests for the agent_skills UI."""

import json

from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages
from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization, UserSettings
from agent_skills.models import MAX_INSTRUCTIONS_CHARS, AgentSkill, SkillTemplate

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsListViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="u@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Acme", slug="acme")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)
        AgentSkill.objects.create(
            slug="sys-x", name="Sys X", instructions="i", level="system",
        )
        AgentSkill.objects.create(
            slug="org-x", name="Org X", instructions="i",
            level="org", organization=self.org,
        )
        AgentSkill.objects.create(
            slug="usr-x", name="My X", instructions="i",
            level="user", created_by=self.user,
        )
        self.org.preferences = {"skills": {"sys-x": {"enabled": True}}}
        self.org.save(update_fields=["preferences"])

    def test_requires_login(self):
        response = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(response.status_code, 302)

    def test_lists_three_sections(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Your skills")
        self.assertContains(response, "Acme")
        self.assertContains(response, "System skills")
        self.assertContains(response, "Sys X")
        self.assertContains(response, "Org X")
        self.assertContains(response, "My X")

    def test_disabled_system_skill_hidden(self):
        self.org.preferences = {"skills": {}}
        self.org.save(update_fields=["preferences"])
        self.client.force_login(self.user)
        response = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Sys X")

    def test_non_member_sees_no_org_section(self):
        outsider = User.objects.create_user(email="out@example.com", password="pw")
        outsider.email_verified = True
        outsider.save(update_fields=["email_verified"])
        self.client.force_login(outsider)
        response = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Acme")
        self.assertNotContains(response, "Org X")


@override_settings(ALLOWED_HOSTS=["testserver"])
class AnnotateSkillsTests(TestCase):
    """The skills-list row annotation: can_edit is derived from a precomputed
    is_org_admin flag (no per-row Membership query)."""

    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="ann@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Acme", slug="acme")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)
        AgentSkill.objects.create(slug="sys-x", name="Sys X", instructions="i", level="system")
        AgentSkill.objects.create(
            slug="org-x", name="Org X", instructions="i", level="org", organization=self.org,
        )
        AgentSkill.objects.create(
            slug="usr-x", name="My X", instructions="i", level="user", created_by=self.user,
        )
        self.org.preferences = {"skills": {"sys-x": {"enabled": True}}}
        self.org.save(update_fields=["preferences"])

    def _rows(self, level, is_org_admin):
        from agent_skills.services import get_accessible_skills, get_user_skill_prefs
        from agent_skills.views import _annotate_skills

        accessible = get_accessible_skills(self.user)
        prefs = get_user_skill_prefs(self.user)
        skills = [s for s in accessible if s.level == level]
        return _annotate_skills(self.user, skills, accessible, prefs, is_org_admin)

    def test_org_skill_editable_only_for_admin(self):
        self.assertTrue(all(not r["can_edit"] for r in self._rows("org", is_org_admin=False)))
        self.assertTrue(all(r["can_edit"] for r in self._rows("org", is_org_admin=True)))

    def test_user_skill_always_editable(self):
        # Ownership-based; independent of the org-admin flag.
        self.assertTrue(all(r["can_edit"] for r in self._rows("user", is_org_admin=False)))

    def test_system_skill_never_editable(self):
        self.assertTrue(all(not r["can_edit"] for r in self._rows("system", is_org_admin=True)))

    def test_list_view_does_not_call_can_edit_skill_per_row(self):
        """Regression: the list path inlines can_edit, so the per-row query
        helper (services.can_edit_skill) must not be invoked during render."""
        from unittest.mock import patch

        self.client.force_login(self.user)
        with patch("agent_skills.views.can_edit_skill") as mock_can_edit:
            response = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(response.status_code, 200)
        mock_can_edit.assert_not_called()


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsCreateViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="c@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Co", slug="co")

    def test_creates_user_skill(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_create"), {"name": "Built one"}
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            AgentSkill.objects.filter(
                level="user", created_by=self.user, name="Built one"
            ).exists()
        )

    def test_member_cannot_create_org_skill(self):
        Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.MEMBER
        )
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_create_org"), {"name": "Forbidden"}
        )
        self.assertEqual(response.status_code, 403)

    def test_admin_creates_org_skill(self):
        Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.ADMIN
        )
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_create_org"), {"name": "Admin made"}
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            AgentSkill.objects.filter(
                level="org", organization=self.org, name="Admin made"
            ).exists()
        )


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsDetailViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="d@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.skill = AgentSkill.objects.create(
            slug="my", name="My Skill", instructions="hello",
            level="user", created_by=self.user,
        )

    def test_owner_sees_editable_form(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("agent_skills_detail", kwargs={"skill_id": self.skill.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "My Skill")
        self.assertNotContains(response, "readonly")

    def test_non_owner_redirected(self):
        outsider = User.objects.create_user(email="o@example.com", password="pw")
        outsider.email_verified = True
        outsider.save(update_fields=["email_verified"])
        self.client.force_login(outsider)
        response = self.client.get(
            reverse("agent_skills_detail", kwargs={"skill_id": self.skill.id})
        )
        self.assertEqual(response.status_code, 302)

    def test_system_skill_read_only_but_has_preview_toggles(self):
        # System skills are not editable, but the detail page still
        # offers the markdown preview toggles so users can read the rendered
        # version (which the client-side JS shows by default).
        system_skill = AgentSkill.objects.create(
            slug="sys", name="System Skill", instructions="# hi", level="system",
        )
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("agent_skills_detail", kwargs={"skill_id": system_skill.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["editable"])
        # Read-only fields…
        self.assertContains(response, "readonly")
        # …but the preview toggle buttons are still rendered.
        self.assertContains(response, 'id="instructions-preview-btn"')
        self.assertContains(response, 'id="description-preview-btn"')


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsSaveViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="s@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.skill = AgentSkill.objects.create(
            slug="my", name="My Skill", instructions="hello",
            level="user", created_by=self.user,
        )

    def _post(self, action, **extra):
        payload = {
            "action": action,
            "name": extra.get("name", "Updated"),
            "emoji": extra.get("emoji", ""),
            "description": extra.get("description", "desc"),
            "instructions": extra.get("instructions", "new instructions"),
            "tool_names_json": extra.get("tool_names_json", "[]"),
            "templates_json": extra.get("templates_json", "[]"),
        }
        return self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": self.skill.id}),
            payload,
        )

    def test_save_updates_in_place(self):
        self.client.force_login(self.user)
        response = self._post("save")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], reverse("agent_skills_list"))
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.name, "Updated")
        self.assertEqual(self.skill.instructions, "new instructions")

    def test_save_strips_non_skill_tool_names(self):
        self.client.force_login(self.user)
        self._post("save", tool_names_json=json.dumps(
            ["skill_template_view", "chat_task_update", "totally_made_up"]
        ))
        self.skill.refresh_from_db()
        # Only the skills-section tool survives.
        self.assertEqual(self.skill.tool_names, ["skill_template_view"])

    def test_save_caps_oversized_instructions(self):
        self.client.force_login(self.user)
        self._post("save", instructions="z" * (MAX_INSTRUCTIONS_CHARS + 1000))
        self.skill.refresh_from_db()
        self.assertEqual(len(self.skill.instructions), MAX_INSTRUCTIONS_CHARS)

    def test_save_persists_emoji(self):
        self.client.force_login(self.user)
        response = self._post("save", emoji="🧪")
        self.assertEqual(response.status_code, 302)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.emoji, "🧪")

    def test_save_clears_emoji_when_empty(self):
        self.skill.emoji = "🧪"
        self.skill.save(update_fields=["emoji"])
        self.client.force_login(self.user)
        response = self._post("save", emoji="")
        self.assertEqual(response.status_code, 302)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.emoji, "")

    def test_save_as_user_creates_copy(self):
        self.client.force_login(self.user)
        before = AgentSkill.objects.filter(level="user").count()
        response = self._post("save_as_user", name="Forked")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], reverse("agent_skills_list"))
        self.assertEqual(AgentSkill.objects.filter(level="user").count(), before + 1)
        # Original is unchanged.
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.name, "My Skill")

    # NOTE: resources (formerly templates) are no longer managed by the Save
    # form — they have dedicated upload/create/update/delete endpoints, covered
    # by SkillResourceEndpointTests. The old form-reconciliation tests were
    # removed with that behavior.

    def test_save_slug_rename_migrates_prefs(self):
        """Renaming the slug on the detail form carries the user's slug-keyed
        selection over (prefs live under preferences["skills"][slug])."""
        us, _ = UserSettings.objects.get_or_create(user=self.user)
        us.preferences = {
            "skills": {"my": {"selected_skill_id": str(self.skill.id)}}
        }
        us.save(update_fields=["preferences"])

        self.client.force_login(self.user)
        payload = {
            "action": "save",
            "name": "My Skill",
            "slug": "my-renamed",
            "description": "desc",
            "instructions": "hello",
            "tool_names_json": "[]",
            "templates_json": "[]",
        }
        response = self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": self.skill.id}),
            payload,
        )
        self.assertEqual(response.status_code, 302)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.slug, "my-renamed")
        us.refresh_from_db()
        skills_prefs = us.preferences["skills"]
        self.assertNotIn("my", skills_prefs)
        self.assertEqual(
            skills_prefs["my-renamed"]["selected_skill_id"], str(self.skill.id)
        )

    def test_non_owner_save_forbidden(self):
        outsider = User.objects.create_user(email="o2@example.com", password="pw")
        outsider.email_verified = True
        outsider.save(update_fields=["email_verified"])
        self.client.force_login(outsider)
        response = self._post("save")
        # get_skill_for_user returns None → redirect to list, not 403.
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], reverse("agent_skills_list"))


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsCopyWorkflowTests(TransactionTestCase):
    """End-to-end tests for the detail-page Copy buttons.

    Uses ``TransactionTestCase`` so each ORM call commits in autocommit
    mode — matching production. With a regular ``TestCase`` the outer
    transaction would be poisoned by an ``IntegrityError`` raised inside
    ``_apply_skill_form``, masking the real-world behaviour.
    """

    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="cw@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="CW Org", slug="cw-org")
        Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.ADMIN
        )

        # System skill with two templates and a tool — exactly the kind of
        # rich source skill a user would copy from the detail page.
        self.source = AgentSkill.objects.create(
            slug="rich-source",
            name="Rich Source",
            description="A rich source skill.",
            instructions="Do the rich thing.",
            tool_names=["skill_create"],
            level="system",
        )
        self.t1 = SkillTemplate.objects.create(
            skill=self.source, name="Template A", content="Body A"
        )
        self.t2 = SkillTemplate.objects.create(
            skill=self.source, name="Template B", content="Body B"
        )
        self.org.preferences = {"skills": {"rich-source": {"enabled": True}}}
        self.org.save(update_fields=["preferences"])

    def _detail_payload(self, action: str, **overrides):
        """Build the POST payload the detail page would submit.

        Mirrors what ``skills_detail.html`` renders into the form: the
        ``templates_json`` carries the *source* template UUIDs (because
        the JS reads them straight from the server-rendered hidden input).
        """
        payload = {
            "action": action,
            "name": overrides.get("name", self.source.name),
            "description": overrides.get("description", self.source.description),
            "instructions": overrides.get("instructions", self.source.instructions),
            "tool_names_json": overrides.get(
                "tool_names_json", json.dumps(list(self.source.tool_names))
            ),
            "templates_json": overrides.get(
                "templates_json",
                json.dumps([
                    {"id": str(self.t1.id), "name": self.t1.name, "content": self.t1.content},
                    {"id": str(self.t2.id), "name": self.t2.name, "content": self.t2.content},
                ]),
            ),
        }
        return payload

    def test_save_as_user_preserves_templates(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": self.source.id}),
            self._detail_payload("save_as_user"),
        )
        self.assertEqual(response.status_code, 302)

        copies = AgentSkill.objects.filter(level="user", created_by=self.user)
        self.assertEqual(copies.count(), 1)
        copy = copies.first()
        self.assertEqual(copy.name, "Rich Source")
        self.assertEqual(copy.instructions, "Do the rich thing.")
        self.assertEqual(copy.description, "A rich source skill.")
        self.assertEqual(copy.tool_names, ["skill_create"])
        self.assertEqual(copy.parent, self.source)

        templates = list(copy.templates.order_by("name"))
        self.assertEqual(
            len(templates), 2,
            "Templates were lost when copying via the detail page form.",
        )
        self.assertEqual(templates[0].name, "Template A")
        self.assertEqual(templates[0].content, "Body A")
        self.assertEqual(templates[1].name, "Template B")
        self.assertEqual(templates[1].content, "Body B")

        # Source must be untouched.
        self.assertEqual(self.source.templates.count(), 2)

    def test_save_as_org_preserves_templates(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": self.source.id}),
            self._detail_payload("save_as_org"),
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], reverse("agent_skills_list"))

        org_copies = AgentSkill.objects.filter(level="org", organization=self.org)
        self.assertEqual(org_copies.count(), 1)
        copy = org_copies.first()
        self.assertEqual(copy.name, "Rich Source")
        self.assertEqual(copy.instructions, "Do the rich thing.")
        self.assertEqual(copy.description, "A rich source skill.")
        self.assertEqual(copy.tool_names, ["skill_create"])
        self.assertEqual(copy.parent, self.source)

        templates = list(copy.templates.order_by("name"))
        self.assertEqual(
            len(templates), 2,
            "Templates were lost when promoting via the detail page form.",
        )
        self.assertEqual(templates[0].name, "Template A")
        self.assertEqual(templates[0].content, "Body A")
        self.assertEqual(templates[1].name, "Template B")
        self.assertEqual(templates[1].content, "Body B")

        # Source must be untouched.
        self.assertEqual(self.source.templates.count(), 2)

    def test_save_as_user_uses_edited_form_values(self):
        """The copy reflects the form's edited text fields; resources are copied
        from the source unchanged (resources aren't edited via the Save form)."""
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": self.source.id}),
            self._detail_payload(
                "save_as_user",
                name="My Edited Copy",
                instructions="Edited instructions.",
            ),
        )
        self.assertEqual(response.status_code, 302)

        copy = AgentSkill.objects.get(level="user", created_by=self.user)
        self.assertEqual(copy.name, "My Edited Copy")
        self.assertEqual(copy.instructions, "Edited instructions.")
        # Resources came across from the source unchanged.
        resources = {t.name: t.content for t in copy.templates.all()}
        self.assertEqual(resources, {"Template A": "Body A", "Template B": "Body B"})

    def test_standalone_copy_url_preserves_templates(self):
        """The dropdown-menu Copy URL must also preserve templates."""
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_copy", kwargs={"skill_id": self.source.id})
        )
        self.assertEqual(response.status_code, 302)

        copy = AgentSkill.objects.get(level="user", created_by=self.user)
        self.assertEqual(copy.templates.count(), 2)

    def test_standalone_copy_to_org_url_preserves_templates(self):
        """The dropdown Copy-as-org URL (shown for system skills) preserves templates.

        A system skill can't be *moved* into the org (promote), so its dropdown
        offers a copy-to-org instead — which must carry templates across.
        """
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_copy_to_org", kwargs={"skill_id": self.source.id})
        )
        self.assertEqual(response.status_code, 302)

        copy = AgentSkill.objects.get(level="org", organization=self.org)
        self.assertEqual(copy.templates.count(), 2)

    def _make_org_skill(self):
        skill = AgentSkill.objects.create(
            slug="org-keeper", name="Org Keeper", instructions="i",
            level="org", organization=self.org,
        )
        SkillTemplate.objects.create(skill=skill, name="Keep", content="kept")
        return skill

    # (Removed: the save_as_org "duplicate template name → validation error"
    # regression test — the Save form no longer reconciles resources, so that
    # error path no longer exists. Resource errors live on the resource
    # endpoints now, covered by SkillResourceEndpointTests.)

    def test_save_as_org_on_existing_org_skill_saves_in_place(self):
        """save_as_org on an org skill in the same org edits it in place —
        no duplicate row, and the message says 'Saved' (no copy was made)."""
        org_skill = self._make_org_skill()
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": org_skill.id}),
            self._detail_payload(
                "save_as_org", name="Renamed Keeper", templates_json="[]"
            ),
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            AgentSkill.objects.filter(
                level="org", organization=self.org, slug="org-keeper"
            ).count(),
            1,
        )
        org_skill.refresh_from_db()
        self.assertEqual(org_skill.name, "Renamed Keeper")
        msgs = [str(m) for m in get_messages(response.wsgi_request)]
        self.assertIn("Saved 'Renamed Keeper'.", msgs)


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsCopyDeleteToggleTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="cdt@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="O", slug="o")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)
        self.sys_skill = AgentSkill.objects.create(
            slug="sys-only", name="Sys Only", instructions="i", level="system",
        )
        self.user_skill = AgentSkill.objects.create(
            slug="my-only", name="Mine", instructions="i",
            level="user", created_by=self.user,
        )
        self.org.preferences = {"skills": {"sys-only": {"enabled": True}}}
        self.org.save(update_fields=["preferences"])

    def test_copy_creates_user_skill(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_copy", kwargs={"skill_id": self.sys_skill.id})
        )
        self.assertEqual(response.status_code, 302)
        copies = AgentSkill.objects.filter(
            level="user", created_by=self.user, name="Sys Only"
        )
        self.assertEqual(copies.count(), 1)
        self.assertEqual(copies.first().parent, self.sys_skill)

    def test_delete_user_skill(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_delete", kwargs={"skill_id": self.user_skill.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(AgentSkill.objects.filter(pk=self.user_skill.pk).exists())

    def test_delete_system_skill_forbidden(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_delete", kwargs={"skill_id": self.sys_skill.id})
        )
        self.assertEqual(response.status_code, 403)
        self.assertTrue(AgentSkill.objects.filter(pk=self.sys_skill.pk).exists())

    def test_toggle_disable(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_toggle", kwargs={"skill_id": self.user_skill.id}),
            {"enabled": "0"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertFalse(data["now_active"])
        self.assertIsNone(data["replaced"])

        us = UserSettings.objects.get(user=self.user)
        self.assertIsNone(us.preferences["skills"]["my-only"]["selected_skill_id"])

    def test_toggle_enable_replaces_default(self):
        # Create org skill with the same slug as user_skill so user_skill
        # currently shadows it. Enable the org version explicitly.
        org_skill = AgentSkill.objects.create(
            slug="my-only", name="Org Mine", instructions="i",
            level="org", organization=self.org,
        )
        self.client.force_login(self.user)
        # This test is about shadow-replacement, not scanning — treat the skill
        # as already approved so the enable doesn't run the scan gate.
        from unittest.mock import patch
        with patch("agent_skills.resources.skill_is_approved", return_value=True):
            response = self.client.post(
                reverse("agent_skills_toggle", kwargs={"skill_id": org_skill.id}),
                {"enabled": "1"},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["now_active"])
        self.assertIsNotNone(data["replaced"])
        self.assertEqual(data["replaced"]["id"], str(self.user_skill.id))


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsPromoteViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.admin = User.objects.create_user(email="adm@example.com", password="pw")
        self.admin.email_verified = True
        self.admin.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="P", slug="p")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        self.user_skill = AgentSkill.objects.create(
            slug="great", name="Great", instructions="i",
            level="user", created_by=self.admin,
        )

    def test_admin_promote(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("agent_skills_promote", kwargs={"skill_id": self.user_skill.id})
        )
        self.assertEqual(response.status_code, 302)
        # Promote moves the skill (changes its type) — same row, now org-level,
        # with nothing left behind at the personal tier.
        self.user_skill.refresh_from_db()
        self.assertEqual(self.user_skill.level, "org")
        self.assertEqual(self.user_skill.organization, self.org)
        self.assertIsNone(self.user_skill.created_by)
        self.assertFalse(
            AgentSkill.objects.filter(level="user", name="Great").exists()
        )

    def test_member_cannot_promote(self):
        member = User.objects.create_user(email="mem@example.com", password="pw")
        member.email_verified = True
        member.save(update_fields=["email_verified"])
        Membership.objects.create(user=member, org=self.org, role=Membership.Role.MEMBER)
        own_skill = AgentSkill.objects.create(
            slug="x", name="X", instructions="i",
            level="user", created_by=member,
        )
        self.client.force_login(member)
        response = self.client.post(
            reverse("agent_skills_promote", kwargs={"skill_id": own_skill.id})
        )
        self.assertEqual(response.status_code, 403)


@override_settings(ALLOWED_HOSTS=["testserver"])
class OrgDisabledSkillViewTests(TestCase):
    """System skills disabled in org preferences are invisible everywhere
    except the org settings page (covered in accounts tests)."""

    def setUp(self):
        AgentSkill.objects.all().delete()
        self.org = Organization.objects.create(
            name="Disabled Org", slug="dis-org",
            preferences={"skills": {"research": {"enabled": False}}},
        )
        self.member = User.objects.create_user(email="dm@example.com", password="pw")
        self.member.email_verified = True
        self.member.save(update_fields=["email_verified"])
        Membership.objects.create(user=self.member, org=self.org, role=Membership.Role.MEMBER)

        self.skill = AgentSkill.objects.create(
            slug="research", name="Research Skill", instructions="i", level="system",
        )

    def test_skills_list_hides_disabled_system_skill(self):
        self.client.force_login(self.member)
        response = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Research Skill")

    def test_skills_detail_redirects_for_disabled_skill(self):
        self.client.force_login(self.member)
        response = self.client.get(
            reverse("agent_skills_detail", kwargs={"skill_id": self.skill.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("agent_skills_list"))

    def test_skills_toggle_returns_404_for_disabled_skill(self):
        self.client.force_login(self.member)
        response = self.client.post(
            reverse("agent_skills_toggle", kwargs={"skill_id": self.skill.id}),
            {"enabled": "1"},
        )
        self.assertEqual(response.status_code, 404)

    def test_superuser_member_also_blocked(self):
        admin = User.objects.create_user(
            email="su@example.com", password="pw", is_superuser=True,
        )
        admin.email_verified = True
        admin.save(update_fields=["email_verified"])
        Membership.objects.create(user=admin, org=self.org, role=Membership.Role.ADMIN)
        self.client.force_login(admin)
        response = self.client.get(reverse("agent_skills_list"))
        self.assertNotContains(response, "Research Skill")


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillResourceEndpointTests(TestCase):
    def setUp(self):
        from unittest.mock import patch

        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="res@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.skill = AgentSkill.objects.create(
            slug="r", name="R", instructions="i", level="user", created_by=self.user,
        )
        self.client.force_login(self.user)
        self._clean = patch.multiple(
            "agent_skills.resources",
            _scan_text_guardrail=lambda *a, **k: ("allow", "", []),
            _scan_text_pii=lambda *a, **k: ({}, False, "", ""),
        )

    def _url(self, name, **kw):
        return reverse(name, kwargs={"skill_id": self.skill.id, **kw})

    def test_create_text_resource(self):
        resp = self.client.post(
            self._url("agent_skills_resource_create"),
            {"name": "Notes", "content": "Body", "kind": "reference"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["resource"]["name"], "Notes")
        self.assertEqual(self.skill.templates.count(), 1)

    def test_create_rejects_duplicate_name(self):
        self.client.post(
            self._url("agent_skills_resource_create"), {"name": "Dup", "content": "a"}
        )
        resp = self.client.post(
            self._url("agent_skills_resource_create"), {"name": "Dup", "content": "b"}
        )
        self.assertEqual(resp.json()["error"], "duplicate_name")
        self.assertEqual(self.skill.templates.count(), 1)

    def test_create_denied_for_non_owner(self):
        # Another user can't even see this user-level skill -> 404 (not 403), so
        # its existence isn't revealed.
        other = User.objects.create_user(email="o@example.com", password="pw")
        self.client.force_login(other)
        resp = self.client.post(
            self._url("agent_skills_resource_create"), {"name": "X", "content": "y"}
        )
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(self.skill.templates.count(), 0)

    def test_upload_text_file_processes_async(self):
        from unittest.mock import patch

        from django.core.files.uploadedfile import SimpleUploadedFile

        from agent_skills import resources as svc
        from agent_skills.models import SkillResource

        # Simulate the worker running the enqueued task inline, with a clean scan.
        def run_inline(rid, uid):
            svc.process_upload(SkillResource.objects.get(pk=rid), None)

        with self._clean, patch(
            "agent_skills.tasks.process_skill_resource_upload_task.delay",
            side_effect=run_inline,
        ) as delayed:
            resp = self.client.post(
                self._url("agent_skills_resource_upload"),
                {"file": SimpleUploadedFile("notes.txt", b"reference body", content_type="text/plain")},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["resources"]), 1)
        self.assertEqual(data["resources"][0]["file_type"], "text")
        delayed.assert_called_once()
        # The (simulated) worker run extracted + scanned it clean.
        rid = data["resources"][0]["id"]
        res = SkillResource.objects.get(pk=rid)
        self.assertEqual(res.status, "ready")
        self.assertIn("reference body", res.content)

    def test_resource_status_endpoint(self):
        from agent_skills.models import SkillResource

        r = self.client.post(
            self._url("agent_skills_resource_create"), {"name": "N", "content": "c"}
        ).json()["resource"]
        SkillResource.objects.filter(pk=r["id"]).update(status="processing")
        resp = self.client.get(self._url("agent_skills_resource_status"))
        self.assertEqual(resp.status_code, 200)
        statuses = {x["id"]: x["status"] for x in resp.json()["resources"]}
        self.assertEqual(statuses[r["id"]], "processing")

    def test_upload_unsupported_type_reported(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        resp = self.client.post(
            self._url("agent_skills_resource_upload"),
            {"file": SimpleUploadedFile("clip.mp3", b"\x00\x01", content_type="audio/mpeg")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["resources"], [])
        self.assertTrue(any("unsupported" in e for e in data["errors"]))

    def test_update_renames_and_edits_content(self):
        r = self.client.post(
            self._url("agent_skills_resource_create"), {"name": "A", "content": "x"}
        ).json()["resource"]
        resp = self.client.post(
            reverse("agent_skills_resource_update",
                    kwargs={"skill_id": self.skill.id, "resource_id": r["id"]}),
            {"name": "B", "content": "y"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["resource"]
        self.assertEqual(data["name"], "B")
        self.assertEqual(data["content"], "y")

    def test_delete_resource(self):
        r = self.client.post(
            self._url("agent_skills_resource_create"), {"name": "A", "content": "x"}
        ).json()["resource"]
        resp = self.client.post(
            reverse("agent_skills_resource_delete",
                    kwargs={"skill_id": self.skill.id, "resource_id": r["id"]}),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.skill.templates.count(), 0)

    def test_toggle_enable_blocks_when_scan_blocks(self):
        from unittest.mock import patch

        with patch("agent_skills.resources.skill_is_approved", return_value=False), \
                patch("agent_skills.resources.scan_and_approve_skill", return_value=False):
            resp = self.client.post(
                reverse("agent_skills_toggle", kwargs={"skill_id": self.skill.id}),
                {"enabled": "1"},
            )
        data = resp.json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"], "blocked")

    def test_toggle_enable_runs_scan_then_enables(self):
        from unittest.mock import patch

        with patch("agent_skills.resources.skill_is_approved", return_value=False), \
                patch("agent_skills.resources.scan_and_approve_skill", return_value=True) as scan:
            resp = self.client.post(
                reverse("agent_skills_toggle", kwargs={"skill_id": self.skill.id}),
                {"enabled": "1"},
            )
        self.assertTrue(resp.json()["ok"])
        scan.assert_called_once()
