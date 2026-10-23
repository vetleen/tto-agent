"""Data-migration test for 0013 (toolkit slug renames).

The three renamed toolkits keep their row ids (thread attachments and forks'
``parent`` links survive), get ``kind="toolkit"``, and the slug-keyed org / user
preference stores are re-keyed with the documented copy-vs-move rules.
"""

from django.contrib.auth import get_user_model
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

from accounts.models import Organization, UserSettings
from chat.models import ChatThread, ChatThreadSkill

User = get_user_model()

MIGRATE_FROM = [("agent_skills", "0012_agentskill_kind_requires")]
MIGRATE_TO = [("agent_skills", "0013_rename_toolkit_slugs")]


def _prefs(user):
    return UserSettings.objects.get(user=user).preferences["skills"]


class ToolkitSlugRenameTests(TransactionTestCase):
    def _set_user_prefs(self, user, skills):
        UserSettings.objects.update_or_create(
            user=user, defaults={"preferences": {"skills": skills}},
        )

    def test_rename_rekeys_rows_and_prefs(self):
        u1 = User.objects.create_user(email="u1@example.com", password="pass")
        u2 = User.objects.create_user(email="u2@example.com", password="pass")
        u3 = User.objects.create_user(email="u3@example.com", password="pass")
        org_a = Organization.objects.create(name="A", slug="org-a", preferences={
            "skills": {
                "canvas_collaborator": {"enabled": True, "tools": {"canvas_read": False}},
                "image_generator": {"enabled": True},
                "skill-creator": {"enabled": True},
            },
        })
        org_b = Organization.objects.create(name="B", slug="org-b", preferences={
            "skills": {"canvas_collaborator": {"enabled": False}},
        })

        # --- roll back to before the rename (the seeded rows get their old slugs) ---
        executor = MigrationExecutor(connection)
        executor.migrate(MIGRATE_FROM)
        old_state = executor.loader.project_state(MIGRATE_FROM).apps
        OldSkill = old_state.get_model("agent_skills", "AgentSkill")

        system, _ = OldSkill.objects.get_or_create(
            slug="canvas_collaborator", level="system",
            defaults={"name": "Canvas Collaborator", "instructions": "i"},
        )
        OldSkill.objects.filter(pk=system.pk).update(kind="playbook")
        fork = OldSkill.objects.create(
            slug="canvas_collaborator", name="My canvas", instructions="i",
            level="user", created_by_id=u3.id, parent_id=system.id,
        )
        org_b_skill = OldSkill.objects.create(
            slug="canvas_collaborator", name="B canvas", instructions="i",
            level="org", organization_id=org_b.id,
        )
        thread = ChatThread.objects.create(created_by=u1, title="t")
        ChatThreadSkill.objects.create(thread=thread, skill_id=system.id)

        self._set_user_prefs(u1, {"canvas_collaborator": {"selected_skill_id": str(system.id)}})
        self._set_user_prefs(u2, {"canvas_collaborator": {"selected_skill_id": None}})
        self._set_user_prefs(u3, {"canvas_collaborator": {"selected_skill_id": str(fork.id)}})

        # --- forward ---
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(MIGRATE_TO)
        new_state = executor.loader.project_state(MIGRATE_TO).apps
        NewSkill = new_state.get_model("agent_skills", "AgentSkill")

        renamed = NewSkill.objects.get(pk=system.pk)
        self.assertEqual(renamed.slug, "canvas_tools")
        self.assertEqual(renamed.kind, "toolkit")
        self.assertTrue(
            ChatThreadSkill.objects.filter(thread=thread, skill_id=system.id).exists()
        )
        # Forks and org copies keep the old slug.
        self.assertEqual(NewSkill.objects.get(pk=fork.pk).slug, "canvas_collaborator")
        self.assertEqual(NewSkill.objects.get(pk=org_b_skill.pk).slug, "canvas_collaborator")

        org_a.refresh_from_db()
        self.assertEqual(
            org_a.preferences["skills"]["canvas_tools"],
            {"enabled": True, "tools": {"canvas_read": False}},
        )
        self.assertNotIn("canvas_collaborator", org_a.preferences["skills"])
        self.assertEqual(org_a.preferences["skills"]["image_generation_tools"], {"enabled": True})
        self.assertNotIn("image_generator", org_a.preferences["skills"])
        self.assertEqual(org_a.preferences["skills"]["skill-creator"], {"enabled": True})
        # Org B owns an org skill under the old slug: the entry is copied, not moved.
        org_b.refresh_from_db()
        self.assertEqual(org_b.preferences["skills"]["canvas_collaborator"], {"enabled": False})
        self.assertEqual(org_b.preferences["skills"]["canvas_tools"], {"enabled": False})

        # u1 selected the system row: moved. u2 disabled the slug: copied.
        # u3 selected the fork: untouched.
        self.assertEqual(_prefs(u1), {"canvas_tools": {"selected_skill_id": str(system.id)}})
        self.assertEqual(_prefs(u2), {
            "canvas_collaborator": {"selected_skill_id": None},
            "canvas_tools": {"selected_skill_id": None},
        })
        self.assertEqual(_prefs(u3), {"canvas_collaborator": {"selected_skill_id": str(fork.id)}})

        # --- backward restores the slug and the moved keys ---
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(MIGRATE_FROM)
        back_state = executor.loader.project_state(MIGRATE_FROM).apps
        BackSkill = back_state.get_model("agent_skills", "AgentSkill")
        self.assertEqual(BackSkill.objects.get(pk=system.pk).slug, "canvas_collaborator")
        org_a.refresh_from_db()
        self.assertIn("canvas_collaborator", org_a.preferences["skills"])
        self.assertNotIn("canvas_tools", org_a.preferences["skills"])
        self.assertEqual(_prefs(u1), {"canvas_collaborator": {"selected_skill_id": str(system.id)}})

    def test_forward_skips_when_target_slug_exists(self):
        executor = MigrationExecutor(connection)
        executor.migrate(MIGRATE_FROM)
        OldSkill = executor.loader.project_state(MIGRATE_FROM).apps.get_model(
            "agent_skills", "AgentSkill"
        )
        old, _ = OldSkill.objects.get_or_create(
            slug="image_generator", level="system",
            defaults={"name": "Image Generator", "instructions": "i"},
        )
        OldSkill.objects.create(
            slug="image_generation_tools", name="Image Generation Tools",
            instructions="i", level="system",
        )

        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(MIGRATE_TO)
        NewSkill = executor.loader.project_state(MIGRATE_TO).apps.get_model(
            "agent_skills", "AgentSkill"
        )
        self.assertEqual(NewSkill.objects.get(pk=old.pk).slug, "image_generator")
        self.assertEqual(
            NewSkill.objects.filter(level="system", slug="image_generation_tools").count(), 1
        )

    def tearDown(self):
        # Leave the schema at HEAD for any following tests.
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())
