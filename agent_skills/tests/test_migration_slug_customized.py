"""Backfill test for the slug_customized migration (0010).

New behavior auto-follows the name on rename while slug_customized is False. The
0010 backfill must freeze existing user skills whose slug was clearly hand-picked
(so they don't drift on the next rename) while leaving auto / placeholder slugs
free to self-correct. Org/system skills must be left untouched.
"""

from django.contrib.auth import get_user_model
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

from accounts.models import Organization

User = get_user_model()

MIGRATE_FROM = [("agent_skills", "0009_agentskill_soft_delete")]
MIGRATE_TO = [("agent_skills", "0010_agentskill_slug_customized")]


class SlugCustomizedBackfillTests(TransactionTestCase):
    def test_backfill_freezes_only_custom_user_slugs(self):
        user = User.objects.create_user(email="mig@example.com", password="pass")
        org = Organization.objects.create(name="Mig Org", slug="mig-org")

        # --- roll agent_skills back to before the new field ---
        executor = MigrationExecutor(connection)
        executor.migrate(MIGRATE_FROM)
        old_state = executor.loader.project_state(MIGRATE_FROM).apps
        OldSkill = old_state.get_model("agent_skills", "AgentSkill")

        in_sync = OldSkill.objects.create(
            slug="foo-bar", name="Foo Bar", instructions="i",
            level="user", created_by_id=user.id,
        )
        placeholder = OldSkill.objects.create(
            slug="untitled-skill", name="My Real Skill", instructions="i",
            level="user", created_by_id=user.id,
        )
        deduped = OldSkill.objects.create(
            slug="dupe-1", name="Dupe", instructions="i",
            level="user", created_by_id=user.id,
        )
        custom = OldSkill.objects.create(
            slug="hand-picked", name="My Skill", instructions="i",
            level="user", created_by_id=user.id,
        )
        # Org skill with a custom-looking slug — must be left untouched.
        org_skill = OldSkill.objects.create(
            slug="org-custom", name="Org Skill", instructions="i",
            level="org", organization_id=org.id,
        )

        # --- apply the migration under test ---
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(MIGRATE_TO)
        new_state = executor.loader.project_state(MIGRATE_TO).apps
        NewSkill = new_state.get_model("agent_skills", "AgentSkill")

        self.assertFalse(NewSkill.objects.get(pk=in_sync.pk).slug_customized)
        self.assertFalse(NewSkill.objects.get(pk=placeholder.pk).slug_customized)
        self.assertFalse(NewSkill.objects.get(pk=deduped.pk).slug_customized)
        self.assertTrue(NewSkill.objects.get(pk=custom.pk).slug_customized)
        # Org skill never processed (backfill filters level="user").
        self.assertFalse(NewSkill.objects.get(pk=org_skill.pk).slug_customized)

    def tearDown(self):
        # Leave the schema at HEAD for any following tests (see the 0006
        # migration test for why TransactionTestCase teardown needs this).
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())
