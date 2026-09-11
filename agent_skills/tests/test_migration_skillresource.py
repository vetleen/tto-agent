"""Data-survival test for the SkillTemplate -> SkillResource migration (0006).

The user's hard requirement: no existing skill data may be lost or corrupted by
the rename. This drives the real migration on a from->to schema pair with rows
created at the old state, then asserts they survive and are transformed.
"""

import hashlib

from django.contrib.auth import get_user_model
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

from accounts.models import Organization

User = get_user_model()

MIGRATE_FROM = [("agent_skills", "0005_agentskill_audience")]
MIGRATE_TO = [("agent_skills", "0006_skillresource")]


class SkillResourceMigrationTests(TransactionTestCase):
    def test_existing_templates_survive_and_transform(self):
        # accounts stays at head throughout, so create the owner up front.
        user = User.objects.create_user(email="mig@example.com", password="pass")
        Organization.objects.create(name="Mig Org", slug="mig-org")

        # --- roll agent_skills back to the pre-rename state ---
        executor = MigrationExecutor(connection)
        executor.migrate(MIGRATE_FROM)
        old_state = executor.loader.project_state(MIGRATE_FROM).apps
        OldSkill = old_state.get_model("agent_skills", "AgentSkill")
        OldTemplate = old_state.get_model("agent_skills", "SkillTemplate")

        sys_skill = OldSkill.objects.create(
            slug="sys-skill", name="System Skill",
            instructions="Do the system thing.", level="system",
        )
        tmpl = OldTemplate.objects.create(
            skill=sys_skill, name="Boilerplate", content="Hello world",
        )
        user_skill = OldSkill.objects.create(
            slug="user-skill", name="User Skill",
            instructions="Do the user thing.", description="A desc.",
            level="user", created_by_id=user.id,
        )

        # --- apply the migration under test ---
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(MIGRATE_TO)
        new_state = executor.loader.project_state(MIGRATE_TO).apps
        SkillResource = new_state.get_model("agent_skills", "SkillResource")
        NewSkill = new_state.get_model("agent_skills", "AgentSkill")

        # The template row survived the RenameModel, same PK, content intact.
        res = SkillResource.objects.get(pk=tmpl.pk)
        self.assertEqual(res.content, "Hello world")
        self.assertEqual(res.name, "Boilerplate")
        self.assertEqual(res.kind, "template")
        self.assertEqual(res.file_type, "text")
        self.assertEqual(res.status, "ready")
        self.assertEqual(
            res.content_sha256,
            hashlib.sha256(b"Hello world").hexdigest(),
        )

        # Existing user/org skills grandfathered as approved so the attach gate
        # never blocks them; system skills stay unscanned (approved in code).
        us = NewSkill.objects.get(pk=user_skill.pk)
        self.assertEqual(us.scan_state, "approved")
        self.assertTrue(us.approved_content_hash)

        sysn = NewSkill.objects.get(pk=sys_skill.pk)
        self.assertEqual(sysn.scan_state, "unscanned")

    def tearDown(self):
        # Leave the schema at head for any following tests.
        executor = MigrationExecutor(connection)
        executor.migrate(MIGRATE_TO)
