"""Data-migration test for 0014 (Skill Creator's tool toggles → the toolkit).

An org's per-tool switches for the moved ``skill_*`` tools are copied from the
``skill-creator`` entry to ``skill_management_tools``; an existing toggle under
the toolkit wins, other tools and orgs are untouched.
"""

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

from accounts.models import Organization

MIGRATE_FROM = [("agent_skills", "0013_rename_toolkit_slugs")]
MIGRATE_TO = [("agent_skills", "0014_skill_management_toolkit_prefs")]


class SkillManagementPrefsTests(TransactionTestCase):
    def test_tool_toggles_are_copied(self):
        org_a = Organization.objects.create(name="A", slug="org-a", preferences={
            "skills": {
                "skill-creator": {
                    "enabled": True,
                    "tools": {"skill_delete": False, "canvas_write": False},
                },
            },
        })
        org_b = Organization.objects.create(name="B", slug="org-b", preferences={
            "skills": {
                "skill-creator": {"tools": {"skill_delete": False, "skill_edit": False}},
                "skill_management_tools": {"enabled": True, "tools": {"skill_delete": True}},
            },
        })
        org_c = Organization.objects.create(name="C", slug="org-c", preferences={
            "skills": {"skill-creator": {"enabled": False}},
        })

        executor = MigrationExecutor(connection)
        executor.migrate(MIGRATE_FROM)
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(MIGRATE_TO)

        org_a.refresh_from_db()
        skills = org_a.preferences["skills"]
        # Only the moved tools are copied; the old entry is left alone.
        self.assertEqual(skills["skill_management_tools"], {"tools": {"skill_delete": False}})
        self.assertEqual(
            skills["skill-creator"]["tools"], {"skill_delete": False, "canvas_write": False}
        )

        # An existing toggle under the toolkit wins; missing ones are filled in.
        org_b.refresh_from_db()
        self.assertEqual(
            org_b.preferences["skills"]["skill_management_tools"],
            {"enabled": True, "tools": {"skill_delete": True, "skill_edit": False}},
        )

        # No tool toggles: nothing written (enabled is never carried over).
        org_c.refresh_from_db()
        self.assertNotIn("skill_management_tools", org_c.preferences["skills"])

    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())
