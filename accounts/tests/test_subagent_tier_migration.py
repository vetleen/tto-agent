"""The 0014 data migration renames sub-agent tier keys in org preferences."""

import importlib

from django.test import SimpleTestCase

migration = importlib.import_module("accounts.migrations.0014_subagent_tier_rename_prefs")


class RemapPreferencesTests(SimpleTestCase):
    def _prefs(self):
        return {
            "feature_models": {
                "chat": "openai/gpt-6-sol",
                "subagent_mid": "openai/gpt-6-luna",
                "subagent_top": "openai/gpt-6-sol",
            },
            "subagents": {
                "parallel": True,
                "context_budget": {"mid": 120_000, "top": 180_000},
                "reasoning": {"mid": {"model": "openai/gpt-6-luna", "level": "low"}},
            },
            "models": {"mid": "openai/gpt-6-luna"},  # slot vocabulary: untouched
        }

    def test_forward_renames_every_tier_key(self):
        prefs = self._prefs()
        self.assertTrue(migration.remap_preferences(prefs, migration._FORWARD))
        self.assertEqual(prefs["feature_models"], {
            "chat": "openai/gpt-6-sol",
            "subagent_medium": "openai/gpt-6-luna",
            "subagent_high": "openai/gpt-6-sol",
        })
        self.assertEqual(prefs["subagents"]["context_budget"], {"medium": 120_000, "high": 180_000})
        self.assertEqual(
            prefs["subagents"]["reasoning"],
            {"medium": {"model": "openai/gpt-6-luna", "level": "low"}},
        )
        self.assertTrue(prefs["subagents"]["parallel"])
        self.assertEqual(prefs["models"], {"mid": "openai/gpt-6-luna"})

    def test_backward_restores_the_original(self):
        prefs = self._prefs()
        migration.remap_preferences(prefs, migration._FORWARD)
        migration.remap_preferences(prefs, migration._BACKWARD)
        self.assertEqual(prefs, self._prefs())

    def test_existing_new_key_wins(self):
        prefs = {"subagents": {"context_budget": {"mid": 1, "medium": 2}}}
        migration.remap_preferences(prefs, migration._FORWARD)
        self.assertEqual(prefs["subagents"]["context_budget"], {"medium": 2})

    def test_nothing_to_rename(self):
        self.assertFalse(migration.remap_preferences({"theme": "dark"}, migration._FORWARD))
