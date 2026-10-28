"""The 0015 data migration renames "mid" to "medium" in stored preferences."""

import importlib

from django.test import SimpleTestCase

migration = importlib.import_module("accounts.migrations.0015_medium_slot_rename_prefs")


class RemapPreferencesTests(SimpleTestCase):
    def test_forward_and_back(self):
        prefs = {
            "models": {"primary": "openai/gpt-6-sol", "mid": "openai/gpt-6-luna"},
            "feature_models": {"auto_mid": "openai/gpt-6-luna", "subagent_medium": "openai/gpt-6-luna"},
            "feature_reasoning": {"auto_mid": {"model": "openai/gpt-6-luna", "level": "low"}},
        }
        self.assertTrue(migration.remap_preferences(prefs))
        self.assertEqual(prefs["models"], {"primary": "openai/gpt-6-sol", "medium": "openai/gpt-6-luna"})
        self.assertEqual(
            prefs["feature_models"], {"auto_medium": "openai/gpt-6-luna", "subagent_medium": "openai/gpt-6-luna"},
        )
        self.assertEqual(prefs["feature_reasoning"], {"auto_medium": {"model": "openai/gpt-6-luna", "level": "low"}})
        self.assertFalse(migration.remap_preferences(prefs))

        self.assertTrue(migration.remap_preferences(prefs, backward=True))
        self.assertEqual(prefs["models"]["mid"], "openai/gpt-6-luna")
        self.assertIn("auto_mid", prefs["feature_models"])

    def test_new_key_wins(self):
        prefs = {"models": {"mid": "old", "medium": "new"}}
        migration.remap_preferences(prefs)
        self.assertEqual(prefs["models"], {"medium": "new"})

    def test_nothing_to_do(self):
        self.assertFalse(migration.remap_preferences({"theme": "dark", "models": "bogus"}))
