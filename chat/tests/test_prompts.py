"""Tests for chat.prompts.build_system_prompt."""

from unittest.mock import MagicMock

from django.test import TestCase

from chat.prompts import build_system_prompt


class BuildSystemPromptTests(TestCase):
    def setUp(self):
        self.project = MagicMock()
        self.project.name = "My Project"

    def test_basic_prompt(self):
        prompt = build_system_prompt(self.project)
        self.assertIn("My Project", prompt)
        self.assertIn("search_documents", prompt)

    def test_no_metadata_no_history_note(self):
        prompt = build_system_prompt(self.project)
        self.assertNotIn("messages total", prompt)

    def test_metadata_when_all_included(self):
        meta = {
            "total_messages": 5,
            "included_messages": 5,
            "has_summary": False,
        }
        prompt = build_system_prompt(self.project, history_meta=meta)
        # All messages included — no note
        self.assertNotIn("messages total", prompt)

    def test_metadata_when_some_excluded_no_summary(self):
        meta = {
            "total_messages": 100,
            "included_messages": 30,
            "has_summary": False,
        }
        prompt = build_system_prompt(self.project, history_meta=meta)
        self.assertIn("100 messages total", prompt)
        self.assertIn("30 most recent", prompt)
        self.assertNotIn("summary", prompt.lower().split("search_documents")[0]
                         if "search_documents" in prompt else "")

    def test_metadata_with_summary(self):
        meta = {
            "total_messages": 100,
            "included_messages": 30,
            "has_summary": True,
        }
        prompt = build_system_prompt(self.project, history_meta=meta)
        self.assertIn("100 messages total", prompt)
        self.assertIn("summary of earlier messages", prompt.lower())
