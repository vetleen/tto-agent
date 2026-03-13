"""Tests for chat.prompts.build_system_prompt."""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

from django.test import TestCase

from chat.prompts import build_system_prompt


class BuildSystemPromptTests(TestCase):
    def setUp(self):
        self.data_room = {"id": 1, "name": "My Data Room"}

    def test_basic_prompt(self):
        prompt = build_system_prompt(data_rooms=[self.data_room])
        self.assertIn("My Data Room", prompt)

    def test_no_data_rooms_omits_tools(self):
        prompt = build_system_prompt()
        self.assertNotIn("search_documents", prompt)
        self.assertNotIn("read_document", prompt)

    def test_no_metadata_no_history_note(self):
        prompt = build_system_prompt(data_rooms=[self.data_room])
        self.assertNotIn("messages total", prompt)

    def test_metadata_when_all_included(self):
        meta = {
            "total_messages": 5,
            "included_messages": 5,
            "has_summary": False,
        }
        prompt = build_system_prompt(data_rooms=[self.data_room], history_meta=meta)
        # All messages included — no note
        self.assertNotIn("messages total", prompt)

    def test_metadata_when_some_excluded_no_summary(self):
        meta = {
            "total_messages": 100,
            "included_messages": 30,
            "has_summary": False,
        }
        prompt = build_system_prompt(data_rooms=[self.data_room], history_meta=meta)
        self.assertIn("100 messages total", prompt)
        self.assertIn("30 most recent", prompt)

    def test_metadata_with_summary(self):
        meta = {
            "total_messages": 100,
            "included_messages": 30,
            "has_summary": True,
        }
        prompt = build_system_prompt(data_rooms=[self.data_room], history_meta=meta)
        self.assertIn("100 messages total", prompt)
        self.assertIn("summary of earlier messages", prompt.lower())

    def test_organization_name_included(self):
        prompt = build_system_prompt(organization_name="MIT TTO")
        self.assertIn("at MIT TTO", prompt)

    def test_no_organization_omits_org_name(self):
        prompt = build_system_prompt()
        self.assertIn("at a technology transfer office", prompt)
        self.assertNotIn("MIT TTO", prompt)

    # ------------------------------------------------------------------ #
    # Data room descriptions in prompt                                    #
    # ------------------------------------------------------------------ #

    def test_data_room_description_in_prompt(self):
        room = {"id": 1, "name": "Patent Portfolio", "description": "Contains patent filings"}
        prompt = build_system_prompt(data_rooms=[room])
        self.assertIn("Patent Portfolio", prompt)
        self.assertIn("Contains patent filings", prompt)

    def test_data_room_without_description(self):
        room = {"id": 1, "name": "Empty Room", "description": ""}
        prompt = build_system_prompt(data_rooms=[room])
        self.assertIn("Empty Room", prompt)

    def test_multiple_data_rooms_with_descriptions(self):
        rooms = [
            {"id": 1, "name": "Room A", "description": "Desc A"},
            {"id": 2, "name": "Room B", "description": ""},
        ]
        prompt = build_system_prompt(data_rooms=rooms)
        self.assertIn("Room A", prompt)
        self.assertIn("Desc A", prompt)
        self.assertIn("Room B", prompt)

    # ------------------------------------------------------------------ #
    # Document type in document listing                                    #
    # ------------------------------------------------------------------ #

    def test_document_type_in_listing(self):
        doc_context = {
            "total_doc_count": 1,
            "documents": [{
                "doc_index": 1,
                "filename": "License.pdf",
                "description": "A license",
                "token_count": 500,
                "document_type": "Agreement",
            }],
        }
        prompt = build_system_prompt(
            data_rooms=[self.data_room],
            doc_context=doc_context,
        )
        self.assertIn("(Agreement)", prompt)
        self.assertIn("License.pdf", prompt)

    def test_document_without_type(self):
        doc_context = {
            "total_doc_count": 1,
            "documents": [{
                "doc_index": 1,
                "filename": "Unknown.pdf",
                "description": "",
                "token_count": 100,
                "document_type": "",
            }],
        }
        prompt = build_system_prompt(
            data_rooms=[self.data_room],
            doc_context=doc_context,
        )
        self.assertIn("Unknown.pdf", prompt)
        # Should not have empty parentheses
        self.assertNotIn("()", prompt)


    # ------------------------------------------------------------------ #
    # Skill injection                                                      #
    # ------------------------------------------------------------------ #

    def _make_skill(self, name, instructions, description=""):
        """Create a mock skill with a templates manager."""
        skill = MagicMock()
        skill.name = name
        skill.description = description
        skill.instructions = instructions
        skill.templates.all.return_value = []
        return skill

    def test_skill_instructions_injected(self):
        skill = self._make_skill("Patent Drafter", "Draft patents carefully.")
        prompt = build_system_prompt(skill=skill)
        self.assertIn("# Relevant skill", prompt)
        self.assertIn("## Patent Drafter", prompt)
        self.assertIn("Draft patents carefully.", prompt)

    def test_skill_description_included(self):
        skill = self._make_skill(
            "Patent Drafter",
            "Draft patents carefully.",
            description="Helps draft patent applications.",
        )
        prompt = build_system_prompt(skill=skill)
        self.assertIn("Helps draft patent applications.", prompt)

    def test_skill_no_description_no_blank(self):
        skill = self._make_skill("Patent Drafter", "Draft patents carefully.")
        prompt = build_system_prompt(skill=skill)
        # The description is empty so it should not leave an extra blank line
        self.assertNotIn("## Patent Drafter\n\n\n", prompt)

    def test_skill_appears_after_instructions_before_data_rooms(self):
        skill = self._make_skill("Test Skill", "Do the test.")
        prompt = build_system_prompt(skill=skill, data_rooms=[self.data_room])
        skill_pos = prompt.index("# Relevant skill")
        instructions_pos = prompt.index("# General instructions")
        data_rooms_pos = prompt.index("# Attached Data Rooms")
        self.assertGreater(skill_pos, instructions_pos)
        self.assertLess(skill_pos, data_rooms_pos)

    def test_skill_headers_deepened(self):
        skill = self._make_skill(
            "Deep Skill",
            "# Top heading\nSome text\n## Sub heading\nMore text",
        )
        prompt = build_system_prompt(skill=skill)
        self.assertIn("### Top heading", prompt)
        self.assertIn("#### Sub heading", prompt)
        # Original single-# should not appear outside of the skill name line
        lines = prompt.split("\n")
        for line in lines:
            if line.strip() == "# Top heading" or line.strip() == "## Top heading":
                self.fail("Expected '# Top heading' to be deepened to '### Top heading'")

    def test_no_skill_no_skill_section(self):
        prompt = build_system_prompt()
        self.assertNotIn("# Relevant skill", prompt)
