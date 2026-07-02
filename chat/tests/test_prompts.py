"""Tests for chat.prompts — build_system_prompt, build_static_system_prompt, build_semi_static_prompt, build_dynamic_context."""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

from django.conf import settings
from django.test import TestCase

from chat.prompts import (
    build_dynamic_context,
    build_last_message_preamble,
    build_loop_turn_delimiter,
    build_semi_static_prompt,
    build_static_system_prompt,
    build_system_prompt,
)


class BuildSystemPromptTests(TestCase):
    def setUp(self):
        self.data_room = {"id": 1, "name": "My Data Room"}

    def test_basic_prompt(self):
        prompt = build_system_prompt(data_rooms=[self.data_room])
        self.assertIn("My Data Room", prompt)

    def test_no_data_rooms_omits_tools(self):
        prompt = build_system_prompt()
        self.assertNotIn("document_search", prompt)
        self.assertNotIn("document_read", prompt)

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
        self.assertIn("an AI assistant.", prompt)
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
        prompt = build_system_prompt(skills=[skill])
        self.assertIn("# Relevant skills", prompt)
        self.assertIn("## Patent Drafter", prompt)
        self.assertIn("Draft patents carefully.", prompt)

    def test_skill_description_included(self):
        skill = self._make_skill(
            "Patent Drafter",
            "Draft patents carefully.",
            description="Helps draft patent applications.",
        )
        prompt = build_system_prompt(skills=[skill])
        self.assertIn("Helps draft patent applications.", prompt)

    def test_skill_no_description_no_blank(self):
        skill = self._make_skill("Patent Drafter", "Draft patents carefully.")
        prompt = build_system_prompt(skills=[skill])
        # The description is empty so it should not leave an extra blank line
        self.assertNotIn("## Patent Drafter\n\n\n", prompt)

    def test_skill_section_has_no_stray_code_comment(self):
        # Regression: a developer comment once lived inside the skill f-string
        # and leaked the literal text "#chr(10) produces a newline..." into the
        # prompt. The skill block must never contain that or a bare chr(10) ref.
        skill = self._make_skill(
            "Patent Drafter",
            "Draft patents carefully.",
            description="Helps draft patent applications.",
        )
        prompt = build_system_prompt(skills=[skill])
        self.assertNotIn("chr(10)", prompt)
        self.assertNotIn("produces a newline", prompt)

    def test_skill_appears_after_instructions_before_data_rooms(self):
        skill = self._make_skill("Test Skill", "Do the test.")
        prompt = build_system_prompt(skills=[skill], data_rooms=[self.data_room])
        skill_pos = prompt.index("# Relevant skills")
        instructions_pos = prompt.index("# General instructions")
        data_rooms_pos = prompt.index("# Attached Data Rooms")
        self.assertGreater(skill_pos, instructions_pos)
        self.assertLess(skill_pos, data_rooms_pos)

    def test_skill_headers_deepened(self):
        skill = self._make_skill(
            "Deep Skill",
            "# Top heading\nSome text\n## Sub heading\nMore text",
        )
        prompt = build_system_prompt(skills=[skill])
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

    def test_empty_skills_list_no_section(self):
        prompt = build_system_prompt(skills=[])
        self.assertNotIn("# Relevant skill", prompt)

    def test_multiple_skills_each_rendered(self):
        a = self._make_skill("Alpha", "Do alpha.", description="Alpha desc.")
        b = self._make_skill("Beta", "Do beta.")
        prompt = build_system_prompt(skills=[a, b])
        # Single pluralized header, one sub-block per skill.
        self.assertEqual(prompt.count("# Relevant skills"), 1)
        self.assertIn("## Alpha", prompt)
        self.assertIn("## Beta", prompt)
        self.assertIn("Do alpha.", prompt)
        self.assertIn("Do beta.", prompt)
        self.assertIn("Alpha desc.", prompt)
        # Order is preserved: Alpha before Beta.
        self.assertLess(prompt.index("## Alpha"), prompt.index("## Beta"))

    def test_multiple_skills_headers_deepened_per_skill(self):
        a = self._make_skill("Alpha", "# A head\ntext")
        b = self._make_skill("Beta", "# B head\ntext")
        prompt = build_system_prompt(skills=[a, b])
        self.assertIn("### A head", prompt)
        self.assertIn("### B head", prompt)

    # ------------------------------------------------------------------ #
    # Task planning prompt                                                 #
    # ------------------------------------------------------------------ #

    def test_task_planning_section_with_tool(self):
        prompt = build_system_prompt(has_task_tool=True)
        self.assertIn("# Task Planning", prompt)
        self.assertIn("chat_task_update", prompt)

    def test_task_planning_includes_when_to_use(self):
        prompt = build_system_prompt(has_task_tool=True)
        self.assertIn("When to create a task plan", prompt)
        self.assertIn("When NOT to create a task plan", prompt)

    def test_task_planning_includes_management_rules(self):
        prompt = build_system_prompt(has_task_tool=True)
        self.assertIn("Task management rules", prompt)
        self.assertIn("in_progress", prompt)

    def test_task_planning_absent_without_tool(self):
        prompt = build_system_prompt(has_task_tool=False)
        self.assertNotIn("# Task Planning", prompt)

    def test_current_tasks_rendered(self):
        tasks = [
            {"title": "Search prior art", "status": "completed"},
            {"title": "Draft summary", "status": "in_progress"},
            {"title": "Review claims", "status": "pending"},
        ]
        prompt = build_system_prompt(has_task_tool=True, tasks=tasks)
        self.assertIn("[x] Search prior art", prompt)
        self.assertIn("[~] Draft summary", prompt)
        self.assertIn("[ ] Review claims", prompt)

    def test_current_tasks_without_tool_flag(self):
        """Tasks render even without has_task_tool (legacy threads)."""
        tasks = [
            {"title": "Old task", "status": "completed"},
        ]
        prompt = build_system_prompt(has_task_tool=False, tasks=tasks)
        self.assertIn("[x] Old task", prompt)

    # ------------------------------------------------------------------ #
    # Sub-agent prompt section                                             #
    # ------------------------------------------------------------------ #

    def test_subagent_section_includes_limit(self):
        prompt = build_system_prompt(has_subagent_tool=True)
        self.assertIn("up to 4 sub-agents concurrently", prompt)

    def test_parallel_subagents_disabled_includes_sequential_instruction(self):
        prompt = build_system_prompt(
            has_subagent_tool=True, parallel_subagents=False
        )
        self.assertIn("Sequential sub-agents only", prompt)
        self.assertIn("one at a time", prompt)

    def test_parallel_subagents_enabled_no_sequential_instruction(self):
        prompt = build_system_prompt(
            has_subagent_tool=True, parallel_subagents=True
        )
        self.assertNotIn("Sequential sub-agents only", prompt)


# ====================================================================== #
# build_static_system_prompt tests                                        #
# ====================================================================== #

class BuildStaticSystemPromptTests(TestCase):
    """Verify build_static_system_prompt contains only truly stable content."""

    def test_contains_identity(self):
        prompt = build_static_system_prompt()
        self.assertIn(settings.ASSISTANT_NAME, prompt)
        self.assertIn("an AI assistant.", prompt)

    def test_contains_org_name(self):
        prompt = build_static_system_prompt(organization_name="MIT TTO")
        self.assertIn("at MIT TTO", prompt)

    def test_no_date(self):
        """Date is semi-static, should not be in the static block."""
        prompt = build_static_system_prompt()
        self.assertNotIn("Today's date", prompt)

    def test_no_data_rooms(self):
        """Data rooms are semi-static, should not be in the static block."""
        prompt = build_static_system_prompt()
        self.assertNotIn("Attached Data Rooms", prompt)

    def test_no_skill(self):
        """Skill is semi-static, should not be in the static block."""
        prompt = build_static_system_prompt()
        self.assertNotIn("Relevant skill", prompt)

    def test_no_canvas_metadata(self):
        """Canvas metadata is semi-static, should not be in the static block."""
        prompt = build_static_system_prompt()
        self.assertNotIn("Canvas workspace", prompt)
        self.assertNotIn("← active", prompt)

    def test_no_rag_results(self):
        """Static prompt must never contain RAG/document retrieval results."""
        prompt = build_static_system_prompt()
        self.assertNotIn("Retrieved Documents", prompt)
        self.assertNotIn("hybrid retrieval RAG", prompt)

    def test_no_task_status(self):
        """Static prompt must not contain task checklist status."""
        prompt = build_static_system_prompt(has_task_tool=True)
        self.assertIn("# Task Planning", prompt)
        self.assertNotIn("[x]", prompt)
        self.assertNotIn("[~]", prompt)
        self.assertNotIn("Current Task Plan", prompt)

    def test_no_subagent_status(self):
        """Static prompt must not contain sub-agent run status."""
        prompt = build_static_system_prompt(has_subagent_tool=True)
        self.assertIn("# Sub-agents", prompt)
        self.assertNotIn("# Sub-agent Status", prompt)

    def test_no_history_meta(self):
        """Static prompt must not contain history meta."""
        prompt = build_static_system_prompt()
        self.assertNotIn("messages total", prompt)

    def test_canvas_boilerplate_moved_to_skill(self):
        # Canvas usage boilerplate (## Diagrams etc.) moved into the
        # canvas_collaborator skill; only non-canvas boilerplate stays static.
        prompt = build_static_system_prompt()
        self.assertNotIn("## Diagrams", prompt)
        self.assertIn("## Emails", prompt)

    def test_task_boilerplate_included(self):
        prompt = build_static_system_prompt(has_task_tool=True)
        self.assertIn("When to create a task plan", prompt)
        self.assertIn("Task management rules", prompt)

    def test_subagent_boilerplate_included(self):
        prompt = build_static_system_prompt(has_subagent_tool=True)
        self.assertIn("chat_subagent_create", prompt)
        self.assertIn("up to 4 sub-agents", prompt)

    def test_sequential_subagents(self):
        prompt = build_static_system_prompt(
            has_subagent_tool=True, parallel_subagents=False
        )
        self.assertIn("Sequential sub-agents only", prompt)


# ====================================================================== #
# build_loop_turn_delimiter tests                                          #
# ====================================================================== #

class BuildLoopTurnDelimiterTests(TestCase):
    """The scheduled-Loop framing that replaces the '# User Message' boundary."""

    def test_frames_text_as_standing_loop_instruction(self):
        delim = build_loop_turn_delimiter()
        self.assertIn("Scheduled Loop Task", delim)
        # Ends with the header the actual loop prompt is appended under.
        last_line = delim.rstrip().splitlines()[-1]
        self.assertTrue(last_line.startswith("# Loop instructions"))

    def test_pushes_autonomous_completion_and_forbids_pushback(self):
        lower = build_loop_turn_delimiter().lower()
        self.assertIn("autonomously", lower)
        # The exact failure modes seen in production: asking what the real ask
        # is, restating to confirm, deferring to a later turn.
        self.assertIn("do not ask for clarification", lower)
        self.assertIn("no one is watching", lower)

    def test_framing_not_in_static_system_prompt(self):
        """The framing must live in the last user message, never the cached
        static system prompt (where it was too easily ignored)."""
        static = build_static_system_prompt()
        self.assertNotIn("Scheduled Loop Task", static)
        self.assertNotIn("Loop instructions", static)
        # The retired wording must not linger in the static prompt either.
        self.assertNotIn("Scheduled recurring turn", static)


# ====================================================================== #
# build_last_message_preamble tests                                        #
# ====================================================================== #

class BuildLastMessagePreambleTests(TestCase):
    """The preamble text prepended to the last user message each turn."""

    def test_empty_when_nothing_to_inject(self):
        self.assertEqual(build_last_message_preamble(), "")

    def test_combines_semi_static_and_dynamic_under_one_header(self):
        preamble = build_last_message_preamble(
            semi_static_system="SEMI", dynamic_context="DYN",
        )
        self.assertEqual(
            preamble, "# Additional Context\nSEMI\n\nDYN\n\n# User Message",
        )

    def test_semi_static_only(self):
        preamble = build_last_message_preamble(semi_static_system="SEMI")
        self.assertEqual(preamble, "# Additional Context\nSEMI\n\n# User Message")

    def test_dynamic_only(self):
        preamble = build_last_message_preamble(dynamic_context="DYN")
        self.assertEqual(preamble, "# Additional Context\nDYN\n\n# User Message")

    def test_loop_turn_replaces_user_message_delimiter_with_framing(self):
        preamble = build_last_message_preamble(
            dynamic_context="DYN", is_loop_turn=True,
        )
        self.assertIn("# Additional Context\nDYN", preamble)
        self.assertNotIn("# User Message", preamble)
        self.assertIn("Scheduled Loop Task", preamble)
        self.assertTrue(preamble.rstrip().endswith("from the user"))

    def test_loop_turn_framing_applied_even_without_context(self):
        """A loop turn always gets framed, even with no semi-static/dynamic."""
        preamble = build_last_message_preamble(is_loop_turn=True)
        self.assertNotEqual(preamble, "")
        self.assertNotIn("# Additional Context", preamble)
        self.assertEqual(preamble, build_loop_turn_delimiter())


# ====================================================================== #
# build_semi_static_prompt tests                                           #
# ====================================================================== #

class BuildSemiStaticPromptTests(TestCase):
    """Verify build_semi_static_prompt contains session-level content."""

    def test_contains_date(self):
        prompt = build_semi_static_prompt()
        self.assertIn("Today's date", prompt)

    def test_contains_data_room_names(self):
        rooms = [{"id": 1, "name": "Patents", "description": "Patent filings"}]
        prompt = build_semi_static_prompt(data_rooms=rooms)
        self.assertIn("Patents", prompt)
        self.assertIn("Patent filings", prompt)

    def test_no_data_rooms_message(self):
        prompt = build_semi_static_prompt()
        self.assertIn("No data rooms are attached", prompt)

    def test_skill_included(self):
        skill = MagicMock()
        skill.name = "Drafter"
        skill.description = ""
        skill.instructions = "Draft carefully."
        skill.templates.all.return_value = []
        prompt = build_semi_static_prompt(skills=[skill])
        self.assertIn("# Relevant skills", prompt)
        self.assertIn("Draft carefully.", prompt)

    def test_canvas_metadata_listed(self):
        canvases = [
            {"title": "Doc A", "chars": 100, "is_active": False},
            {"title": "Doc B", "chars": 200, "is_active": True},
        ]
        prompt = build_semi_static_prompt(canvases=canvases)
        self.assertIn("Doc A", prompt)
        self.assertIn("Doc B", prompt)
        self.assertIn("in context", prompt)

    def test_no_active_canvas_content(self):
        """Semi-static prompt must not contain active canvas content."""
        canvases = [{"title": "Draft", "chars": 500, "is_active": True}]
        prompt = build_semi_static_prompt(canvases=canvases)
        self.assertIn("Draft", prompt)
        self.assertIn("500 chars", prompt)
        self.assertNotIn("Active Canvas Content", prompt)

    def test_no_rag_results(self):
        """Semi-static prompt must not contain RAG results."""
        prompt = build_semi_static_prompt(
            data_rooms=[{"id": 1, "name": "Room"}],
        )
        self.assertNotIn("Retrieved Documents", prompt)
        self.assertNotIn("hybrid retrieval RAG", prompt)

    def test_no_task_status(self):
        """Semi-static prompt must not contain task status."""
        prompt = build_semi_static_prompt()
        self.assertNotIn("[x]", prompt)
        self.assertNotIn("Current Task Plan", prompt)

    def test_no_subagent_status(self):
        """Semi-static prompt must not contain sub-agent status."""
        prompt = build_semi_static_prompt()
        self.assertNotIn("Sub-agent Status", prompt)

    def test_web_content_safety_moved_to_skill_with_data_rooms(self):
        """Web content safety moved into the web_research_tools skill — it is no
        longer in the always-on prompt, even when data rooms are attached."""
        rooms = [{"id": 1, "name": "Patents", "description": "Patent filings"}]
        prompt = build_semi_static_prompt(data_rooms=rooms)
        self.assertNotIn("# Web Content Safety", prompt)

    def test_web_content_safety_moved_to_skill_without_data_rooms(self):
        """Same, with no data rooms attached."""
        prompt = build_semi_static_prompt()
        self.assertNotIn("# Web Content Safety", prompt)


# ====================================================================== #
# build_dynamic_context tests                                              #
# ====================================================================== #

class BuildDynamicContextTests(TestCase):
    """Verify build_dynamic_context produces correct per-turn content."""

    def test_always_contains_current_time(self):
        result = build_dynamic_context()
        self.assertIn("# Current time", result)
        self.assertIn("<context>", result)

    def test_current_time_with_all_none(self):
        result = build_dynamic_context(
            doc_context=None, active_canvas=None, tasks=None,
            subagent_runs=None, history_meta=None,
        )
        self.assertIn("# Current time", result)

    def test_doc_context_included(self):
        doc_context = {
            "total_doc_count": 2,
            "documents": [{
                "doc_index": 1,
                "filename": "Patent.pdf",
                "description": "A patent",
                "token_count": 1000,
                "document_type": "Patent",
            }],
        }
        result = build_dynamic_context(doc_context=doc_context)
        self.assertIn("<context>", result)
        self.assertIn("</context>", result)
        self.assertIn("Retrieved Documents", result)
        self.assertIn("Patent.pdf", result)
        self.assertIn("(Patent)", result)

    def test_doc_context_includes_dates(self):
        doc_context = {
            "total_doc_count": 1,
            "documents": [{
                "doc_index": 1,
                "filename": "Contract.pdf",
                "description": "A contract",
                "token_count": 500,
                "document_type": "Agreement",
                "uploaded_at": "2025-03-15",
                "file_metadata_date": "2024-11-02",
                "document_date": "2024-10-28",
            }],
        }
        result = build_dynamic_context(doc_context=doc_context)
        self.assertIn("uploaded to data room: 2025-03-15", result)
        self.assertIn("file date: 2024-11-02", result)
        self.assertIn("document date: 2024-10-28", result)

    def test_doc_context_omits_missing_dates(self):
        doc_context = {
            "total_doc_count": 1,
            "documents": [{
                "doc_index": 1,
                "filename": "Notes.txt",
                "description": "",
                "token_count": 100,
                "document_type": "",
                "uploaded_at": "2025-01-01",
                "file_metadata_date": None,
                "document_date": None,
            }],
        }
        result = build_dynamic_context(doc_context=doc_context)
        self.assertIn("uploaded to data room: 2025-01-01", result)
        self.assertNotIn("file date:", result)
        self.assertNotIn("document date:", result)

    def test_no_docs_with_data_rooms(self):
        result = build_dynamic_context(
            data_rooms=[{"id": 1, "name": "Room"}],
        )
        self.assertIn("no documents uploaded yet", result)

    def test_active_canvas_content(self):
        class FakeCanvas:
            title = "My Draft"
            content = "Hello world"

        result = build_dynamic_context(active_canvases=[FakeCanvas()])
        self.assertIn("Active Canvas Content", result)
        self.assertIn("My Draft", result)
        self.assertIn("Hello world", result)

    def test_multiple_active_canvases(self):
        class FakeCanvas:
            def __init__(self, title, content):
                self.title = title
                self.content = content

        result = build_dynamic_context(active_canvases=[
            FakeCanvas("Draft A", "Content A"),
            FakeCanvas("Draft B", "Content B"),
        ])
        self.assertIn("Draft A", result)
        self.assertIn("Content A", result)
        self.assertIn("Draft B", result)
        self.assertIn("Content B", result)

    def test_single_canvas_content(self):
        """Old single-canvas API via 'canvas' param."""
        class FakeCanvas:
            title = "Old"
            content = "old stuff"

        result = build_dynamic_context(canvas=FakeCanvas())
        self.assertIn("Active Canvas Content", result)
        self.assertIn("old stuff", result)

    def test_tasks_rendered(self):
        tasks = [
            {"title": "Search prior art", "status": "completed"},
            {"title": "Draft summary", "status": "in_progress"},
            {"title": "Review claims", "status": "pending"},
        ]
        result = build_dynamic_context(tasks=tasks)
        self.assertIn("[x] Search prior art", result)
        self.assertIn("[~] Draft summary", result)
        self.assertIn("[ ] Review claims", result)

    def test_subagent_runs_rendered(self):
        runs = [{
            "id": uuid.uuid4(), "status": "completed",
            "prompt": "Research patents", "model_tier": "mid",
            "result": "Found 3 patents.", "error": "",
        }]
        result = build_dynamic_context(subagent_runs=runs)
        self.assertIn("Sub-agent Status", result)
        self.assertIn("COMPLETED", result)
        self.assertIn("delivered as message", result.lower())
        self.assertNotIn("Found 3 patents.", result)

    def test_completed_result_not_in_dynamic_context(self):
        """Completed sub-agent results are persisted as messages, not injected in context."""
        runs = [{
            "id": uuid.uuid4(), "status": "completed",
            "prompt": "Research topic", "model_tier": "mid",
            "result": "Some findings from the web.", "error": "",
        }]
        result = build_dynamic_context(subagent_runs=runs)
        self.assertNotIn("Some findings from the web.", result)
        self.assertNotIn("<subagent_result>", result)
        self.assertIn("delivered as message", result.lower())

    def test_completed_empty_result_shows_failure(self):
        """Completed sub-agent with empty result tells orchestrator not to fabricate."""
        runs = [{
            "id": uuid.uuid4(), "status": "completed",
            "prompt": "Research topic", "model_tier": "mid",
            "result": "", "error": "",
        }]
        result = build_dynamic_context(subagent_runs=runs)
        self.assertIn("COMPLETED", result)
        self.assertNotIn("delivered as message", result.lower())
        self.assertIn("no usable result", result.lower())
        self.assertIn("do NOT fabricate", result)

    def test_history_meta_rendered(self):
        meta = {
            "total_messages": 100,
            "included_messages": 30,
            "has_summary": True,
        }
        result = build_dynamic_context(history_meta=meta)
        self.assertIn("100 messages total", result)
        self.assertIn("30 most recent", result)
        self.assertIn("summary of earlier messages", result.lower())

    def test_history_meta_all_included_no_truncation_note(self):
        meta = {
            "total_messages": 5,
            "included_messages": 5,
            "has_summary": False,
        }
        result = build_dynamic_context(history_meta=meta)
        self.assertNotIn("messages total", result)

    def test_context_tags_wrap_output(self):
        tasks = [{"title": "Do thing", "status": "pending"}]
        result = build_dynamic_context(tasks=tasks)
        self.assertTrue(result.startswith("<context>"))
        self.assertTrue(result.endswith("</context>"))

    def test_multiple_sections_combined(self):
        doc_context = {
            "total_doc_count": 1,
            "documents": [{
                "doc_index": 1, "filename": "a.pdf",
                "description": "", "token_count": 100,
                "document_type": "",
            }],
        }
        tasks = [{"title": "Task 1", "status": "pending"}]
        meta = {"total_messages": 50, "included_messages": 20, "has_summary": False}
        result = build_dynamic_context(
            doc_context=doc_context, tasks=tasks, history_meta=meta,
        )
        self.assertIn("Retrieved Documents", result)
        self.assertIn("Current Task Plan", result)
        self.assertIn("50 messages total", result)


# ====================================================================== #
# Wrapper regression test                                                  #
# ====================================================================== #

class BuildSystemPromptRegressionTests(TestCase):
    """Verify the wrapper build_system_prompt still produces expected output."""

    def test_wrapper_includes_static_and_dynamic(self):
        """The wrapper should contain both static and dynamic content."""
        doc_context = {
            "total_doc_count": 1,
            "documents": [{
                "doc_index": 1, "filename": "test.pdf",
                "description": "desc", "token_count": 100,
                "document_type": "Report",
            }],
        }
        tasks = [{"title": "Analyze", "status": "in_progress"}]
        prompt = build_system_prompt(
            data_rooms=[{"id": 1, "name": "Room"}],
            doc_context=doc_context,
            tasks=tasks,
            has_task_tool=True,
        )
        # Static content
        self.assertIn(settings.ASSISTANT_NAME, prompt)
        self.assertIn("# Task Planning", prompt)
        self.assertIn("Room", prompt)
        # Dynamic content
        self.assertIn("test.pdf", prompt)
        self.assertIn("[~] Analyze", prompt)

    def test_wrapper_always_has_current_time(self):
        """build_system_prompt always injects current time into context."""
        prompt = build_system_prompt()
        self.assertIn(settings.ASSISTANT_NAME, prompt)
        self.assertIn("<context>", prompt)
        self.assertIn("# Current time", prompt)

    def test_static_is_stable_across_calls(self):
        """Static portion should be identical across calls."""
        static1 = build_static_system_prompt(has_task_tool=True)
        static2 = build_static_system_prompt(has_task_tool=True)
        self.assertEqual(static1, static2)

    def test_semi_static_is_stable_with_same_inputs(self):
        """Semi-static portion should be identical with same inputs."""
        rooms = [{"id": 1, "name": "Room"}]
        ss1 = build_semi_static_prompt(data_rooms=rooms)
        ss2 = build_semi_static_prompt(data_rooms=rooms)
        self.assertEqual(ss1, ss2)


# ====================================================================== #
# Dynamic context injection into messages                                  #
# ====================================================================== #

class PreambleInjectionMechanicsTests(TestCase):
    """The message-list splice in ChatConsumer (last user message, multimodal).

    Exercises the real ``_prepend_preamble_to_last_user_message`` helper rather
    than a replica, so the splice mechanics can't drift from production.
    """

    def _make_messages(self, roles_and_contents):
        """Create a list of Message objects from (role, content) tuples."""
        from llm.types import Message
        return [Message(role=r, content=c) for r, c in roles_and_contents]

    def _inject(self, messages, preamble):
        from chat.consumers import _prepend_preamble_to_last_user_message
        _prepend_preamble_to_last_user_message(messages, preamble)
        return messages

    def test_injects_into_last_user_message(self):
        messages = self._make_messages([
            ("system", "You are Wilfred."),
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "second question"),
        ])
        result = self._inject(messages, "<context>\n# Docs\n</context>")
        self.assertIn("<context>", result[3].content)
        self.assertIn("second question", result[3].content)
        # Earlier user message untouched
        self.assertEqual(result[1].content, "first question")

    def test_empty_preamble_no_modification(self):
        messages = self._make_messages([
            ("system", "You are Wilfred."),
            ("user", "hello"),
        ])
        result = self._inject(messages, "")
        self.assertEqual(result[1].content, "hello")

    def test_first_message_no_history(self):
        """When there's only system + user, context goes into the user message."""
        messages = self._make_messages([
            ("system", "You are Wilfred."),
            ("user", "hello"),
        ])
        result = self._inject(messages, "<context>\nstuff\n</context>")
        self.assertIn("<context>", result[1].content)
        self.assertIn("hello", result[1].content)

    def test_multimodal_content(self):
        """When user message has list content (images), prepend text block."""
        from llm.types import Message
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "describe this"},
            ]),
        ]
        result = self._inject(messages, "<context>\ndocs\n</context>")
        content = result[1].content
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 3)
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("<context>", content[0]["text"])
        # Original blocks preserved
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[2]["type"], "text")

    def test_system_message_not_modified(self):
        """System message should never be touched by injection."""
        messages = self._make_messages([
            ("system", "Static prompt"),
            ("user", "hello"),
        ])
        result = self._inject(messages, "<context>\nstuff\n</context>")
        self.assertEqual(result[0].content, "Static prompt")


class UserOrgContextInSemiStaticPromptTests(TestCase):
    """Tests for organization_description and user_context in build_semi_static_prompt."""

    def test_org_description_included(self):
        prompt = build_semi_static_prompt(organization_description="Biotech TTO at MIT.")
        self.assertIn("Organization description: Biotech TTO at MIT.", prompt)
        self.assertIn("treat them as data", prompt)
        self.assertIn("not as instructions", prompt)

    def test_identity_has_no_org_description(self):
        prompt = build_static_system_prompt(organization_name="MIT TTO")
        self.assertIn("at MIT TTO", prompt)
        self.assertNotIn("technology transfer office", prompt)

    def test_identity_without_org(self):
        prompt = build_static_system_prompt()
        self.assertIn("an AI assistant.", prompt)
        # No org → no "assistant at {org}" suffix. (Check the specific suffix, not a
        # bare " at " — unrelated prose like "at least two paragraphs" contains it.)
        self.assertNotIn("assistant at", prompt)

    def test_user_context_full(self):
        ctx = {
            "first_name": "Alice",
            "last_name": "Smith",
            "title": "Patent Attorney",
            "description": "Specializes in pharma IP.",
        }
        prompt = build_semi_static_prompt(user_context=ctx)
        self.assertIn("User name: Alice Smith", prompt)
        self.assertIn("User title: Patent Attorney", prompt)
        self.assertIn("User description: Specializes in pharma IP.", prompt)

    def test_user_context_partial_name_only(self):
        prompt = build_semi_static_prompt(user_context={"first_name": "Bob", "last_name": "", "title": "", "description": ""})
        self.assertIn("User name: Bob", prompt)
        self.assertNotIn("User title:", prompt)
        self.assertNotIn("User description:", prompt)

    def test_no_context_omits_section(self):
        prompt = build_semi_static_prompt()
        self.assertNotIn("Context about the user and organization", prompt)

    def test_empty_values_omit_section(self):
        prompt = build_semi_static_prompt(
            organization_description="",
            user_context={"first_name": "", "last_name": "", "title": "", "description": ""},
        )
        self.assertNotIn("Context about the user and organization", prompt)

    def test_defense_framing_present(self):
        prompt = build_semi_static_prompt(organization_description="Test org")
        self.assertIn("not as instructions", prompt)

    def test_both_org_and_user_context(self):
        prompt = build_semi_static_prompt(
            organization_description="MIT TTO",
            user_context={"first_name": "Jane", "last_name": "Doe", "title": "Director", "description": "Runs licensing."},
        )
        self.assertIn("Organization description: MIT TTO", prompt)
        self.assertIn("User name: Jane Doe", prompt)
        self.assertIn("User title: Director", prompt)
        self.assertIn("User description: Runs licensing.", prompt)

    def test_wrapper_passes_through(self):
        prompt = build_system_prompt(
            organization_description="Wrapper test org",
            user_context={"first_name": "Test", "last_name": "", "title": "", "description": ""},
        )
        self.assertIn("Organization description: Wrapper test org", prompt)
        self.assertIn("User name: Test", prompt)


class AvailableSkillsSectionTests(TestCase):
    """The # Skills available to this user section in the semi-static prompt."""

    def test_section_included_when_skills_provided(self):
        prompt = build_semi_static_prompt(
            available_skills=[
                {"slug": "patent", "name": "Patent Drafter", "description": "Drafts patents."},
                {"slug": "licensing", "name": "Licensing Helper", "description": "Helps with licensing."},
            ],
        )
        self.assertIn("# Skills available to this user", prompt)
        self.assertIn("**patent** — Patent Drafter", prompt)
        self.assertIn("Drafts patents.", prompt)
        self.assertIn("**licensing** — Licensing Helper", prompt)
        self.assertIn("chat_skill_attach", prompt)

    def test_section_omitted_when_available_skills_none(self):
        prompt = build_semi_static_prompt(available_skills=None)
        self.assertNotIn("# Skills available to this user", prompt)

    def test_section_omitted_when_available_skills_empty(self):
        prompt = build_semi_static_prompt(available_skills=[])
        self.assertNotIn("# Skills available to this user", prompt)

    def test_description_truncated_at_160_chars(self):
        long_desc = "x" * 400
        prompt = build_semi_static_prompt(
            available_skills=[
                {"slug": "big", "name": "Big", "description": long_desc},
            ],
        )
        self.assertIn("**big** — Big", prompt)
        self.assertIn("x" * 100, prompt)  # It's there
        self.assertIn("...", prompt)  # truncation marker
        # The bullet line itself should be at most ~ slug+name+160 chars + markup
        bullet_line = next(
            line for line in prompt.splitlines() if line.startswith("- **big**")
        )
        self.assertLess(len(bullet_line), 250)

    def test_description_optional(self):
        prompt = build_semi_static_prompt(
            available_skills=[
                {"slug": "bare", "name": "Bare", "description": ""},
            ],
        )
        self.assertIn("**bare** — Bare", prompt)

    def test_build_system_prompt_forwards_available_skills(self):
        prompt = build_system_prompt(
            available_skills=[
                {"slug": "via-wrapper", "name": "Wrapper", "description": "w"},
            ],
        )
        self.assertIn("**via-wrapper** — Wrapper", prompt)
        self.assertIn("# Skills available to this user", prompt)

    def test_newlines_in_description_collapsed(self):
        prompt = build_semi_static_prompt(
            available_skills=[
                {"slug": "multi", "name": "Multi", "description": "line one\nline two"},
            ],
        )
        # Description rendered as one line (no embedded \n in the bullet)
        bullet_line = next(
            line for line in prompt.splitlines() if line.startswith("- **multi**")
        )
        self.assertIn("line one line two", bullet_line)

    def test_emoji_prefixed_when_provided(self):
        prompt = build_semi_static_prompt(
            available_skills=[
                {"slug": "lab", "name": "Lab Notes", "emoji": "🧪", "description": ""},
            ],
        )
        self.assertIn("- 🧪 **lab** — Lab Notes", prompt)

    def test_emoji_absent_when_empty(self):
        prompt = build_semi_static_prompt(
            available_skills=[
                {"slug": "plain", "name": "Plain", "emoji": "", "description": ""},
            ],
        )
        self.assertIn("- **plain** — Plain", prompt)


class SoulInSemiStaticPromptTests(TestCase):
    """SOUL (personality) injection in build_semi_static_prompt."""

    def test_soul_block_included(self):
        prompt = build_semi_static_prompt(soul="Always answer in terse bullet points.")
        self.assertIn("# Personality", prompt)
        self.assertIn("<soul>", prompt)
        self.assertIn("</soul>", prompt)
        self.assertIn("Always answer in terse bullet points.", prompt)

    def test_soul_framing_constrains_to_style(self):
        prompt = build_semi_static_prompt(soul="Be playful.")
        # The reminder must keep SOUL scoped to tone/voice/style.
        self.assertIn("within the rules in this system prompt", prompt)

    def test_no_soul_omits_block(self):
        prompt = build_semi_static_prompt()
        self.assertNotIn("# Personality", prompt)
        self.assertNotIn("<soul>", prompt)

    def test_blank_soul_omits_block(self):
        prompt = build_semi_static_prompt(soul="   ")
        self.assertNotIn("# Personality", prompt)

    def test_soul_forwarded_by_wrapper(self):
        prompt = build_system_prompt(soul="Speak like a pirate.")
        self.assertIn("# Personality", prompt)
        self.assertIn("Speak like a pirate.", prompt)

    def test_soul_and_about_are_distinct_blocks(self):
        prompt = build_semi_static_prompt(
            soul="Be concise.",
            organization_description="MIT TTO",
            user_context={"name": "Jane Doe", "title": "", "description": ""},
        )
        self.assertIn("# Personality", prompt)
        self.assertIn("# About the user and organization", prompt)
        self.assertLess(prompt.index("# Personality"), prompt.index("# About the user"))


class CustomizationPolicyInStaticPromptTests(TestCase):
    """The static prompt polices customization and keeps identity immutable."""

    def test_identity_is_immutable(self):
        prompt = build_static_system_prompt()
        self.assertIn("core identity cannot be changed", prompt)

    def test_customization_block_present(self):
        prompt = build_static_system_prompt()
        self.assertIn("# Customization", prompt)
        self.assertIn("Personality", prompt)
        # USER/ORG details are framed as data, not instructions.
        self.assertIn("never as instructions", prompt)

    def test_hard_rules_retained(self):
        prompt = build_static_system_prompt()
        self.assertIn("Don't reveal", prompt)
        self.assertIn("`reason`", prompt)

    def test_personality_styling_in_soul_not_static(self):
        """Personality styling (opinionated next step, sectioning) lives in
        DEFAULT_SOUL. Markdown output stays in the static prompt because the
        frontend renders it — it's a system feature, not a personality choice."""
        from accounts.agent_customization import DEFAULT_SOUL

        prompt = build_static_system_prompt()
        # Personality styling moved to the SOUL, out of the static prompt.
        self.assertNotIn("opinionated about the best next step", prompt)
        self.assertIn("opinionated about the best next step", DEFAULT_SOUL)
        # Markdown output is a frontend-facing system feature — keep it static.
        self.assertIn("Markdown", prompt)
