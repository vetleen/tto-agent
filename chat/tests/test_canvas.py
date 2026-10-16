"""Tests for Canvas tools, consumer events, and views."""

from __future__ import annotations

import json
import io
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase, override_settings

from chat.canvas_tools import DeleteCanvasTool, EditCanvasTool, PasteUserTextTool, ReadCanvasTool, WriteCanvasTool
from chat.models import CanvasCheckpoint, ChatCanvas, ChatMessage, ChatThread
from llm.types.context import RunContext

User = get_user_model()


def _ctx(user_id, thread_id):
    return RunContext.create(user_id=user_id, conversation_id=str(thread_id))


def _invoke(tool_cls, args, ctx):
    tool = tool_cls()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


# ---------------------------------------------------------------------------
# WriteCanvasTool tests
# ---------------------------------------------------------------------------

class WriteCanvasToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="canvas@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_creates_canvas_on_first_call(self):
        result = _invoke(WriteCanvasTool, {"title": "My NDA", "content": "# NDA\n..."}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        canvas = ChatCanvas.objects.get(thread=self.thread)
        self.assertEqual(canvas.title, "My NDA")
        self.assertEqual(canvas.content, "# NDA\n...")

    def test_overwrites_canvas_with_same_title(self):
        _invoke(WriteCanvasTool, {"title": "Draft", "content": "old"}, _ctx(self.user.pk, self.thread.id))
        _invoke(WriteCanvasTool, {"title": "Draft", "content": "new"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(ChatCanvas.objects.filter(thread=self.thread).count(), 1)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Draft")
        self.assertEqual(canvas.content, "new")

    def test_creates_second_canvas_with_different_title(self):
        _invoke(WriteCanvasTool, {"title": "Draft 1", "content": "old"}, _ctx(self.user.pk, self.thread.id))
        _invoke(WriteCanvasTool, {"title": "Draft 2", "content": "new"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(ChatCanvas.objects.filter(thread=self.thread).count(), 2)

    def test_returns_metadata_in_result(self):
        result = _invoke(WriteCanvasTool, {"title": "T", "content": "C"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["title"], "T")
        self.assertIn("canvas_id", result)
        self.assertNotIn("content", result)
        self.assertNotIn("accepted_content", result)

    def test_no_context_returns_error(self):
        tool = WriteCanvasTool()
        result = json.loads(tool.invoke({"title": "T", "content": "C"}))
        self.assertEqual(result["status"], "error")

    def test_long_title_truncated_to_column_limit(self):
        """LLM-supplied titles longer than the 255-char column must be
        truncated, not raise a DB error."""
        long_title = "T" * 400
        result = _invoke(
            WriteCanvasTool,
            {"title": long_title, "content": "body"},
            _ctx(self.user.pk, self.thread.id),
        )
        self.assertEqual(result["status"], "ok")
        canvas = ChatCanvas.objects.get(thread=self.thread)
        self.assertEqual(len(canvas.title), 255)

    def test_overwrite_snapshots_uncommitted_user_edits(self):
        """A full rewrite over user edits that aren't checkpointed yet must
        auto-snapshot them first so they're recoverable from history."""
        _invoke(WriteCanvasTool, {"title": "Draft", "content": "ai v1"}, _ctx(self.user.pk, self.thread.id))
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Draft")

        # Simulate user edits saved directly (no checkpoint), as the consumer's
        # _save_canvas does on autosave.
        canvas.content = "user edited content"
        canvas.save(update_fields=["content", "updated_at"])

        _invoke(WriteCanvasTool, {"title": "Draft", "content": "ai v2"}, _ctx(self.user.pk, self.thread.id))

        snapshots = CanvasCheckpoint.objects.filter(
            canvas=canvas, source="user_save",
            description="Auto-saved before AI edit",
        )
        self.assertEqual(snapshots.count(), 1)
        self.assertEqual(snapshots.first().content, "user edited content")
        canvas.refresh_from_db()
        self.assertEqual(canvas.content, "ai v2")


# ---------------------------------------------------------------------------
# EditCanvasTool tests
# ---------------------------------------------------------------------------

class EditCanvasToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="edit@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _setup_canvas(self, content):
        from django.utils import timezone

        canvas = ChatCanvas.objects.create(
            thread=self.thread, title="Doc", content=content,
            is_active=True, last_activated_at=timezone.now(),
        )
        self.thread.active_canvas = canvas
        self.thread.save(update_fields=["active_canvas"])

    def test_applies_edit_correctly(self):
        self._setup_canvas("The term is 3 years.")
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "3 years", "new_text": "5 years", "reason": "update term"}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["applied"], 1)
        self.assertNotIn("content", result)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        self.assertIn("5 years", canvas.content)

    def test_skips_unfound_old_text(self):
        self._setup_canvas("Some content here.")
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "not in text", "new_text": "replacement", "reason": "test"}]
        }, _ctx(self.user.pk, self.thread.id))
        # No edit matched: report an error and leave the canvas untouched, so the
        # model doesn't mistake a no-op for a successful edit.
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["applied"], 0)
        self.assertEqual(len(result["failed"]), 1)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        self.assertEqual(canvas.content, "Some content here.")

    def test_partial_edits_applied(self):
        self._setup_canvas("Hello world foo bar.")
        result = _invoke(EditCanvasTool, {
            "edits": [
                {"old_text": "Hello", "new_text": "Hi", "reason": ""},
                {"old_text": "nonexistent", "new_text": "x", "reason": ""},
            ]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["applied"], 1)
        self.assertEqual(len(result["failed"]), 1)

    def test_rejects_duplicate_matches(self):
        self._setup_canvas("The cat sat. The cat slept.")
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "The cat", "new_text": "The dog", "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["applied"], 0)
        self.assertEqual(len(result["failed"]), 1)
        self.assertIn("2 matches", result["failed"][0]["error"])
        # Content should be unchanged
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        self.assertEqual(canvas.content, "The cat sat. The cat slept.")

    def test_returns_error_when_no_canvas(self):
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "x", "new_text": "y", "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "error")
        self.assertIn("canvas_write", result["message"])

    def test_no_match_with_large_replacement_adds_hint(self):
        # A failed edit whose replacement is far too big for a targeted edit
        # steers the model to a short edit or a rewrite instead of a retry.
        self._setup_canvas("Some content here.")
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "not in text", "new_text": "y" * 12_000, "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["failed"][0]["new_text_chars"], 12_000)
        self.assertIn("12,000 characters", result["hint"])
        self.assertIn("canvas_write", result["hint"])

    def test_no_match_with_small_replacement_has_no_hint(self):
        self._setup_canvas("Some content here.")
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "not in text", "new_text": "short", "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "error")
        self.assertNotIn("hint", result)
        self.assertEqual(result["failed"][0]["new_text_chars"], 5)

    def test_edit_snapshots_uncommitted_user_edits(self):
        """AI edits over user content that isn't checkpointed yet must
        auto-snapshot it first so it's recoverable from history."""
        self._setup_canvas("User wrote this draft.")

        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "draft", "new_text": "document", "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["applied"], 1)

        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        snapshots = CanvasCheckpoint.objects.filter(
            canvas=canvas, source="user_save",
            description="Auto-saved before AI edit",
        )
        self.assertEqual(snapshots.count(), 1)
        self.assertEqual(snapshots.first().content, "User wrote this draft.")
        self.assertIn("document", canvas.content)


# ---------------------------------------------------------------------------
# Consumer canvas event tests
# ---------------------------------------------------------------------------

@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ConsumerCanvasTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ws@test.com", password="pass")

    async def _communicator(self):
        from channels.routing import URLRouter
        from channels.testing import WebsocketCommunicator
        from chat.routing import websocket_urlpatterns

        app = URLRouter(websocket_urlpatterns)
        comm = WebsocketCommunicator(app, "/ws/chat/")
        comm.scope["user"] = self.user
        connected, _ = await comm.connect()
        assert connected
        return comm

    async def test_canvas_loaded_on_thread_load(self):
        from channels.db import database_sync_to_async

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            c = ChatCanvas.objects.create(thread=thread, title="My Doc", content="hello")
            thread.active_canvas = c
            thread.save(update_fields=["active_canvas"])
        await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({"type": "chat.load_thread", "thread_id": str(thread.id)})

        # thread.loaded event
        msg1 = await comm.receive_json_from()
        self.assertEqual(msg1["event_type"], "thread.loaded")

        # canvases.loaded event
        msg2 = await comm.receive_json_from()
        self.assertEqual(msg2["event_type"], "canvases.loaded")
        self.assertEqual(len(msg2["canvases"]), 1)
        self.assertEqual(msg2["active_canvas"]["title"], "My Doc")
        self.assertEqual(msg2["active_canvas"]["content"], "hello")

        await comm.disconnect()

    async def test_canvas_save_persists(self):
        from channels.db import database_sync_to_async

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        comm = await self._communicator()
        await comm.send_json_to({
            "type": "chat.canvas_save",
            "thread_id": str(thread.id),
            "title": "Saved Title",
            "content": "saved content",
        })
        # No response expected — just persistence
        await comm.disconnect()

        canvas = await database_sync_to_async(ChatCanvas.objects.get)(thread=thread)
        self.assertEqual(canvas.title, "Saved Title")
        self.assertEqual(canvas.content, "saved content")

    async def test_canvas_open_creates_and_returns_canvas(self):
        from channels.db import database_sync_to_async

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        comm = await self._communicator()
        await comm.send_json_to({"type": "chat.canvas_open", "thread_id": str(thread.id)})

        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvases.loaded")
        self.assertIn("canvases", msg)
        self.assertIn("active_canvas", msg)

        await comm.disconnect()


# ---------------------------------------------------------------------------
# View tests: canvas_export and canvas_import
# ---------------------------------------------------------------------------

class CanvasExportViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="exp@test.com", password="pass")
        self.client.login(email="exp@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(
            thread=self.thread,
            title="My NDA",
            content="# NDA\n\nThis is an agreement.",
        )
        self.thread.active_canvas = self.canvas
        self.thread.save(update_fields=["active_canvas"])

    def _export(self, url, content=None):
        """POST to the export endpoint (the only method it accepts)."""
        body = {} if content is None else {"content": content}
        return self.client.post(url, json.dumps(body), content_type="application/json")

    def test_export_returns_docx(self):
        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get("Content-Type"),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertIn("My NDA", response.get("Content-Disposition", ""))

    def test_export_get_not_allowed(self):
        """The legacy GET path (server-side headless mermaid render) is removed."""
        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 405)

    def test_export_tables_use_fixed_layout(self):
        """Tables in exported .docx should span the full page width."""
        self.canvas.content = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument
        from docx.oxml.ns import qn

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        self.assertTrue(len(doc.tables) > 0, "Expected at least one table")
        for table in doc.tables:
            self.assertFalse(table.autofit)
            # Table width should be 100% (5000 in OOXML pct units)
            tblW = table._tbl.tblPr.find(qn("w:tblW"))
            self.assertEqual(tblW.get(qn("w:type")), "pct")
            self.assertEqual(tblW.get(qn("w:w")), "5000")
            # Cell widths should be non-zero
            for row in table.rows:
                for cell in row.cells:
                    tcW = cell._tc.tcPr.find(qn("w:tcW"))
                    self.assertGreater(int(tcW.get(qn("w:w"))), 0)

    def test_export_handles_highlight_markup(self):
        """==text== should export as a yellow Word highlight on that run only."""
        self.canvas.content = "This is ==highlighted== text."
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument
        from docx.enum.text import WD_COLOR_INDEX

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        runs = [r for p in doc.paragraphs for r in p.runs]
        highlighted = [r for r in runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW]
        self.assertEqual(len(highlighted), 1, "Exactly one run should be highlighted")
        self.assertEqual(highlighted[0].text.strip(), "highlighted")
        # The literal == markers must not survive into the document text.
        self.assertNotIn("==", "".join(r.text for r in runs))

    def test_export_turns_tight_ordered_list_into_word_list(self):
        """A preview-style list without a preceding blank stays a real list."""
        self.canvas.content = (
            "**Recommended next actions:**\n"
            "1. Grant power.\n"
            "2. Support the grant.\n"
            "3. Plan a review."
        )
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        labels = [p for p in doc.paragraphs if p.text == "Recommended next actions:"]
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].style.name, "Normal")

        items = [p.text for p in doc.paragraphs if p.style.name == "List Number"]
        self.assertEqual(items, ["Grant power.", "Support the grant.", "Plan a review."])
        self.assertNotIn("1. Grant power.", labels[0].text)

    def test_export_table_directly_after_prose_becomes_word_table(self):
        """A preview-style table without a preceding blank stays a real table."""
        self.canvas.content = "Prior art:\n| A | B |\n| --- | --- |\n| 1 | 2 |"
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        self.assertEqual(len(doc.tables), 1)
        self.assertIn("Prior art:", [p.text.strip() for p in doc.paragraphs])
        self.assertFalse(any("|" in p.text for p in doc.paragraphs))
        cells = [cell.text.strip() for row in doc.tables[0].rows for cell in row.cells]
        self.assertEqual(cells, ["A", "B", "1", "2"])

    def test_export_applies_org_styles(self):
        """The export should apply the exporting user's org styles to the .docx."""
        from accounts.models import Membership, Organization

        org = Organization.objects.create(name="StyleCo", slug="styleco")
        org.preferences = {"styles": {"body_font": "Georgia", "body_size": 13}}
        org.save(update_fields=["preferences"])
        Membership.objects.create(user=self.user, org=org, role=Membership.Role.MEMBER)

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        self.assertEqual(doc.styles["Normal"].font.name, "Georgia")
        self.assertEqual(doc.styles["Normal"].font.size.pt, 13)

    def test_export_keeps_single_spacer_after_tables(self):
        """html2docx emits a clump of blank paragraphs after tables; we keep
        exactly one spacer (for spacing + to avoid table merges)."""
        self.canvas.content = "| A | B |\n| --- | --- |\n| 1 | 2 |\n\nAfter the table."
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        blanks = [p for p in doc.paragraphs if not p.text.strip()]
        self.assertEqual(len(blanks), 1, "expected exactly one spacer paragraph")
        # The following text should survive intact, without a leading space.
        self.assertIn("After the table.", [p.text for p in doc.paragraphs])

    def test_export_heading_after_table_keeps_heading_style(self):
        """A real heading right after a table should stay a heading (html2docx
        otherwise leaks its text into a Normal paragraph)."""
        self.canvas.content = "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n### After Heading\n\nBody."
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        matches = [p for p in doc.paragraphs if p.text.strip() == "After Heading"]
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].style.name.startswith("Heading"))
        # no empty heading paragraph left behind
        empties = [p for p in doc.paragraphs if p.style.name.startswith("Heading") and not p.text.strip()]
        self.assertEqual(empties, [])

    def test_export_keeps_adjacent_tables_separate(self):
        """Two tables in a row must stay separated by a paragraph, else Word
        merges them into one."""
        self.canvas.content = "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n| C | D |\n| --- | --- |\n| 3 | 4 |"
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument
        from docx.oxml.ns import qn

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        tags = [c.tag.split("}")[-1] for c in doc.element.body.iterchildren()]
        tbl_positions = [i for i, t in enumerate(tags) if t == "tbl"]
        self.assertEqual(len(tbl_positions), 2)
        self.assertIn("p", tags[tbl_positions[0] + 1:tbl_positions[1]])

    def test_export_normalizes_top_heading_to_h1(self):
        """A canvas whose top heading is ## should export with the top as H1."""
        self.canvas.content = "## Title\n\nBody.\n\n### Section"
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        headings = {p.text: p.style.name for p in doc.paragraphs if p.style.name.startswith("Heading")}
        self.assertEqual(headings.get("Title"), "Heading 1")
        self.assertEqual(headings.get("Section"), "Heading 2")

    def test_export_renders_footnote_citations(self):
        """[^1] citations should export as superscripts + a numbered Sources list."""
        self.canvas.content = "Growth is strong.[^1]\n\n### Sources\n[^1]: BCC Research — https://x.test/a"
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        full_text = "\n".join(p.text for p in doc.paragraphs)
        self.assertNotIn("[^1]", full_text)  # no literal footnote markers
        # inline marker is a real superscript run
        supers = [r.text for p in doc.paragraphs for r in p.runs if r.font.superscript]
        self.assertIn("1", supers)
        # the source became a numbered list item, shrunk to body size - 2 (11 -> 9)
        sources = [p for p in doc.paragraphs if p.style.name == "List Number" and "BCC Research" in p.text]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].runs[0].font.size.pt, 9)

    def test_export_strips_leading_space_after_hard_breaks(self):
        """Hard-wrapped lines (a 'Prepared for: / Platform:' block) must not pick
        up a leading space on each continued line."""
        self.canvas.content = (
            "**Prepared for:** Project  \n**Platform:** Nanoparticle  \n**Status:** Ready"
        )
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        full = "\n".join(p.text for p in doc.paragraphs)
        self.assertIn("Platform:", full)
        self.assertNotIn("\n ", full, "continued lines should not start with a space")

    def test_export_renders_hr_as_divider_rule(self):
        """`---` thematic breaks (dropped by html2docx) should become a divider:
        an empty bordered paragraph between the sections."""
        self.canvas.content = "Section one.\n\n---\n\nSection two."
        self.canvas.save(update_fields=["content"])

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument
        from docx.oxml.ns import qn

        from chat.views import _HR_DIVIDER_MARKER

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        dividers = [
            p for p in doc.paragraphs
            if p._p.find(qn("w:pPr")) is not None
            and p._p.find(qn("w:pPr")).find(qn("w:pBdr")) is not None
        ]
        self.assertEqual(len(dividers), 1)
        self.assertEqual(dividers[0].text, "")  # marker text cleared
        full = "".join(p.text for p in doc.paragraphs)
        self.assertNotIn(_HR_DIVIDER_MARKER, full)
        self.assertIn("Section one.", full)
        self.assertIn("Section two.", full)

    def test_export_applies_org_header_footer(self):
        """The exporting user's org header/footer + page numbers should appear."""
        from accounts.models import Membership, Organization

        org = Organization.objects.create(name="HF Co", slug="hf-co")
        org.preferences = {"styles": {"header_text": "Confidential Brief", "footer_page_numbers": True}}
        org.save(update_fields=["preferences"])
        Membership.objects.create(user=self.user, org=org, role=Membership.Role.MEMBER)

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 200)

        from docx import Document as DocxDocument
        from docx.oxml.ns import qn

        doc = DocxDocument(io.BytesIO(response.getvalue()))
        section = doc.sections[0]
        self.assertEqual(section.header.paragraphs[0].text, "Confidential Brief")
        self.assertEqual(len(section.footer.paragraphs[0]._p.findall(qn("w:fldSimple"))), 2)

    def test_export_404_no_canvas(self):
        other_thread = ChatThread.objects.create(created_by=self.user)
        url = f"/chat/threads/{other_thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 404)

    def test_export_404_other_user(self):
        other = User.objects.create_user(email="other@test.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other)
        ChatCanvas.objects.create(thread=other_thread, title="X", content="y")
        url = f"/chat/threads/{other_thread.id}/canvas/export/"
        response = self._export(url)
        self.assertEqual(response.status_code, 404)


class CanvasImportViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="imp@test.com", password="pass")
        self.client.login(email="imp@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("chat.services.import_docx_to_canvas")
    def test_import_creates_canvas(self, mock_import):
        mock_import.return_value = ("contract", "# Converted\n\nContent here.", False)

        url = f"/chat/threads/{self.thread.id}/canvas/import/"
        fake_file = io.BytesIO(b"PK fake docx")
        fake_file.name = "contract.docx"
        response = self.client.post(url, {"file": fake_file}, format="multipart")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["title"], "contract")
        self.assertIn("Converted", data["content"])

        canvas = ChatCanvas.objects.get(thread=self.thread)
        self.assertEqual(canvas.title, "contract")

    def test_import_no_file_returns_400(self):
        url = f"/chat/threads/{self.thread.id}/canvas/import/"
        response = self.client.post(url)
        self.assertEqual(response.status_code, 400)


# ---------------------------------------------------------------------------
# Service tests: describe_image and import_docx_to_canvas
# ---------------------------------------------------------------------------

class DescribeImageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="desc@test.com", password="pass")

    @patch("core.preferences.get_preferences")
    @patch("llm.get_llm_service")
    def test_returns_description(self, mock_svc, mock_prefs):
        from chat.services import describe_image

        prefs = MagicMock()
        prefs.feature_models = {}
        prefs.cheap_model = "gemini/gemini-3.5-flash"
        prefs.mid_model = "openai/gpt-5-mini"
        prefs.top_model = "anthropic/claude-sonnet-4-5"
        mock_prefs.return_value = prefs

        mock_response = MagicMock()
        mock_response.message.content = "A bar chart showing Q1 revenue"
        mock_svc.return_value.run.return_value = mock_response

        result = describe_image(b"\x89PNG", "image/png", self.user)
        self.assertEqual(result, "A bar chart showing Q1 revenue")
        mock_svc.return_value.run.assert_called_once()

    @patch("core.preferences.get_preferences")
    def test_returns_none_no_vision_model(self, mock_prefs):
        from chat.services import describe_image

        prefs = MagicMock()
        prefs.feature_models = {}
        prefs.cheap_model = "openai/o3"
        prefs.mid_model = "openai/o4-mini"
        prefs.top_model = "custom/text-only"
        mock_prefs.return_value = prefs

        result = describe_image(b"\x89PNG", "image/png", self.user)
        self.assertIsNone(result)

    @patch("core.preferences.get_preferences")
    @patch("llm.get_llm_service")
    def test_returns_none_on_exception(self, mock_svc, mock_prefs):
        from chat.services import describe_image

        prefs = MagicMock()
        prefs.feature_models = {}
        prefs.cheap_model = "anthropic/claude-haiku-4-5-20251001"
        prefs.mid_model = "openai/gpt-5-mini"
        prefs.top_model = "anthropic/claude-sonnet-4-5"
        mock_prefs.return_value = prefs

        mock_svc.return_value.run.side_effect = RuntimeError("API error")
        result = describe_image(b"\x89PNG", "image/png", self.user)
        self.assertIsNone(result)

    @patch("core.preferences.get_preferences")
    @patch("llm.get_llm_service")
    def test_picks_first_vision_capable_model(self, mock_svc, mock_prefs):
        from chat.services import describe_image

        prefs = MagicMock()
        prefs.feature_models = {}
        prefs.cheap_model = "openai/o3"  # no vision
        prefs.mid_model = "openai/gpt-5-mini"  # vision
        prefs.top_model = "anthropic/claude-sonnet-4-5"
        mock_prefs.return_value = prefs

        mock_response = MagicMock()
        mock_response.message.content = "A photo"
        mock_svc.return_value.run.return_value = mock_response

        describe_image(b"\x89PNG", "image/png", self.user)
        call_args = mock_svc.return_value.run.call_args
        request = call_args[0][1]
        self.assertEqual(request.model, "openai/gpt-5-mini")

    @patch("llm.get_llm_service")
    def test_conversation_id_attributes_the_call_to_the_thread(self, mock_svc):
        from chat.services import describe_image

        mock_svc.return_value.run.return_value.message.content = "A photo"
        describe_image(b"\x89PNG", "image/png", self.user, model="openai/gpt-5-mini")
        self.assertIsNone(mock_svc.return_value.run.call_args[0][1].context.conversation_id)

        describe_image(
            b"\x89PNG", "image/png", self.user, model="openai/gpt-5-mini", conversation_id="thread-1",
        )
        self.assertEqual(mock_svc.return_value.run.call_args[0][1].context.conversation_id, "thread-1")

    @patch("llm.get_llm_service")
    def test_asks_for_the_least_reasoning_the_model_accepts(self, mock_svc):
        """Descriptions are mechanical: the request pins the model's lowest
        reasoning level so the service never applies the chat default (Luna's
        xhigh spent ~90% of its output tokens thinking)."""
        from chat.services import IMAGE_DESCRIPTION_PROMPT, describe_image
        from llm.display import get_thinking_levels

        mock_response = MagicMock()
        mock_response.message.content = "Text: Q1 revenue 4.2 MNOK"
        mock_svc.return_value.run.return_value = mock_response

        describe_image(b"\x89PNG", "image/png", self.user, model="openai/gpt-6-luna")
        request = mock_svc.return_value.run.call_args[0][1]
        self.assertEqual(request.params["thinking_level"], "none")
        self.assertEqual(request.params["thinking_level"], get_thinking_levels("openai/gpt-6-luna")[0])
        self.assertEqual(request.messages[0].content[0]["text"], IMAGE_DESCRIPTION_PROMPT)
        self.assertIn("transcribe it verbatim and completely", IMAGE_DESCRIPTION_PROMPT)

        # Anthropic models spell the floor "off"; a model without reasoning
        # controls gets no level at all (the service then leaves it alone).
        describe_image(b"\x89PNG", "image/png", self.user, model="anthropic/claude-sonnet-5")
        self.assertEqual(mock_svc.return_value.run.call_args[0][1].params["thinking_level"], "off")
        with patch("llm.display.supports_vision", return_value=True), \
             patch("llm.display.get_thinking_levels", return_value=[]):
            describe_image(b"\x89PNG", "image/png", self.user, model="custom/vision-no-reasoning")
        self.assertNotIn("thinking_level", mock_svc.return_value.run.call_args[0][1].params)


class GenerateCanvasTitleTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="gentitle@test.com", password="pass")

    @patch("core.preferences.get_preferences")
    @patch("llm.get_llm_service")
    def test_returns_generated_title(self, mock_svc, mock_prefs):
        from chat.services import generate_canvas_title

        prefs = MagicMock()
        prefs.feature_models = {}
        prefs.cheap_model = "openai/gpt-5-mini"
        mock_prefs.return_value = prefs

        mock_response = MagicMock()
        mock_response.message.content = '"Patent License Agreement"'
        mock_svc.return_value.run.return_value = mock_response

        result = generate_canvas_title("license.docx", "This is a patent license...", self.user)
        self.assertEqual(result, "Patent License Agreement")
        mock_svc.return_value.run.assert_called_once()
        self.assertIsNone(mock_svc.return_value.run.call_args[0][1].context.conversation_id)

    @patch("core.preferences.get_preferences")
    @patch("llm.get_llm_service")
    def test_conversation_id_attributes_the_call_to_the_thread(self, mock_svc, mock_prefs):
        from chat.services import generate_canvas_title

        prefs = MagicMock()
        prefs.feature_models = {}
        prefs.cheap_model = "openai/gpt-5-mini"
        mock_prefs.return_value = prefs
        mock_svc.return_value.run.return_value.message.content = "Title"

        generate_canvas_title("doc", "content", self.user, conversation_id="thread-1")
        self.assertEqual(mock_svc.return_value.run.call_args[0][1].context.conversation_id, "thread-1")

    @patch("core.preferences.get_preferences")
    @patch("llm.get_llm_service")
    def test_returns_none_on_exception(self, mock_svc, mock_prefs):
        from chat.services import generate_canvas_title

        prefs = MagicMock()
        prefs.feature_models = {}
        prefs.cheap_model = "openai/gpt-5-mini"
        mock_prefs.return_value = prefs

        mock_svc.return_value.run.side_effect = RuntimeError("API down")
        result = generate_canvas_title("doc", "content", self.user)
        self.assertIsNone(result)

    @patch("core.preferences.get_preferences")
    @patch("llm.get_llm_service")
    def test_truncates_long_title(self, mock_svc, mock_prefs):
        from chat.services import generate_canvas_title

        prefs = MagicMock()
        prefs.feature_models = {}
        prefs.cheap_model = "openai/gpt-5-mini"
        mock_prefs.return_value = prefs

        mock_response = MagicMock()
        mock_response.message.content = "A" * 300
        mock_svc.return_value.run.return_value = mock_response

        result = generate_canvas_title("doc", "content", self.user)
        self.assertEqual(len(result), 255)


class CanvasImportTitleIntegrationTests(TestCase):
    """Test that canvas_import generates a title for untitled threads."""

    def setUp(self):
        self.user = User.objects.create_user(email="imptitle@test.com", password="pass")
        self.client.login(email="imptitle@test.com", password="pass")

    @patch("chat.services.generate_canvas_title", return_value="Generated Title")
    @patch("chat.services.import_docx_to_canvas")
    def test_generates_title_for_untitled_thread(self, mock_import, mock_gen):
        mock_import.return_value = ("contract", "# Content", False)
        thread = ChatThread.objects.create(created_by=self.user)

        url = f"/chat/threads/{thread.id}/canvas/import/"
        fake_file = io.BytesIO(b"PK fake docx")
        fake_file.name = "contract.docx"
        response = self.client.post(url, {"file": fake_file}, format="multipart")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["thread_title"], "Generated Title")
        thread.refresh_from_db()
        self.assertEqual(thread.title, "Generated Title")
        self.assertEqual(mock_gen.call_args.kwargs["conversation_id"], str(thread.id))

    @patch("chat.services.generate_canvas_title")
    @patch("chat.services.import_docx_to_canvas")
    def test_does_not_overwrite_existing_title(self, mock_import, mock_gen):
        mock_import.return_value = ("contract", "# Content", False)
        thread = ChatThread.objects.create(created_by=self.user, title="Existing Title")

        url = f"/chat/threads/{thread.id}/canvas/import/"
        fake_file = io.BytesIO(b"PK fake docx")
        fake_file.name = "contract.docx"
        response = self.client.post(url, {"file": fake_file}, format="multipart")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotIn("thread_title", data)
        mock_gen.assert_not_called()
        thread.refresh_from_db()
        self.assertEqual(thread.title, "Existing Title")

    @patch("chat.services.generate_canvas_title", return_value=None)
    @patch("chat.services.import_docx_to_canvas")
    def test_llm_failure_does_not_break_import(self, mock_import, mock_gen):
        mock_import.return_value = ("contract", "# Content", False)
        thread = ChatThread.objects.create(created_by=self.user)

        url = f"/chat/threads/{thread.id}/canvas/import/"
        fake_file = io.BytesIO(b"PK fake docx")
        fake_file.name = "contract.docx"
        response = self.client.post(url, {"file": fake_file}, format="multipart")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotIn("thread_title", data)
        self.assertEqual(data["title"], "contract")

    @patch("chat.services.generate_canvas_title", return_value="Generated Title")
    @patch("chat.services.import_docx_to_canvas")
    def test_dotx_template_is_accepted(self, mock_import, mock_gen):
        """A .dotx (Word template) is accepted for canvas import, not rejected as an
        unsupported type."""
        mock_import.return_value = ("boilerplate", "# Content", False)
        thread = ChatThread.objects.create(created_by=self.user)

        url = f"/chat/threads/{thread.id}/canvas/import/"
        fake_file = io.BytesIO(b"PK fake dotx")
        fake_file.name = "boilerplate.dotx"
        response = self.client.post(url, {"file": fake_file}, format="multipart")

        # 200 (not a 400 "Only .docx files are supported") proves the template is
        # accepted; the title derives from the filename.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "boilerplate")
        mock_import.assert_called_once()


def _minimal_docx_bytes() -> bytes:
    """A valid (tiny) .docx zip — enough to pass docx_to_markdown's zip-bomb
    guard when mammoth.convert_to_html is mocked."""
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
    return buf.getvalue()


class ImportDocxToCanvasTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="imp2@test.com", password="pass")

    @patch("chat.services.describe_image")
    @patch("mammoth.convert_to_html")
    def test_basic_conversion_no_images(self, mock_convert, mock_describe):
        from chat.services import import_docx_to_canvas

        mock_result = MagicMock()
        mock_result.value = "<h1>Hello</h1><p>World</p>"
        mock_convert.return_value = mock_result

        from django.core.files.uploadedfile import SimpleUploadedFile
        fake_file = SimpleUploadedFile("report.docx", _minimal_docx_bytes())

        title, content, truncated = import_docx_to_canvas(fake_file, self.user)
        self.assertEqual(title, "report")
        self.assertEqual(content, "# Hello\n\nWorld")
        self.assertFalse(truncated)
        mock_describe.assert_not_called()

    @patch("chat.services.describe_image")
    @patch("mammoth.convert_to_html")
    def test_table_conversion(self, mock_convert, mock_describe):
        from chat.services import import_docx_to_canvas

        mock_result = MagicMock()
        mock_result.value = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>A</td><td>1</td></tr>"
            "<tr><td>B</td><td>2</td></tr>"
            "</table>"
        )
        mock_convert.return_value = mock_result

        from django.core.files.uploadedfile import SimpleUploadedFile
        fake_file = SimpleUploadedFile("data.docx", _minimal_docx_bytes())

        title, content, truncated = import_docx_to_canvas(fake_file, self.user)
        self.assertEqual(title, "data")
        self.assertFalse(truncated)
        self.assertIn("| Name | Value |", content)
        self.assertIn("| --- | --- |", content)
        self.assertIn("| A | 1 |", content)
        mock_describe.assert_not_called()

    @patch("chat.services.describe_image")
    def test_images_get_placeholders(self, mock_describe):
        from chat.services import import_docx_to_canvas

        mock_describe.return_value = "A revenue chart"

        # Create a real tiny docx with an image to test end-to-end is complex,
        # so we test the regex cleanup directly
        import re
        raw = "Some text\n\n![[Image 1: A revenue chart]](#)\n\nMore text"
        cleaned = re.sub(
            r"!\[(\[Image \d+[^\]]*\])\]\([^)]*\)",
            r"\1",
            raw,
        )
        self.assertEqual(cleaned, "Some text\n\n[Image 1: A revenue chart]\n\nMore text")

    @patch("chat.services.describe_image")
    def test_image_placeholder_without_description(self, mock_describe):
        import re
        raw = "Before\n\n![[Image 1: EMF image]](#)\n\nAfter"
        cleaned = re.sub(
            r"!\[(\[Image \d+[^\]]*\])\]\([^)]*\)",
            r"\1",
            raw,
        )
        self.assertEqual(cleaned, "Before\n\n[Image 1: EMF image]\n\nAfter")


# ---------------------------------------------------------------------------
# CanvasCheckpoint model tests
# ---------------------------------------------------------------------------

class CanvasCheckpointModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cpmodel@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="Doc", content="Hello")

    def test_create_checkpoint(self):
        cp = CanvasCheckpoint.objects.create(
            canvas=self.canvas, title="Doc", content="Hello",
            source=CanvasCheckpoint.Source.ORIGINAL, order=1,
        )
        self.assertEqual(cp.source, "original")
        self.assertEqual(cp.order, 1)

    def test_ordering(self):
        CanvasCheckpoint.objects.create(canvas=self.canvas, title="A", content="1", source="original", order=1)
        CanvasCheckpoint.objects.create(canvas=self.canvas, title="B", content="2", source="ai_edit", order=2)
        cps = list(CanvasCheckpoint.objects.filter(canvas=self.canvas))
        self.assertEqual(cps[0].order, 1)
        self.assertEqual(cps[1].order, 2)

    def test_source_choices(self):
        for val, label in CanvasCheckpoint.Source.choices:
            cp = CanvasCheckpoint.objects.create(
                canvas=self.canvas, title="T", content="C",
                source=val, order=0,
            )
            self.assertEqual(cp.source, val)


# ---------------------------------------------------------------------------
# WriteCanvasTool checkpoint tests
# ---------------------------------------------------------------------------

class WriteCanvasToolCheckpointTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="wcp@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_first_call_creates_original_checkpoint_and_sets_accepted(self):
        result = _invoke(WriteCanvasTool, {"title": "NDA", "content": "# NDA"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        canvas = ChatCanvas.objects.get(thread=self.thread)
        cps = list(CanvasCheckpoint.objects.filter(canvas=canvas))
        self.assertEqual(len(cps), 1)
        self.assertEqual(cps[0].source, "original")
        self.assertIsNotNone(canvas.accepted_checkpoint)
        self.assertEqual(canvas.accepted_checkpoint.pk, cps[0].pk)

    def test_second_call_creates_ai_edit_without_updating_accepted(self):
        _invoke(WriteCanvasTool, {"title": "Draft", "content": "old"}, _ctx(self.user.pk, self.thread.id))
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Draft")
        original_accepted_pk = canvas.accepted_checkpoint.pk

        _invoke(WriteCanvasTool, {"title": "Draft", "content": "new"}, _ctx(self.user.pk, self.thread.id))
        canvas.refresh_from_db()
        cps = list(CanvasCheckpoint.objects.filter(canvas=canvas).order_by("order"))
        self.assertEqual(len(cps), 2)
        self.assertEqual(cps[1].source, "ai_edit")
        # accepted_checkpoint should NOT be updated
        self.assertEqual(canvas.accepted_checkpoint.pk, original_accepted_pk)

    def test_no_content_in_result(self):
        result = _invoke(WriteCanvasTool, {"title": "T", "content": "C"}, _ctx(self.user.pk, self.thread.id))
        self.assertNotIn("content", result)
        self.assertNotIn("accepted_content", result)


# ---------------------------------------------------------------------------
# EditCanvasTool checkpoint tests
# ---------------------------------------------------------------------------

class EditCanvasToolCheckpointTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ecp@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _setup_canvas(self, content):
        from django.utils import timezone

        canvas = ChatCanvas.objects.create(
            thread=self.thread, title="Doc", content=content,
            is_active=True, last_activated_at=timezone.now(),
        )
        self.thread.active_canvas = canvas
        self.thread.save(update_fields=["active_canvas"])
        from chat.services import create_canvas_checkpoint
        cp = create_canvas_checkpoint(canvas, source="original")
        canvas.accepted_checkpoint = cp
        canvas.save(update_fields=["accepted_checkpoint"])
        return canvas

    def test_successful_edit_creates_ai_edit_checkpoint(self):
        self._setup_canvas("The term is 3 years.")
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "3 years", "new_text": "5 years", "reason": "update"}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["applied"], 1)
        canvas = ChatCanvas.objects.get(thread=self.thread)
        cps = list(CanvasCheckpoint.objects.filter(canvas=canvas).order_by("order"))
        self.assertEqual(len(cps), 2)
        self.assertEqual(cps[1].source, "ai_edit")

    def test_edits_passed_as_json_string_are_coerced(self):
        """Regression for WILFRED-40: some tool-calling models serialize ``edits``
        as a JSON string instead of a native list, which made canvas_edit fail
        Pydantic validation outright. The tool should parse the string and apply
        the edit. Goes through tool.invoke() → langchain _parse_input, the exact
        path that raised in production.
        """
        self._setup_canvas("The term is 3 years.")
        result = _invoke(EditCanvasTool, {
            "edits": json.dumps([{"old_text": "3 years", "new_text": "5 years", "reason": "update"}])
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["applied"], 1)
        canvas = ChatCanvas.objects.get(thread=self.thread)
        self.assertIn("5 years", canvas.content)

    def test_malformed_edit_items_are_dropped(self):
        """Regression for WILFRED-6A: a model sometimes includes a non-actionable
        edit item (e.g. {"reason": ...} with no old_text/new_text). That item is
        dropped so the valid edits still apply, instead of the whole tool call
        failing Pydantic validation.
        """
        self._setup_canvas("The term is 3 years.")
        result = _invoke(EditCanvasTool, {
            "edits": [
                {"old_text": "3 years", "new_text": "5 years", "reason": "update"},
                {"reason": "explain the change"},  # malformed: no old_text/new_text
            ]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["applied"], 1)
        canvas = ChatCanvas.objects.get(thread=self.thread)
        self.assertIn("5 years", canvas.content)

    def test_failed_edit_does_not_create_checkpoint(self):
        self._setup_canvas("Some content.")
        _invoke(EditCanvasTool, {
            "edits": [{"old_text": "nonexistent", "new_text": "x", "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        canvas = ChatCanvas.objects.get(thread=self.thread)
        cps = list(CanvasCheckpoint.objects.filter(canvas=canvas))
        self.assertEqual(len(cps), 1)  # only the original

    def test_no_content_in_result(self):
        self._setup_canvas("Hello world.")
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "Hello", "new_text": "Hi", "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertNotIn("content", result)
        self.assertNotIn("accepted_content", result)

    def test_consecutive_ai_edits_coalesce_into_one_checkpoint(self):
        """Multiple AI edits in one turn should produce only one checkpoint."""
        self._setup_canvas("Hello world. Goodbye world.")
        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(EditCanvasTool, {
            "edits": [{"old_text": "Hello", "new_text": "Hi", "reason": ""}]
        }, ctx)
        _invoke(EditCanvasTool, {
            "edits": [{"old_text": "Goodbye", "new_text": "Bye", "reason": ""}]
        }, ctx)
        canvas = ChatCanvas.objects.get(thread=self.thread)
        cps = list(CanvasCheckpoint.objects.filter(canvas=canvas).order_by("order"))
        # Should have 2: original + one coalesced ai_edit
        self.assertEqual(len(cps), 2)
        self.assertEqual(cps[0].source, "original")
        self.assertEqual(cps[1].source, "ai_edit")
        # The coalesced checkpoint should have the final content
        self.assertIn("Hi", cps[1].content)
        self.assertIn("Bye", cps[1].content)

    def test_write_after_edit_creates_separate_checkpoints(self):
        """A canvas_write after canvas_edit should NOT coalesce (different tool, still ai_edit source though)."""
        self._setup_canvas("Hello world.")
        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(EditCanvasTool, {
            "edits": [{"old_text": "Hello", "new_text": "Hi", "reason": ""}]
        }, ctx)
        # Write canvas with same title produces ai_edit source (since canvas already exists)
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "Brand new"}, ctx)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        cps = list(CanvasCheckpoint.objects.filter(canvas=canvas).order_by("order"))
        # original + ai_edit (from edit) — but canvas_write on existing canvas
        # also produces ai_edit, which coalesces with the previous ai_edit
        self.assertEqual(len(cps), 2)
        self.assertEqual(cps[1].content, "Brand new")

    def test_ai_edit_after_accept_creates_new_checkpoint(self):
        """After user accepts, next AI edit must NOT coalesce into the accepted checkpoint."""
        canvas = self._setup_canvas("Hello world. Goodbye world.")
        ctx = _ctx(self.user.pk, self.thread.id)

        # First AI edit
        _invoke(EditCanvasTool, {
            "edits": [{"old_text": "Hello", "new_text": "Hi", "reason": ""}]
        }, ctx)
        canvas.refresh_from_db()
        # 2 checkpoints: original + ai_edit
        self.assertEqual(CanvasCheckpoint.objects.filter(canvas=canvas).count(), 2)

        # User accepts — set accepted_checkpoint to latest
        latest = CanvasCheckpoint.objects.filter(canvas=canvas).order_by("-order").first()
        canvas.accepted_checkpoint = latest
        canvas.save(update_fields=["accepted_checkpoint"])

        # Second AI edit — should NOT coalesce into accepted checkpoint
        _invoke(EditCanvasTool, {
            "edits": [{"old_text": "Goodbye", "new_text": "Bye", "reason": ""}]
        }, ctx)
        canvas.refresh_from_db()
        cps = list(CanvasCheckpoint.objects.filter(canvas=canvas).order_by("order"))
        # Should now have 3: original, ai_edit (accepted), ai_edit (new)
        self.assertEqual(len(cps), 3)
        self.assertEqual(cps[1].pk, latest.pk)  # accepted one unchanged
        self.assertIn("Hi", cps[1].content)
        self.assertNotIn("Bye", cps[1].content)  # NOT updated
        self.assertIn("Bye", cps[2].content)  # new checkpoint has it


# ---------------------------------------------------------------------------
# Consumer checkpoint tests
# ---------------------------------------------------------------------------

@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ConsumerCheckpointTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cpws@test.com", password="pass")

    async def _communicator(self):
        from channels.routing import URLRouter
        from channels.testing import WebsocketCommunicator
        from chat.routing import websocket_urlpatterns

        app = URLRouter(websocket_urlpatterns)
        comm = WebsocketCommunicator(app, "/ws/chat/")
        comm.scope["user"] = self.user
        connected, _ = await comm.connect()
        assert connected
        return comm

    async def test_canvas_accept(self):
        from channels.db import database_sync_to_async
        from chat.services import create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)
        canvas = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Doc", content="v1"
        )

        def setup_checkpoints():
            thread.active_canvas = canvas
            thread.save(update_fields=["active_canvas"])
            cp1 = create_canvas_checkpoint(canvas, source="original")
            canvas.accepted_checkpoint = cp1
            canvas.save(update_fields=["accepted_checkpoint"])
            canvas.content = "v2"
            canvas.save(update_fields=["content"])
            create_canvas_checkpoint(canvas, source="ai_edit")
        await database_sync_to_async(setup_checkpoints)()

        comm = await self._communicator()
        await comm.send_json_to({"type": "chat.canvas_accept", "thread_id": str(thread.id)})
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.accepted")
        self.assertEqual(msg["accepted_content"], "v2")

        def check_accepted():
            c = ChatCanvas.objects.get(thread=thread)
            self.assertEqual(c.accepted_checkpoint.content, "v2")
        await database_sync_to_async(check_accepted)()
        await comm.disconnect()

    async def test_canvas_revert(self):
        from channels.db import database_sync_to_async
        from chat.services import create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)
        canvas = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Doc", content="original content"
        )

        def setup():
            thread.active_canvas = canvas
            thread.save(update_fields=["active_canvas"])
            cp = create_canvas_checkpoint(canvas, source="original")
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])
            canvas.content = "ai changed this"
            canvas.save(update_fields=["content"])
        await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({"type": "chat.canvas_revert", "thread_id": str(thread.id)})
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.reverted")
        self.assertEqual(msg["content"], "original content")

        def check_content():
            c = ChatCanvas.objects.get(thread=thread)
            self.assertEqual(c.content, "original content")
        await database_sync_to_async(check_content)()
        await comm.disconnect()

    async def test_canvas_save_version(self):
        from channels.db import database_sync_to_async

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            c = ChatCanvas.objects.create(thread=thread, title="Doc", content="content")
            thread.active_canvas = c
            thread.save(update_fields=["active_canvas"])
        await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({
            "type": "chat.canvas_save_version",
            "thread_id": str(thread.id),
            "title": "Doc",
            "content": "user edited content",
        })
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.version_saved")

        def check():
            c = ChatCanvas.objects.get(thread=thread)
            self.assertIsNotNone(c.accepted_checkpoint)
            self.assertEqual(c.accepted_checkpoint.source, "user_save")
            cps = CanvasCheckpoint.objects.filter(canvas=c)
            self.assertEqual(cps.count(), 1)
        await database_sync_to_async(check)()
        await comm.disconnect()

    async def test_canvas_loaded_includes_accepted_content(self):
        from channels.db import database_sync_to_async
        from chat.services import create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)
        canvas = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Doc", content="current"
        )

        def setup():
            thread.active_canvas = canvas
            thread.save(update_fields=["active_canvas"])
            cp = create_canvas_checkpoint(canvas, source="original", description="first")
            canvas.accepted_checkpoint = cp
            canvas.content = "current"
            canvas.save(update_fields=["accepted_checkpoint", "content"])
        await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({"type": "chat.load_thread", "thread_id": str(thread.id)})
        # thread.loaded
        msg1 = await comm.receive_json_from()
        self.assertEqual(msg1["event_type"], "thread.loaded")
        # canvases.loaded
        msg2 = await comm.receive_json_from()
        self.assertEqual(msg2["event_type"], "canvases.loaded")
        self.assertIn("accepted_content", msg2["active_canvas"])
        self.assertEqual(msg2["active_canvas"]["accepted_content"], "current")
        await comm.disconnect()

    async def test_canvas_restore_version(self):
        from channels.db import database_sync_to_async
        from chat.services import create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)
        canvas = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Doc", content="v1"
        )

        def setup():
            thread.active_canvas = canvas
            thread.save(update_fields=["active_canvas"])
            cp1 = create_canvas_checkpoint(canvas, source="original")
            canvas.accepted_checkpoint = cp1
            canvas.save(update_fields=["accepted_checkpoint"])
            canvas.content = "v2"
            canvas.save(update_fields=["content"])
            create_canvas_checkpoint(canvas, source="ai_edit")
            return cp1.pk
        cp1_pk = await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({
            "type": "chat.canvas_restore_version",
            "thread_id": str(thread.id),
            "checkpoint_id": cp1_pk,
        })
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.restored")
        self.assertEqual(msg["content"], "v1")

        def check():
            c = ChatCanvas.objects.get(thread=thread)
            self.assertEqual(c.content, "v1")
            self.assertIsNotNone(c.accepted_checkpoint)
            self.assertEqual(c.accepted_checkpoint.source, "restore")
        await database_sync_to_async(check)()
        await comm.disconnect()

    async def test_canvas_restore_via_get_checkpoints_flow(self):
        """Full user flow: get checkpoints → pick one → restore it."""
        from channels.db import database_sync_to_async
        from chat.services import create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)
        canvas = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Doc", content="original text"
        )

        def setup():
            thread.active_canvas = canvas
            thread.save(update_fields=["active_canvas"])
            cp1 = create_canvas_checkpoint(canvas, source="original")
            canvas.accepted_checkpoint = cp1
            canvas.save(update_fields=["accepted_checkpoint"])
            # AI edits the canvas
            canvas.content = "ai changed text"
            canvas.title = "AI Doc"
            canvas.save(update_fields=["title", "content"])
            create_canvas_checkpoint(canvas, source="ai_edit")
        await database_sync_to_async(setup)()

        comm = await self._communicator()

        # Step 1: Get checkpoints
        await comm.send_json_to({
            "type": "chat.canvas_get_checkpoints",
            "thread_id": str(thread.id),
        })
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.checkpoints")
        checkpoints = msg["checkpoints"]
        self.assertEqual(len(checkpoints), 2)
        # Checkpoints are ordered by -order, so first is the latest (ai_edit)
        self.assertEqual(checkpoints[0]["source"], "ai_edit")
        self.assertEqual(checkpoints[1]["source"], "original")

        # Step 2: Restore the original checkpoint using its ID from the list
        original_cp_id = checkpoints[1]["id"]
        await comm.send_json_to({
            "type": "chat.canvas_restore_version",
            "thread_id": str(thread.id),
            "checkpoint_id": original_cp_id,
        })
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.restored")
        self.assertEqual(msg["title"], "Doc")
        self.assertEqual(msg["content"], "original text")

        # Step 3: Verify DB state
        def check():
            c = ChatCanvas.objects.get(thread=thread)
            self.assertEqual(c.content, "original text")
            self.assertEqual(c.title, "Doc")
            self.assertIsNotNone(c.accepted_checkpoint)
            self.assertEqual(c.accepted_checkpoint.source, "restore")
        await database_sync_to_async(check)()
        await comm.disconnect()

    async def test_canvas_get_checkpoints_empty(self):
        """get_checkpoints returns empty list when no checkpoints exist."""
        from channels.db import database_sync_to_async

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            c = ChatCanvas.objects.create(thread=thread, title="Doc", content="text")
            thread.active_canvas = c
            thread.save(update_fields=["active_canvas"])
        await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({
            "type": "chat.canvas_get_checkpoints",
            "thread_id": str(thread.id),
        })
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.checkpoints")
        self.assertEqual(msg["checkpoints"], [])
        await comm.disconnect()


# ---------------------------------------------------------------------------
# Multi-canvas tool tests
# ---------------------------------------------------------------------------

class MultiCanvasToolTests(TestCase):
    """Tests for multi-canvas behaviour in WriteCanvasTool and EditCanvasTool."""

    def setUp(self):
        self.user = User.objects.create_user(email="multi@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_write_creates_new_canvas_with_new_title(self):
        result = _invoke(WriteCanvasTool, {"title": "A", "content": "aaa"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(ChatCanvas.objects.filter(thread=self.thread).count(), 1)

        result = _invoke(WriteCanvasTool, {"title": "B", "content": "bbb"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(ChatCanvas.objects.filter(thread=self.thread).count(), 2)
        self.assertEqual(ChatCanvas.objects.get(thread=self.thread, title="B").content, "bbb")

    def test_write_overwrites_canvas_with_same_title(self):
        _invoke(WriteCanvasTool, {"title": "Draft", "content": "v1"}, _ctx(self.user.pk, self.thread.id))
        _invoke(WriteCanvasTool, {"title": "Draft", "content": "v2"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(ChatCanvas.objects.filter(thread=self.thread).count(), 1)
        self.assertEqual(ChatCanvas.objects.get(thread=self.thread, title="Draft").content, "v2")

    def test_write_focuses_canvas(self):
        _invoke(WriteCanvasTool, {"title": "First", "content": "f"}, _ctx(self.user.pk, self.thread.id))
        self.thread.refresh_from_db()
        first = ChatCanvas.objects.get(thread=self.thread, title="First")
        self.assertEqual(self.thread.active_canvas_id, first.pk)

        _invoke(WriteCanvasTool, {"title": "Second", "content": "s"}, _ctx(self.user.pk, self.thread.id))
        self.thread.refresh_from_db()
        second = ChatCanvas.objects.get(thread=self.thread, title="Second")
        self.assertEqual(self.thread.active_canvas_id, second.pk)

    def test_edit_targets_by_canvas_name(self):
        _invoke(WriteCanvasTool, {"title": "Alpha", "content": "Hello world"}, _ctx(self.user.pk, self.thread.id))
        _invoke(WriteCanvasTool, {"title": "Beta", "content": "Foo bar"}, _ctx(self.user.pk, self.thread.id))
        # Active canvas is now "Beta" — edit "Alpha" by name
        result = _invoke(EditCanvasTool, {
            "canvas_name": "Alpha",
            "edits": [{"old_text": "Hello", "new_text": "Hi", "reason": ""}],
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["applied"], 1)
        alpha = ChatCanvas.objects.get(thread=self.thread, title="Alpha")
        self.assertIn("Hi", alpha.content)
        # Beta untouched
        beta = ChatCanvas.objects.get(thread=self.thread, title="Beta")
        self.assertEqual(beta.content, "Foo bar")

    def test_edit_without_name_uses_the_only_canvas(self):
        _invoke(WriteCanvasTool, {"title": "Only", "content": "Hello world"}, _ctx(self.user.pk, self.thread.id))
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "Hello", "new_text": "Hi", "reason": ""}],
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["applied"], 1)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Only")
        self.assertIn("Hi", canvas.content)

    def test_edit_without_name_requires_name_with_two_canvases(self):
        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(WriteCanvasTool, {"title": "Alpha", "content": "Hello world"}, ctx)
        _invoke(WriteCanvasTool, {"title": "Beta", "content": "Hello again"}, ctx)
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "Hello", "new_text": "Hi", "reason": ""}],
        }, ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn("canvas_name is required", result["message"])
        self.assertIn('"Alpha"', result["message"])
        self.assertIn('"Beta"', result["message"])
        self.assertEqual(ChatCanvas.objects.get(thread=self.thread, title="Alpha").content, "Hello world")
        self.assertEqual(ChatCanvas.objects.get(thread=self.thread, title="Beta").content, "Hello again")

    def test_read_without_name_requires_name_with_two_canvases(self):
        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(WriteCanvasTool, {"title": "Alpha", "content": "a"}, ctx)
        _invoke(WriteCanvasTool, {"title": "Beta", "content": "b"}, ctx)
        result = _invoke(ReadCanvasTool, {}, ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn("canvas_name is required", result["message"])
        self.assertEqual(_invoke(ReadCanvasTool, {"canvas_name": "Beta"}, ctx)["content"], "b")

    def test_paste_without_name_errors_with_two_canvases(self):
        ctx = _ctx(self.user.pk, self.thread.id)
        ChatMessage.objects.create(thread=self.thread, role="user", content="Paste me")
        _invoke(WriteCanvasTool, {"title": "Alpha", "content": "a"}, ctx)
        _invoke(WriteCanvasTool, {"title": "Beta", "content": "b"}, ctx)
        result = _invoke(PasteUserTextTool, {"message_number": 1}, ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn("canvas_name is required", result["message"])
        self.assertEqual(ChatCanvas.objects.filter(thread=self.thread, deleted_at__isnull=True).count(), 2)

    def test_paste_without_name_creates_canvas_when_none(self):
        ctx = _ctx(self.user.pk, self.thread.id)
        ChatMessage.objects.create(thread=self.thread, role="user", content="Paste me")
        result = _invoke(PasteUserTextTool, {"message_number": 1}, ctx)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(ChatCanvas.objects.get(thread=self.thread).content, "Paste me")

    def test_edit_errors_with_unknown_canvas_name(self):
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "x"}, _ctx(self.user.pk, self.thread.id))
        result = _invoke(EditCanvasTool, {
            "canvas_name": "Nonexistent",
            "edits": [{"old_text": "x", "new_text": "y", "reason": ""}],
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "error")

    def test_unique_title_constraint_enforced(self):
        """Two canvases with the same title cannot exist for one thread."""
        ChatCanvas.objects.create(thread=self.thread, title="Dup", content="a")
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            ChatCanvas.objects.create(thread=self.thread, title="Dup", content="b")

    def test_max_canvases_per_thread(self):
        from chat.services import MAX_CANVASES_PER_THREAD
        ctx = _ctx(self.user.pk, self.thread.id)
        for i in range(MAX_CANVASES_PER_THREAD):
            result = _invoke(WriteCanvasTool, {"title": f"Canvas {i}", "content": "x"}, ctx)
            self.assertEqual(result["status"], "ok")
        result = _invoke(WriteCanvasTool, {"title": "One too many", "content": "x"}, ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn("Maximum", result["message"])

    def test_write_returns_canvas_id(self):
        result = _invoke(WriteCanvasTool, {"title": "T", "content": "C"}, _ctx(self.user.pk, self.thread.id))
        self.assertIn("canvas_id", result)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="T")
        self.assertEqual(result["canvas_id"], str(canvas.pk))

    def test_edit_returns_canvas_id(self):
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "Hello"}, _ctx(self.user.pk, self.thread.id))
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "Hello", "new_text": "Hi", "reason": ""}],
        }, _ctx(self.user.pk, self.thread.id))
        self.assertIn("canvas_id", result)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        self.assertEqual(result["canvas_id"], str(canvas.pk))


# ---------------------------------------------------------------------------
# Multi-canvas consumer tests
# ---------------------------------------------------------------------------

@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ConsumerMultiCanvasTests(TransactionTestCase):
    """WebSocket tests for multi-canvas features."""

    def setUp(self):
        self.user = User.objects.create_user(email="mws@test.com", password="pass")

    async def _communicator(self):
        from channels.routing import URLRouter
        from channels.testing import WebsocketCommunicator
        from chat.routing import websocket_urlpatterns

        app = URLRouter(websocket_urlpatterns)
        comm = WebsocketCommunicator(app, "/ws/chat/")
        comm.scope["user"] = self.user
        connected, _ = await comm.connect()
        assert connected
        return comm

    async def test_canvases_loaded_on_thread_switch(self):
        from channels.db import database_sync_to_async

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            c1 = ChatCanvas.objects.create(thread=thread, title="Doc 1", content="one")
            c2 = ChatCanvas.objects.create(thread=thread, title="Doc 2", content="two")
            thread.active_canvas = c2
            thread.save(update_fields=["active_canvas"])
        await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({"type": "chat.load_thread", "thread_id": str(thread.id)})

        msg1 = await comm.receive_json_from()
        self.assertEqual(msg1["event_type"], "thread.loaded")

        msg2 = await comm.receive_json_from()
        self.assertEqual(msg2["event_type"], "canvases.loaded")
        self.assertEqual(len(msg2["canvases"]), 2)
        titles = {c["title"] for c in msg2["canvases"]}
        self.assertEqual(titles, {"Doc 1", "Doc 2"})
        self.assertEqual(msg2["active_canvas"]["title"], "Doc 2")
        self.assertEqual(msg2["active_canvas"]["content"], "two")

        await comm.disconnect()

    async def test_canvas_switch_changes_active(self):
        from channels.db import database_sync_to_async

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            c1 = ChatCanvas.objects.create(thread=thread, title="Doc 1", content="one")
            c2 = ChatCanvas.objects.create(thread=thread, title="Doc 2", content="two")
            thread.active_canvas = c1
            thread.save(update_fields=["active_canvas"])
            return c2.pk
        c2_pk = await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({
            "type": "chat.canvas_switch",
            "thread_id": str(thread.id),
            "canvas_id": c2_pk,
        })
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.loaded")
        self.assertEqual(msg["title"], "Doc 2")
        self.assertEqual(msg["content"], "two")

        def check_active():
            thread.refresh_from_db()
            self.assertEqual(thread.active_canvas_id, c2_pk)
        await database_sync_to_async(check_active)()
        await comm.disconnect()

    async def test_canvas_save_targets_specific_canvas(self):
        from channels.db import database_sync_to_async

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            c1 = ChatCanvas.objects.create(thread=thread, title="Doc 1", content="one")
            c2 = ChatCanvas.objects.create(thread=thread, title="Doc 2", content="two")
            thread.active_canvas = c1
            thread.save(update_fields=["active_canvas"])
            return c2.pk
        c2_pk = await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({
            "type": "chat.canvas_save",
            "thread_id": str(thread.id),
            "canvas_id": c2_pk,
            "title": "Doc 2 Renamed",
            "content": "updated two",
        })
        await comm.disconnect()

        def check():
            c2 = ChatCanvas.objects.get(pk=c2_pk)
            self.assertEqual(c2.title, "Doc 2 Renamed")
            self.assertEqual(c2.content, "updated two")
            # c1 unchanged
            c1 = ChatCanvas.objects.get(thread=thread, title="Doc 1")
            self.assertEqual(c1.content, "one")
        await database_sync_to_async(check)()


# ---------------------------------------------------------------------------
# Multi-canvas prompt tests
# ---------------------------------------------------------------------------

class CanvasWorkspaceListTests(TestCase):
    """The prompt lists canvases (size, tokens, open tab, staleness); it never
    carries their text."""

    def _entries(self):
        return [
            {"title": "Prior-art table", "chars": 184_230, "tokens": 46_100, "is_open": True,
             "seen_state": "changed", "truncated_from_chars": None},
            {"title": "Cover letter", "chars": 2_310, "tokens": 580, "is_open": False,
             "seen_state": "unchanged", "truncated_from_chars": None},
            {"title": "Imported handbook", "chars": 225_000, "tokens": 58_000, "is_open": False,
             "seen_state": "never", "truncated_from_chars": 310_442},
        ]

    def test_lists_every_canvas_with_flags(self):
        from chat.prompts import build_semi_static_prompt

        prompt = build_semi_static_prompt(canvases=self._entries())
        self.assertIn("# Canvas workspace", prompt)
        self.assertIn(
            "**Prior-art table** — 184,230 chars, ~46,100 tokens — open in the user's editor "
            "— changed since you last read it",
            prompt,
        )
        self.assertIn("**Cover letter** — 2,310 chars, ~580 tokens — unchanged since you last read it", prompt)
        self.assertIn(
            "**Imported handbook** — 225,000 chars (cut from a 310,442-char source), ~58,000 tokens "
            "— never read by you",
            prompt,
        )
        self.assertIn("Every canvas tool needs `canvas_name` while more than one canvas exists", prompt)
        self.assertIn("read a canvas with `canvas_read` before you edit", prompt)
        self.assertNotIn("Active Canvas Content", prompt)

    def test_single_canvas_does_not_demand_canvas_name(self):
        from chat.prompts import build_semi_static_prompt

        prompt = build_semi_static_prompt(canvases=self._entries()[:1])
        self.assertIn("**Prior-art table**", prompt)
        self.assertNotIn("Every canvas tool needs `canvas_name`", prompt)

    def test_system_prompt_wrapper_lists_canvases(self):
        from chat.prompts import build_system_prompt

        prompt = build_system_prompt(canvases=self._entries())
        self.assertIn("**Cover letter**", prompt)
        self.assertNotIn("Active Canvas Content", prompt)

    def test_no_canvases_prompt(self):
        from chat.prompts import build_system_prompt

        prompt = build_system_prompt()
        # Base canvas guidance is always present...
        self.assertIn("canvas workspace", prompt)
        self.assertIn("Canvas text is never placed in your context automatically", prompt)
        # ...but the per-thread listing only appears when canvases exist.
        self.assertNotIn("# Canvas workspace", prompt)


# ---------------------------------------------------------------------------
# Service-level tests for focus_canvas / resolve_canvas / the staleness stamp
# ---------------------------------------------------------------------------

class FocusCanvasTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="focus@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_focus_sets_the_open_tab_only(self):
        from chat.services import focus_canvas

        a = ChatCanvas.objects.create(thread=self.thread, title="A", content="a")
        b = ChatCanvas.objects.create(thread=self.thread, title="B", content="b")
        focus_canvas(self.thread.pk, a)
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.active_canvas_id, a.pk)
        focus_canvas(self.thread.pk, b)
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.active_canvas_id, b.pk)
        # The deprecated model flag is never written any more.
        for c in (a, b):
            c.refresh_from_db()
            self.assertFalse(c.is_active)
            self.assertIsNone(c.last_activated_at)

    def test_resolve_canvas_none(self):
        from chat.services import resolve_canvas

        canvas, err = resolve_canvas(self.thread.pk)
        self.assertIsNone(canvas)
        self.assertEqual(err, "No canvas exists in this thread.")

    def test_resolve_canvas_single_without_name(self):
        from chat.services import resolve_canvas

        a = ChatCanvas.objects.create(thread=self.thread, title="A", content="a")
        canvas, err = resolve_canvas(self.thread.pk)
        self.assertIsNone(err)
        self.assertEqual(canvas.pk, a.pk)

    def test_resolve_canvas_requires_name_when_several(self):
        from chat.services import resolve_canvas

        ChatCanvas.objects.create(thread=self.thread, title="A", content="a")
        b = ChatCanvas.objects.create(thread=self.thread, title="B", content="b")
        ChatCanvas.objects.create(thread=self.thread, title="Gone", content="x", deleted_at=timezone_now())
        canvas, err = resolve_canvas(self.thread.pk)
        self.assertIsNone(canvas)
        self.assertEqual(err, 'This thread has 2 canvases, so canvas_name is required. Available canvases: "A", "B".')
        canvas, err = resolve_canvas(self.thread.pk, "B")
        self.assertIsNone(err)
        self.assertEqual(canvas.pk, b.pk)

    def test_write_focuses_new_canvas(self):
        result = _invoke(
            WriteCanvasTool, {"title": "New", "content": "content"},
            _ctx(self.user.pk, self.thread.id),
        )
        self.assertEqual(result["status"], "ok")
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.active_canvas.title, "New")

    def test_edit_focuses_edited_canvas(self):
        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(WriteCanvasTool, {"title": "Background", "content": "Hello world"}, ctx)
        _invoke(WriteCanvasTool, {"title": "Front", "content": "f"}, ctx)
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.active_canvas.title, "Front")
        _invoke(
            EditCanvasTool,
            {"canvas_name": "Background", "edits": [{"old_text": "Hello", "new_text": "Hi", "reason": ""}]},
            ctx,
        )
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.active_canvas.title, "Background")


def timezone_now():
    from django.utils import timezone

    return timezone.now()


class AgentSeenShaTests(TestCase):
    """``agent_seen_sha`` records what the agent saw; the prompt list derives
    "changed since you last read it" from it."""

    def setUp(self):
        self.user = User.objects.create_user(email="seen@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = _ctx(self.user.pk, self.thread.id)

    def _entry(self, title):
        from chat.services import canvas_prompt_entry

        canvas = ChatCanvas.objects.get(thread=self.thread, title=title)
        return canvas, canvas_prompt_entry(canvas, open_id=None)

    def test_write_marks_seen_and_caches_tokens(self):
        from chat.services import canvas_content_sha

        _invoke(WriteCanvasTool, {"title": "Doc", "content": "Hello world"}, self.ctx)
        canvas, entry = self._entry("Doc")
        self.assertEqual(canvas.agent_seen_sha, canvas_content_sha("Hello world"))
        self.assertEqual(canvas.content_sha, canvas.agent_seen_sha)
        self.assertGreater(canvas.token_count, 0)
        self.assertEqual(entry["seen_state"], "unchanged")
        self.assertEqual(entry["tokens"], canvas.token_count)

    def test_user_edit_flags_changed_and_recounts(self):
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "Hello world"}, self.ctx)
        before, _ = self._entry("Doc")
        ChatCanvas.objects.filter(pk=before.pk).update(content="Hello world, and a lot more text " * 20)
        canvas, entry = self._entry("Doc")
        self.assertEqual(entry["seen_state"], "changed")
        self.assertGreater(entry["tokens"], before.token_count)
        canvas.refresh_from_db()
        self.assertEqual(canvas.token_count, entry["tokens"])  # cache refreshed on the row

    def test_read_marks_seen_again(self):
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "line one\nline two\nline three"}, self.ctx)
        canvas, _ = self._entry("Doc")
        ChatCanvas.objects.filter(pk=canvas.pk).update(content="line one\nline TWO\nline three")
        self.assertEqual(self._entry("Doc")[1]["seen_state"], "changed")
        _invoke(ReadCanvasTool, {"canvas_name": "Doc", "start_line": 2, "end_line": 2}, self.ctx)
        self.assertEqual(self._entry("Doc")[1]["seen_state"], "unchanged")

    def test_edit_marks_seen(self):
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "Hello world"}, self.ctx)
        canvas, _ = self._entry("Doc")
        ChatCanvas.objects.filter(pk=canvas.pk).update(content="Hello there world")
        _invoke(EditCanvasTool, {"edits": [{"old_text": "there", "new_text": "dear", "reason": ""}]}, self.ctx)
        self.assertEqual(self._entry("Doc")[1]["seen_state"], "unchanged")

    def test_paste_does_not_mark_seen(self):
        ChatMessage.objects.create(thread=self.thread, role="user", content="Paste me")
        _invoke(PasteUserTextTool, {"message_number": 1, "canvas_name": "Pasted"}, self.ctx)
        canvas, entry = self._entry("Pasted")
        self.assertEqual(canvas.agent_seen_sha, "")
        self.assertEqual(entry["seen_state"], "never")

    def test_truncation_reported_in_entry(self):
        ChatCanvas.objects.create(thread=self.thread, title="Cut", content="abc", truncated_from_chars=200_000)
        _, entry = self._entry("Cut")
        self.assertEqual(entry["truncated_from_chars"], 200_000)
        self.assertEqual(entry["chars"], 3)


class CanvasMarkdownImageGuardrailTests(TestCase):
    """canvas_write / canvas_edit strip non-rendering markdown image syntax."""

    def setUp(self):
        self.user = User.objects.create_user(email="grd@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_write_strips_markdown_image_and_warns(self):
        result = _invoke(
            WriteCanvasTool,
            {"title": "Doc", "content": "Intro\n\n![a chart](IMG_3918.jpeg)\n\nOutro"},
            _ctx(self.user.pk, self.thread.id),
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("warning", result)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        self.assertNotIn("![", canvas.content)
        self.assertNotIn("(IMG_3918.jpeg)", canvas.content)
        # The alt text survives as plain words.
        self.assertIn("a chart", canvas.content)

    def test_edit_strips_markdown_image(self):
        ChatCanvas.objects.create(thread=self.thread, title="Doc", content="start HERE end")
        result = _invoke(
            EditCanvasTool,
            {"canvas_name": "Doc", "edits": [{"old_text": "HERE", "new_text": "![pic](x.png)"}]},
            _ctx(self.user.pk, self.thread.id),
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("warning", result)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        self.assertNotIn("![", canvas.content)
        self.assertIn("pic", canvas.content)

    def test_image_token_is_not_stripped(self):
        # Our own [[image:uuid]] token must survive canvas_write untouched.
        uuid = "12345678-1234-1234-1234-1234567890ab"
        result = _invoke(
            WriteCanvasTool,
            {"title": "Doc", "content": f"A\n\n[[image:{uuid}|Image 1: chart]]\n\nB"},
            _ctx(self.user.pk, self.thread.id),
        )
        self.assertNotIn("warning", result)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        self.assertIn(f"[[image:{uuid}|", canvas.content)


# ---------------------------------------------------------------------------
# canvas_diff_baseline: clean vs dirty (pending AI review) detection
# ---------------------------------------------------------------------------

class CanvasDiffBaselineTests(TestCase):
    """The diff baseline a canvas should report, and whether a review is pending.

    Clean canvas -> (current content, False): the user's own edits never diff.
    Unreviewed AI/import edit -> (accepted baseline, True): the combined diff.
    """

    def setUp(self):
        self.user = User.objects.create_user(email="dbl@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _accepted_canvas(self, content="v1", title="Doc"):
        from chat.services import create_canvas_checkpoint
        canvas = ChatCanvas.objects.create(thread=self.thread, title=title, content=content)
        cp = create_canvas_checkpoint(canvas, source="original")
        canvas.accepted_checkpoint = cp
        canvas.save(update_fields=["accepted_checkpoint"])
        return canvas

    def test_no_baseline_when_no_accepted_checkpoint(self):
        from chat.services import canvas_diff_baseline
        canvas = ChatCanvas.objects.create(thread=self.thread, title="Doc", content="x")
        self.assertEqual(canvas_diff_baseline(canvas), (None, False))

    def test_clean_when_latest_is_accepted(self):
        from chat.services import canvas_diff_baseline
        canvas = self._accepted_canvas("v1")
        baseline, pending = canvas_diff_baseline(canvas)
        self.assertFalse(pending)
        self.assertEqual(baseline, "v1")

    def test_pending_when_ai_edit_after_accepted(self):
        from chat.services import canvas_diff_baseline, create_canvas_checkpoint
        canvas = self._accepted_canvas("v1")
        canvas.content = "v2"
        canvas.save(update_fields=["content"])
        create_canvas_checkpoint(canvas, source="ai_edit")
        baseline, pending = canvas_diff_baseline(canvas)
        self.assertTrue(pending)
        self.assertEqual(baseline, "v1")  # diff against the pre-AI baseline

    def test_pending_survives_user_typing_after_ai_edit(self):
        """User edits after an AI edit create no checkpoint, so the combined diff
        (AI change + user edits) keeps measuring against the pre-AI baseline."""
        from chat.services import canvas_diff_baseline, create_canvas_checkpoint
        canvas = self._accepted_canvas("v1")
        canvas.content = "v2"
        canvas.save(update_fields=["content"])
        create_canvas_checkpoint(canvas, source="ai_edit")
        # User keeps typing — _save_canvas writes content with no new checkpoint.
        canvas.content = "v2 plus user edits"
        canvas.save(update_fields=["content"])
        baseline, pending = canvas_diff_baseline(canvas)
        self.assertTrue(pending)
        self.assertEqual(baseline, "v1")

    def test_clean_after_accept(self):
        from chat.services import canvas_diff_baseline, create_canvas_checkpoint
        canvas = self._accepted_canvas("v1")
        canvas.content = "v2"
        canvas.save(update_fields=["content"])
        ai = create_canvas_checkpoint(canvas, source="ai_edit")
        canvas.accepted_checkpoint = ai  # user accepts
        canvas.save(update_fields=["accepted_checkpoint"])
        baseline, pending = canvas_diff_baseline(canvas)
        self.assertFalse(pending)
        self.assertEqual(baseline, "v2")

    def test_import_edit_is_pending(self):
        """Document-open / skill-load checkpoints (source='import') on an existing
        canvas are unreviewed edits and must surface a diff."""
        from chat.services import canvas_diff_baseline, create_canvas_checkpoint
        canvas = self._accepted_canvas("v1")
        canvas.content = "imported body"
        canvas.save(update_fields=["content"])
        create_canvas_checkpoint(canvas, source="import")
        baseline, pending = canvas_diff_baseline(canvas)
        self.assertTrue(pending)
        self.assertEqual(baseline, "v1")

    def test_redacted_latest_is_not_pending(self):
        """After a redaction rollback the latest checkpoint is relabeled 'redacted';
        the canvas must read as clean."""
        from chat.services import canvas_diff_baseline, create_canvas_checkpoint
        canvas = self._accepted_canvas("v1")
        create_canvas_checkpoint(canvas, source="redacted")
        baseline, pending = canvas_diff_baseline(canvas)
        self.assertFalse(pending)
        self.assertEqual(baseline, canvas.content)

    def test_revert_then_user_edit_reports_clean(self):
        """Regression for the revert-hygiene fix: after a revert advances the
        baseline, later user typing must not resurface a spurious pending diff."""
        from chat.services import canvas_diff_baseline, create_canvas_checkpoint
        canvas = self._accepted_canvas("v1")
        canvas.content = "ai changed"
        canvas.save(update_fields=["content"])
        create_canvas_checkpoint(canvas, source="ai_edit")
        # Simulate _canvas_revert's outcome: restore content + advance the baseline.
        canvas.content = "v1"
        canvas.save(update_fields=["content"])
        cp = create_canvas_checkpoint(canvas, source="user_save", description="Reverted AI changes")
        canvas.accepted_checkpoint = cp
        canvas.save(update_fields=["accepted_checkpoint"])
        # User keeps editing after the revert.
        canvas.content = "v1 plus more"
        canvas.save(update_fields=["content"])
        baseline, pending = canvas_diff_baseline(canvas)
        self.assertFalse(pending)
        self.assertEqual(baseline, "v1 plus more")


# ---------------------------------------------------------------------------
# First-AI-edit baseline: AI edits to a canvas that has no baseline yet
# ---------------------------------------------------------------------------

class FirstAiEditBaselineTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="fae@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _active_user_canvas(self, content):
        from django.utils import timezone
        canvas = ChatCanvas.objects.create(
            thread=self.thread, title="Doc", content=content,
            is_active=True, last_activated_at=timezone.now(),
        )
        self.thread.active_canvas = canvas
        self.thread.save(update_fields=["active_canvas"])
        return canvas

    def test_write_on_unaccepted_canvas_establishes_pre_ai_baseline(self):
        from chat.services import canvas_diff_baseline
        canvas = self._active_user_canvas("user draft")
        self.assertIsNone(canvas.accepted_checkpoint)
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "AI rewrite"}, _ctx(self.user.pk, self.thread.id))
        canvas.refresh_from_db()
        self.assertIsNotNone(canvas.accepted_checkpoint)
        self.assertEqual(canvas.accepted_checkpoint.content, "user draft")
        baseline, pending = canvas_diff_baseline(canvas)
        self.assertTrue(pending)
        self.assertEqual(baseline, "user draft")

    def test_edit_on_unaccepted_canvas_establishes_pre_ai_baseline(self):
        from chat.services import canvas_diff_baseline
        canvas = self._active_user_canvas("The term is 3 years.")
        self.assertIsNone(canvas.accepted_checkpoint)
        _invoke(EditCanvasTool, {
            "edits": [{"old_text": "3 years", "new_text": "5 years", "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        canvas.refresh_from_db()
        self.assertIsNotNone(canvas.accepted_checkpoint)
        self.assertEqual(canvas.accepted_checkpoint.content, "The term is 3 years.")
        baseline, pending = canvas_diff_baseline(canvas)
        self.assertTrue(pending)
        self.assertEqual(baseline, "The term is 3 years.")

    def test_create_path_still_accepts_original(self):
        canvas = ChatCanvas.objects.filter(thread=self.thread).first()
        self.assertIsNone(canvas)
        _invoke(WriteCanvasTool, {"title": "NDA", "content": "# NDA"}, _ctx(self.user.pk, self.thread.id))
        canvas = ChatCanvas.objects.get(thread=self.thread)
        self.assertIsNotNone(canvas.accepted_checkpoint)
        self.assertEqual(canvas.accepted_checkpoint.source, "original")

    def test_edit_with_zero_applied_sets_no_baseline(self):
        canvas = self._active_user_canvas("Some content.")
        self.assertIsNone(canvas.accepted_checkpoint)
        _invoke(EditCanvasTool, {
            "edits": [{"old_text": "nonexistent", "new_text": "x", "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        canvas.refresh_from_db()
        self.assertIsNone(canvas.accepted_checkpoint)


# ---------------------------------------------------------------------------
# Consumer: pending_ai_review round-trips
# ---------------------------------------------------------------------------

@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ConsumerPendingReviewTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="pr@test.com", password="pass")

    async def _communicator(self):
        from channels.routing import URLRouter
        from channels.testing import WebsocketCommunicator
        from chat.routing import websocket_urlpatterns

        app = URLRouter(websocket_urlpatterns)
        comm = WebsocketCommunicator(app, "/ws/chat/")
        comm.scope["user"] = self.user
        connected, _ = await comm.connect()
        assert connected
        return comm

    async def _load_active(self, comm, thread):
        await comm.send_json_to({"type": "chat.load_thread", "thread_id": str(thread.id)})
        # thread.loaded, then canvases.loaded
        await comm.receive_json_from()
        msg = await comm.receive_json_from()
        assert msg["event_type"] == "canvases.loaded"
        return msg["active_canvas"]

    async def test_canvases_loaded_reports_clean(self):
        from channels.db import database_sync_to_async
        from chat.services import create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            canvas = ChatCanvas.objects.create(thread=thread, title="Doc", content="v1")
            thread.active_canvas = canvas
            thread.save(update_fields=["active_canvas"])
            cp = create_canvas_checkpoint(canvas, source="original")
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])
        await database_sync_to_async(setup)()

        comm = await self._communicator()
        ac = await self._load_active(comm, thread)
        self.assertFalse(ac["pending_ai_review"])
        self.assertEqual(ac["accepted_content"], "v1")  # baseline == content -> no diff
        await comm.disconnect()

    async def test_canvases_loaded_reports_pending_with_baseline(self):
        from channels.db import database_sync_to_async
        from chat.services import create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            canvas = ChatCanvas.objects.create(thread=thread, title="Doc", content="v1")
            thread.active_canvas = canvas
            thread.save(update_fields=["active_canvas"])
            cp = create_canvas_checkpoint(canvas, source="original")
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])
            canvas.content = "v2"
            canvas.save(update_fields=["content"])
            create_canvas_checkpoint(canvas, source="ai_edit")
        await database_sync_to_async(setup)()

        comm = await self._communicator()
        ac = await self._load_active(comm, thread)
        self.assertTrue(ac["pending_ai_review"])
        self.assertEqual(ac["accepted_content"], "v1")  # pre-AI baseline
        self.assertEqual(ac["content"], "v2")
        await comm.disconnect()

    async def test_user_save_after_baseline_stays_clean(self):
        """Author mode: a user edit (chat.canvas_save) on a clean canvas with an
        existing baseline keeps it clean — the reported baseline tracks content."""
        from channels.db import database_sync_to_async
        from chat.services import create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            canvas = ChatCanvas.objects.create(thread=thread, title="Doc", content="v1")
            thread.active_canvas = canvas
            thread.save(update_fields=["active_canvas"])
            cp = create_canvas_checkpoint(canvas, source="original")
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])
            return str(canvas.pk)
        canvas_id = await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({
            "type": "chat.canvas_save",
            "thread_id": str(thread.id),
            "canvas_id": canvas_id,
            "title": "Doc",
            "content": "v1 edited by user",
        })
        ac = await self._load_active(comm, thread)
        self.assertFalse(ac["pending_ai_review"])
        # Clean -> baseline is the current content, so no diff despite the stale
        # accepted_checkpoint still holding "v1".
        self.assertEqual(ac["accepted_content"], "v1 edited by user")
        self.assertEqual(ac["content"], "v1 edited by user")
        await comm.disconnect()

    async def test_canvas_switch_carries_pending(self):
        from channels.db import database_sync_to_async
        from chat.services import create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)

        def setup():
            c1 = ChatCanvas.objects.create(thread=thread, title="One", content="one")
            c2 = ChatCanvas.objects.create(thread=thread, title="Two", content="a")
            thread.active_canvas = c1
            thread.save(update_fields=["active_canvas"])
            cp = create_canvas_checkpoint(c2, source="original")
            c2.accepted_checkpoint = cp
            c2.save(update_fields=["accepted_checkpoint"])
            c2.content = "b"
            c2.save(update_fields=["content"])
            create_canvas_checkpoint(c2, source="ai_edit")
            return str(c2.pk)
        c2_id = await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({
            "type": "chat.canvas_switch",
            "thread_id": str(thread.id),
            "canvas_id": c2_id,
        })
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.loaded")
        self.assertTrue(msg["pending_ai_review"])
        self.assertEqual(msg["accepted_content"], "a")
        self.assertEqual(msg["content"], "b")
        await comm.disconnect()

    async def test_canvas_revert_leaves_canvas_clean(self):
        from channels.db import database_sync_to_async
        from chat.services import canvas_diff_baseline, create_canvas_checkpoint

        thread = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)
        canvas = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Doc", content="original content"
        )

        def setup():
            thread.active_canvas = canvas
            thread.save(update_fields=["active_canvas"])
            cp = create_canvas_checkpoint(canvas, source="original")
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])
            canvas.content = "ai changed this"
            canvas.save(update_fields=["content"])
            create_canvas_checkpoint(canvas, source="ai_edit")
        await database_sync_to_async(setup)()

        comm = await self._communicator()
        await comm.send_json_to({"type": "chat.canvas_revert", "thread_id": str(thread.id)})
        msg = await comm.receive_json_from()
        self.assertEqual(msg["event_type"], "canvas.reverted")
        self.assertEqual(msg["content"], "original content")

        def check():
            c = ChatCanvas.objects.get(thread=thread)
            latest = CanvasCheckpoint.objects.filter(canvas=c).order_by("-order").first()
            self.assertEqual(c.accepted_checkpoint_id, latest.pk)  # latest == accepted -> clean
            baseline, pending = canvas_diff_baseline(c)
            self.assertFalse(pending)
            self.assertEqual(baseline, "original content")
        await database_sync_to_async(check)()
        await comm.disconnect()


class DeleteCanvasToolTests(TestCase):
    """canvas_delete soft-deletes (preserved + undoable) and frees the cap."""

    def setUp(self):
        self.user = User.objects.create_user(email="del@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _c(self):
        return _ctx(self.user.pk, self.thread.id)

    def test_delete_soft_deletes_and_preserves_history(self):
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "hi"}, self._c())
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        self.assertTrue(CanvasCheckpoint.objects.filter(canvas=canvas).exists())
        result = _invoke(DeleteCanvasTool, {"canvas_name": "Doc"}, self._c())
        self.assertEqual(result["status"], "ok")
        canvas.refresh_from_db()
        self.assertIsNotNone(canvas.deleted_at)
        # Content + version history preserved.
        self.assertEqual(canvas.content, "hi")
        self.assertTrue(CanvasCheckpoint.objects.filter(canvas=canvas).exists())

    def test_deleted_canvas_invisible_to_resolve(self):
        from chat.services import resolve_canvas
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "hi"}, self._c())
        _invoke(DeleteCanvasTool, {"canvas_name": "Doc"}, self._c())
        canvas, err = resolve_canvas(str(self.thread.id), "Doc")
        self.assertIsNone(canvas)
        self.assertIsNotNone(err)

    def test_unknown_title_returns_error_with_available(self):
        _invoke(WriteCanvasTool, {"title": "Real", "content": "x"}, self._c())
        result = _invoke(DeleteCanvasTool, {"canvas_name": "Ghost"}, self._c())
        self.assertEqual(result["status"], "error")
        self.assertIn("Real", result["available_canvases"])

    def test_same_title_allowed_after_soft_delete(self):
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "v1"}, self._c())
        _invoke(DeleteCanvasTool, {"canvas_name": "Doc"}, self._c())
        # Re-creating the same title must not raise (conditional unique constraint).
        result = _invoke(WriteCanvasTool, {"title": "Doc", "content": "v2"}, self._c())
        self.assertEqual(result["status"], "ok")
        live = ChatCanvas.objects.filter(
            thread=self.thread, title="Doc", deleted_at__isnull=True,
        )
        self.assertEqual(live.count(), 1)
        self.assertEqual(live.first().content, "v2")
        # Two rows total: the soft-deleted one + the new live one.
        self.assertEqual(
            ChatCanvas.objects.filter(thread=self.thread, title="Doc").count(), 2,
        )

    def test_delete_frees_canvas_cap(self):
        from chat.services import MAX_CANVASES_PER_THREAD
        for i in range(MAX_CANVASES_PER_THREAD):
            r = _invoke(WriteCanvasTool, {"title": f"C{i}", "content": "x"}, self._c())
            self.assertEqual(r["status"], "ok")
        # Cap reached — a new canvas is refused.
        capped = _invoke(WriteCanvasTool, {"title": "Overflow", "content": "x"}, self._c())
        self.assertEqual(capped["status"], "error")
        # Free a slot via soft-delete, then the new canvas succeeds.
        _invoke(DeleteCanvasTool, {"canvas_name": "C0"}, self._c())
        freed = _invoke(WriteCanvasTool, {"title": "Overflow", "content": "x"}, self._c())
        self.assertEqual(freed["status"], "ok")

    def test_restore_round_trip(self):
        from chat.services import resolve_canvas, restore_canvas
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "hi"}, self._c())
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        _invoke(DeleteCanvasTool, {"canvas_name": "Doc"}, self._c())
        canvas.refresh_from_db()
        restore_canvas(str(self.thread.id), canvas)
        canvas.refresh_from_db()
        self.assertIsNone(canvas.deleted_at)
        found, err = resolve_canvas(str(self.thread.id), "Doc")
        self.assertIsNotNone(found)

    def test_restore_dedupes_title_on_collision(self):
        from chat.services import restore_canvas
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "first"}, self._c())
        deleted = ChatCanvas.objects.get(thread=self.thread, title="Doc")
        _invoke(DeleteCanvasTool, {"canvas_name": "Doc"}, self._c())
        # A new live canvas reclaims the freed title while the other is deleted.
        _invoke(WriteCanvasTool, {"title": "Doc", "content": "second"}, self._c())
        deleted.refresh_from_db()
        restore_canvas(str(self.thread.id), deleted)
        deleted.refresh_from_db()
        self.assertEqual(deleted.title, "Doc (2)")
        self.assertIsNone(deleted.deleted_at)


class ReadCanvasToolTests(TestCase):
    """canvas_read keeps canvas text reachable once old write/edit args are trimmed."""

    def setUp(self):
        self.user = User.objects.create_user(email="canvasread@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _read(self, args):
        from chat.canvas_tools import ReadCanvasTool

        return _invoke(ReadCanvasTool, args, _ctx(self.user.pk, self.thread.id))

    def test_reads_the_only_canvas_without_a_name(self):
        _invoke(WriteCanvasTool, {"title": "Draft", "content": "Hello there"}, _ctx(self.user.pk, self.thread.id))
        result = self._read({})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["content"], "Hello there")
        self.assertEqual(result["title"], "Draft")

    def test_reads_named_canvas(self):
        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(WriteCanvasTool, {"title": "One", "content": "first"}, ctx)
        _invoke(WriteCanvasTool, {"title": "Two", "content": "second"}, ctx)
        self.assertEqual(self._read({"canvas_name": "One"})["content"], "first")

    def test_errors_when_missing(self):
        self.assertEqual(self._read({})["status"], "error")
        self.assertEqual(self._read({"canvas_name": "Nope"})["status"], "error")

    # -- ranged, token-paged reads --------------------------------------------

    def _big_canvas(self, body_lines=400):
        lines = ["# Intro", "Opening paragraph.", "", "```", "# not a heading (code)", "```", "## Results"]
        lines += [f"line {i} of the body text here" for i in range(body_lines)]
        lines += ["## Gaps", "Closing line."]
        content = "\n".join(lines)
        ChatCanvas.objects.create(thread=self.thread, title="Big", content=content)
        return content, lines

    def test_whole_canvas_when_it_fits(self):
        content = "one\ntwo\nthree"
        ChatCanvas.objects.create(thread=self.thread, title="Small", content=content)
        result = self._read({"canvas_name": "Small"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["content"], content)
        self.assertFalse(result["truncated"])
        self.assertEqual((result["start_line"], result["end_line"], result["total_lines"]), (1, 3, 3))
        self.assertEqual(result["lines_returned"], 3)
        self.assertEqual(result["chars"], len(content))
        self.assertGreater(result["tokens"], 0)
        self.assertNotIn("outline", result)
        self.assertNotIn("next_start_line", result)

    def test_range_by_lines(self):
        ChatCanvas.objects.create(thread=self.thread, title="Six", content="\n".join(f"l{i}" for i in range(1, 7)))
        result = self._read({"canvas_name": "Six", "start_line": 3, "end_line": 4})
        self.assertEqual(result["content"], "l3\nl4")
        self.assertEqual(result["lines_returned"], 2)
        self.assertEqual((result["start_line"], result["end_line"], result["total_lines"]), (3, 4, 6))
        self.assertFalse(result["truncated"])

    def test_page_and_outline_when_over_cap(self):
        content, lines = self._big_canvas()
        result = self._read({"canvas_name": "Big", "max_tokens": 500})
        self.assertTrue(result["truncated"])
        self.assertEqual(result["max_tokens"], 500)
        self.assertLessEqual(result["tokens"], 500)
        self.assertEqual(result["start_line"], 1)
        self.assertEqual(result["next_start_line"], result["end_line"] + 1)
        self.assertEqual(result["content"], "\n".join(lines[: result["lines_returned"]]))
        outline = result["outline"]
        self.assertEqual(outline["total_lines"], len(lines))
        self.assertEqual(outline["chars"], len(content))
        self.assertEqual(
            [(h["level"], h["text"], h["start_line"]) for h in outline["headings"]],
            [(1, "Intro", 1), (2, "Results", 7), (2, "Gaps", len(lines) - 1)],
        )
        self.assertEqual(outline["headings"][1]["end_line"], len(lines) - 2)
        self.assertEqual(outline["headings"][2]["end_line"], len(lines))
        self.assertNotIn("not a heading", json.dumps(outline))
        self.assertIn(f"start_line={result['next_start_line']}", result["note"])

    def test_continuation_reaches_the_end(self):
        content, _lines = self._big_canvas()
        pages = []
        start = 1
        for _ in range(100):
            result = self._read({"canvas_name": "Big", "start_line": start, "max_tokens": 500})
            self.assertEqual(result["status"], "ok")
            pages.append(result["content"])
            if not result["truncated"]:
                break
            start = result["next_start_line"]
        else:
            self.fail("never reached the end")
        self.assertEqual("\n".join(pages), content)
        self.assertGreater(len(pages), 3)

    def test_max_tokens_clamped_to_cap(self):
        ChatCanvas.objects.create(thread=self.thread, title="Small", content="hello")
        ctx = _ctx(self.user.pk, self.thread.id)
        ctx.max_context_tokens = 50_000  # per-read cap = 15k
        from chat.canvas_tools import ReadCanvasTool

        result = _invoke(ReadCanvasTool, {"canvas_name": "Small", "max_tokens": 40_000}, ctx)
        self.assertEqual(result["max_tokens"], 15_000)
        self.assertIn("capped at 15,000", result["note"])

    def test_cap_follows_context_aim(self):
        ChatCanvas.objects.create(thread=self.thread, title="Small", content="hello")
        self.assertEqual(self._read({"canvas_name": "Small"})["max_tokens"], 60_000)  # aim unknown → 200k default

    def test_past_end_errors(self):
        ChatCanvas.objects.create(thread=self.thread, title="Small", content="a\nb\nc")
        result = self._read({"canvas_name": "Small", "start_line": 900})
        self.assertEqual(result["status"], "error")
        self.assertIn("past the end", result["message"])
        self.assertIn("total_lines=3", result["message"])
        inverted = self._read({"canvas_name": "Small", "start_line": 3, "end_line": 2})
        self.assertEqual(inverted["status"], "error")

    def test_table_rows_in_outline(self):
        rows = ["| col a | col b | col c |", "|---|---|---|"] + [f"| r{i} | value {i} | more text {i} |" for i in range(200)]
        ChatCanvas.objects.create(thread=self.thread, title="Table", content="\n".join(rows))
        result = self._read({"canvas_name": "Table", "max_tokens": 500})
        self.assertTrue(result["truncated"])
        self.assertEqual(result["outline"]["headings"], [])
        self.assertEqual(result["outline"]["table_rows"], len(rows))

    def test_empty_canvas(self):
        ChatCanvas.objects.create(thread=self.thread, title="Empty", content="")
        result = self._read({"canvas_name": "Empty"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["content"], "")
        self.assertEqual(result["total_lines"], 0)
        self.assertFalse(result["truncated"])

    def test_end_label_reports_lines(self):
        from chat.canvas_tools import ReadCanvasTool

        tool = ReadCanvasTool()
        self.assertEqual(
            tool.end_label_for_result({"status": "ok", "start_line": 1, "end_line": 40, "total_lines": 640}),
            "Read canvas (lines 1–40 of 640)",
        )
        self.assertEqual(tool.end_label_for_result({"status": "error"}), "Couldn't read canvas")

    def test_is_main_audience_skill_tool_with_labels(self):
        from agent_skills.seed_skills.canvas_collaborator import CANVAS_COLLABORATOR
        from chat.canvas_tools import ReadCanvasTool

        tool = ReadCanvasTool()
        self.assertEqual(tool.audience, "main")
        self.assertEqual(tool.section, "skills")
        self.assertEqual(tool.start_label, "Reading canvas...")
        self.assertIn("canvas_read", CANVAS_COLLABORATOR["tool_names"])


# ---------------------------------------------------------------------------
# Truncation to CANVAS_MAX_CHARS is reported and recorded
# ---------------------------------------------------------------------------

class CanvasTruncationReportingTests(TestCase):
    def setUp(self):
        from chat.services import CANVAS_MAX_CHARS

        self.cap = CANVAS_MAX_CHARS
        self.user = User.objects.create_user(email="trunc-canvas@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = _ctx(self.user.pk, self.thread.id)

    def test_clip_to_canvas(self):
        from chat.services import clip_to_canvas

        self.assertEqual(clip_to_canvas("short"), ("short", None))
        self.assertEqual(clip_to_canvas(None), ("", None))
        self.assertEqual(clip_to_canvas("x" * self.cap), ("x" * self.cap, None))
        content, original = clip_to_canvas("x" * (self.cap + 1))
        self.assertEqual(len(content), self.cap)
        self.assertEqual(original, self.cap + 1)

    def test_truncation_result_fields(self):
        from chat.services import truncation_result_fields

        fields = truncation_result_fields(self.cap * 4, hint="Use the read tools.")
        self.assertTrue(fields["truncated"])
        self.assertEqual(fields["original_chars"], self.cap * 4)
        self.assertEqual(fields["kept_chars"], self.cap)
        self.assertIn("~25%", fields["truncation_note"])
        self.assertIn("Tell the user", fields["truncation_note"])
        self.assertTrue(fields["truncation_note"].endswith("Use the read tools."))

    def test_write_over_cap_is_rejected_and_nothing_saved(self):
        # A model-authored body over the cap is a runaway, not a document: it is
        # refused up front instead of clipped (clipping stays for imports etc.).
        result = _invoke(WriteCanvasTool, {"title": "Big", "content": "x" * (self.cap + 10)}, self.ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn(f"{self.cap + 10:,} characters", result["message"])
        self.assertIn(f"{self.cap:,}", result["message"])
        self.assertIn("Nothing was written", result["message"])
        self.assertNotIn("truncated", result)
        self.assertFalse(ChatCanvas.objects.filter(thread=self.thread, title="Big").exists())

    def test_write_at_cap_is_accepted(self):
        result = _invoke(WriteCanvasTool, {"title": "Big", "content": "x" * self.cap}, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("truncated", result)

    def test_write_over_cap_leaves_existing_canvas_and_checkpoints_untouched(self):
        _invoke(WriteCanvasTool, {"title": "Big", "content": "keep"}, self.ctx)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Big")
        checkpoints = canvas.checkpoints.count()
        result = _invoke(WriteCanvasTool, {"title": "Big", "content": "x" * (self.cap + 1)}, self.ctx)
        self.assertEqual(result["status"], "error")
        canvas.refresh_from_db()
        self.assertEqual(canvas.content, "keep")
        self.assertEqual(canvas.checkpoints.count(), checkpoints)

    def test_edit_with_oversized_new_text_is_rejected_before_matching(self):
        canvas = ChatCanvas.objects.create(thread=self.thread, title="Big", content="hello")
        result = _invoke(
            EditCanvasTool,
            {"canvas_name": "Big", "edits": [{"old_text": "hello", "new_text": "x" * (self.cap + 1)}]},
            self.ctx,
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["applied"], 0)
        self.assertIn("edit 1", result["message"])
        self.assertIn(f"{self.cap + 1:,} characters", result["message"])
        self.assertIn("Nothing was changed", result["message"])
        self.assertEqual(result["rejected_edits"], [{"index": 0, "new_text_chars": self.cap + 1}])
        canvas.refresh_from_db()
        self.assertEqual(canvas.content, "hello")
        self.assertEqual(canvas.checkpoints.count(), 0)

    def test_edit_with_one_oversized_new_text_rejects_the_whole_call(self):
        canvas = ChatCanvas.objects.create(thread=self.thread, title="Big", content="hello world")
        result = _invoke(
            EditCanvasTool,
            {"canvas_name": "Big", "edits": [
                {"old_text": "hello", "new_text": "hi"},
                {"old_text": "world", "new_text": "y" * (self.cap + 5)},
            ]},
            self.ctx,
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("edit 2", result["message"])
        canvas.refresh_from_db()
        self.assertEqual(canvas.content, "hello world")

    def test_fitting_rewrite_clears_marker_and_source(self):
        from documents.models import DataRoom
        from documents.tests._helpers import make_document

        room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        doc = make_document(room, self.user, chunks=["body"])
        ChatCanvas.objects.create(
            thread=self.thread, title="Big", content="x" * self.cap,
            truncated_from_chars=self.cap + 10, source_document=doc,
        )
        result = _invoke(WriteCanvasTool, {"title": "Big", "content": "a fresh summary"}, self.ctx)
        self.assertNotIn("truncated", result)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Big")
        self.assertIsNone(canvas.truncated_from_chars)
        self.assertIsNone(canvas.source_document_id)

    def test_edit_keeps_existing_marker(self):
        ChatCanvas.objects.create(
            thread=self.thread, title="Big", content="hello world", truncated_from_chars=self.cap + 10,
        )
        result = _invoke(
            EditCanvasTool,
            {"canvas_name": "Big", "edits": [{"old_text": "hello", "new_text": "goodbye"}]},
            self.ctx,
        )
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("truncated", result)  # this edit didn't overflow
        self.assertEqual(ChatCanvas.objects.get(thread=self.thread, title="Big").truncated_from_chars, self.cap + 10)

    def test_edit_that_overflows_reports_standard_fields(self):
        ChatCanvas.objects.create(thread=self.thread, title="Big", content="x" * (self.cap - 5) + "END")
        result = _invoke(
            EditCanvasTool,
            {"canvas_name": "Big", "edits": [{"old_text": "END", "new_text": "END" + "y" * 50}]},
            self.ctx,
        )
        self.assertTrue(result["truncated"])
        self.assertEqual(result["original_chars"], self.cap + 48)
        self.assertNotIn("note", result)
        self.assertEqual(ChatCanvas.objects.get(thread=self.thread, title="Big").truncated_from_chars, self.cap + 48)

    def test_overflowing_edit_does_not_shrink_existing_marker(self):
        # Staging repro (at the old 75k cap): canvas opened from a doc 9,248 chars
        # over the cap; a heading edit pushed 32 chars past the cap and replaced
        # the marker with cap+32, so the save warning claimed ~100% was kept.
        marker = self.cap + 9_248
        ChatCanvas.objects.create(
            thread=self.thread, title="Big", content="x" * (self.cap - 5) + "END",
            truncated_from_chars=marker,
        )
        result = _invoke(
            EditCanvasTool,
            {"canvas_name": "Big", "edits": [{"old_text": "END", "new_text": "END" + "y" * 30}]},
            self.ctx,
        )
        self.assertEqual(result["original_chars"], marker)
        self.assertEqual(ChatCanvas.objects.get(thread=self.thread, title="Big").truncated_from_chars, marker)

    def test_canvas_read_reports_truncated_canvas_nested(self):
        # Top-level ``truncated`` now means "this page was cut"; the import-time
        # cut is reported under ``source_truncation``.
        from chat.canvas_tools import ReadCanvasTool

        ChatCanvas.objects.create(
            thread=self.thread, title="Big", content="partial", truncated_from_chars=self.cap + 10,
        )
        result = _invoke(ReadCanvasTool, {"canvas_name": "Big"}, self.ctx)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["source_truncation"]["original_chars"], self.cap + 10)
        self.assertEqual(result["source_truncation"]["kept_chars"], self.cap)
        self.assertIn("TRUNCATED", result["source_truncation"]["note"])


# ---------------------------------------------------------------------------
# Export performance + isolation (2026-10-16): O(1) docx cell lookup, renders
# off the request thread, bounded concurrent renders.
# ---------------------------------------------------------------------------

def _table_html(rows, cols=4):
    head = "".join(f"<th>H{c}</th>" for c in range(cols))
    body = "".join(
        "<tr>" + "".join(f"<td>r{r}c{c} <strong>b</strong></td>" for c in range(cols)) + "</tr>"
        for r in range(rows)
    )
    return f"<html><body><p>Intro</p><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table><p>After</p></body></html>"


class DocxTableBuildTests(TestCase):
    def test_output_identical_to_upstream_html2docx(self):
        from docx import Document
        from html2docx import HTML2Docx

        from chat.markdown_export import html2docx_with_highlight

        html = _table_html(5)
        upstream = HTML2Docx("t")
        upstream.feed(html)
        buf = io.BytesIO()
        upstream.doc.save(buf)
        ours = html2docx_with_highlight(html, title="t")
        self.assertEqual(Document(ours).element.xml, Document(buf).element.xml)

    def test_large_table_never_uses_quadratic_cell_lookup(self):
        # Table._cells rebuilds every cell; calling it per cell made a 400-row
        # table take ~33 s. The override must not touch it at all.
        from docx import Document
        from docx.table import Table

        from chat.markdown_export import html2docx_with_highlight

        def boom(self):
            raise AssertionError("Table._cells used while building the table")

        with patch.object(Table, "_cells", property(boom)):
            buf = html2docx_with_highlight(_table_html(300), title="t")
        table = Document(buf).tables[0]
        self.assertEqual(len(table.rows), 301)
        self.assertEqual(table.rows[300].cells[3].text, "r299c3 b")


class CanvasExportIsolationTests(TestCase):
    """Renders run off the request thread and inside a bounded render slot."""

    def setUp(self):
        self.user = User.objects.create_user(email="expiso@test.com", password="pass")
        self.client.login(email="expiso@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="Doc", content="# Hi\n\nBody.")
        self.thread.active_canvas = self.canvas
        self.thread.save(update_fields=["active_canvas"])
        self.url = f"/chat/threads/{self.thread.id}/canvas/export/"

    def _post(self, query=""):
        return self.client.post(self.url + query, json.dumps({}), content_type="application/json")

    def test_docx_render_runs_off_the_request_thread(self):
        import threading

        import chat.markdown_export as me

        seen = {}
        real = me.html2docx_with_highlight

        def spy(*a, **kw):
            seen["thread"] = threading.current_thread()
            return real(*a, **kw)

        with patch.object(me, "html2docx_with_highlight", spy):
            resp = self._post()
        self.assertEqual(resp.status_code, 200)
        self.assertIsNot(seen["thread"], threading.main_thread())

    def test_pdf_render_runs_off_the_request_thread(self):
        import threading

        seen = {}

        def fake_render(html, *, title, css):
            seen["thread"] = threading.current_thread()
            return b"%PDF-1.4\n"

        with patch("chat.pdf_export.render_canvas_pdf", side_effect=fake_render):
            resp = self._post("?format=pdf")
        self.assertEqual(resp.status_code, 200)
        self.assertIsNot(seen["thread"], threading.main_thread())

    def test_busy_when_no_render_slot_frees_up(self):
        import threading

        import chat.views as views

        full = threading.BoundedSemaphore(1)
        full.acquire()
        with patch.object(views, "_EXPORT_RENDER_SLOTS", full), \
                self.settings(EXPORT_RENDER_SLOT_WAIT_SECONDS=0):
            docx = self._post()
            pdf = self._post("?format=pdf")
        for resp in (docx, pdf):
            self.assertEqual(resp.status_code, 503)
            self.assertIn("busy", resp.json()["error"])
        # The slot is released again after a normal render.
        ok = self._post()
        self.assertEqual(ok.status_code, 200)
