"""Tests for Canvas tools, consumer events, and views."""

from __future__ import annotations

import json
import io
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase, override_settings

from chat.canvas_tools import EditCanvasTool, WriteCanvasTool
from chat.models import ChatCanvas, ChatThread
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

    def test_overwrites_on_second_call(self):
        _invoke(WriteCanvasTool, {"title": "Draft 1", "content": "old"}, _ctx(self.user.pk, self.thread.id))
        _invoke(WriteCanvasTool, {"title": "Draft 2", "content": "new"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(ChatCanvas.objects.filter(thread=self.thread).count(), 1)
        canvas = ChatCanvas.objects.get(thread=self.thread)
        self.assertEqual(canvas.title, "Draft 2")
        self.assertEqual(canvas.content, "new")

    def test_returns_content_in_result(self):
        result = _invoke(WriteCanvasTool, {"title": "T", "content": "C"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["content"], "C")
        self.assertEqual(result["title"], "T")

    def test_no_context_returns_error(self):
        tool = WriteCanvasTool()
        result = json.loads(tool.invoke({"title": "T", "content": "C"}))
        self.assertEqual(result["status"], "error")


# ---------------------------------------------------------------------------
# EditCanvasTool tests
# ---------------------------------------------------------------------------

class EditCanvasToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="edit@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _setup_canvas(self, content):
        ChatCanvas.objects.create(thread=self.thread, title="Doc", content=content)

    def test_applies_edit_correctly(self):
        self._setup_canvas("The term is 3 years.")
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "3 years", "new_text": "5 years", "reason": "update term"}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["applied"], 1)
        self.assertIn("5 years", result["content"])
        canvas = ChatCanvas.objects.get(thread=self.thread)
        self.assertIn("5 years", canvas.content)

    def test_skips_unfound_old_text(self):
        self._setup_canvas("Some content here.")
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "not in text", "new_text": "replacement", "reason": "test"}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["applied"], 0)
        self.assertEqual(len(result["failed"]), 1)

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

    def test_returns_error_when_no_canvas(self):
        result = _invoke(EditCanvasTool, {
            "edits": [{"old_text": "x", "new_text": "y", "reason": ""}]
        }, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "error")
        self.assertIn("write_canvas", result["message"])


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
        await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="My Doc", content="hello"
        )

        comm = await self._communicator()
        await comm.send_json_to({"type": "chat.load_thread", "thread_id": str(thread.id)})

        # thread.loaded event
        msg1 = await comm.receive_json_from()
        self.assertEqual(msg1["event_type"], "thread.loaded")

        # canvas.loaded event
        msg2 = await comm.receive_json_from()
        self.assertEqual(msg2["event_type"], "canvas.loaded")
        self.assertEqual(msg2["title"], "My Doc")
        self.assertEqual(msg2["content"], "hello")

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
        self.assertEqual(msg["event_type"], "canvas.loaded")
        self.assertIn("title", msg)
        self.assertIn("content", msg)

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

    @patch("html2docx.html2docx")
    @patch("markdown.markdown")
    def test_export_returns_docx(self, mock_md, mock_h2d):
        mock_md.return_value = "<h1>NDA</h1>"
        buf = MagicMock()
        buf.getvalue.return_value = b"PK fake docx bytes"
        mock_h2d.return_value = buf

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get("Content-Type"),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertIn("My NDA", response.get("Content-Disposition", ""))

    def test_export_404_no_canvas(self):
        other_thread = ChatThread.objects.create(created_by=self.user)
        url = f"/chat/threads/{other_thread.id}/canvas/export/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_export_404_other_user(self):
        other = User.objects.create_user(email="other@test.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other)
        ChatCanvas.objects.create(thread=other_thread, title="X", content="y")
        url = f"/chat/threads/{other_thread.id}/canvas/export/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)


class CanvasImportViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="imp@test.com", password="pass")
        self.client.login(email="imp@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("mammoth.convert_to_markdown")
    def test_import_creates_canvas(self, mock_convert):
        mock_result = MagicMock()
        mock_result.value = "# Converted\n\nContent here."
        mock_convert.return_value = mock_result

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
