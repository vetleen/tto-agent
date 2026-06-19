"""Tests for same-turn image viewing: the pipeline injection helper and the
show_image tool."""

import tempfile

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from llm.pipelines.simple_chat import SimpleChatPipeline
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest

User = get_user_model()

_MEDIA = tempfile.mkdtemp()


def _req(model, pending):
    ctx = RunContext.create(user_id=1)
    ctx.pending_image_assets = pending
    return ChatRequest(
        messages=[Message(role="user", content="hi")],
        model=model,
        stream=False,
        tools=[],
        context=ctx,
    ), ctx


class AppendPendingImagesTests(TestCase):
    def test_vision_model_gets_native_image_block(self):
        req, ctx = _req("anthropic/claude-opus-4-8", [
            {"asset_id": "", "b64": "AAAA", "media_type": "image/png", "description": "a bar chart"},
        ])
        new_messages = []
        SimpleChatPipeline._append_pending_images(new_messages, req)

        self.assertEqual(len(new_messages), 1)
        msg = new_messages[0]
        self.assertEqual(msg.role, "user")
        self.assertIsInstance(msg.content, list)
        self.assertTrue(any(isinstance(b, dict) and b.get("type") == "image" for b in msg.content))
        # Collector is drained so it isn't re-injected next iteration.
        self.assertEqual(ctx.pending_image_assets, [])

    def test_non_vision_model_gets_text_fallback(self):
        req, ctx = _req("openai/whisper-1", [
            {"asset_id": "", "b64": "AAAA", "media_type": "image/png", "description": "a bar chart"},
        ])
        new_messages = []
        SimpleChatPipeline._append_pending_images(new_messages, req)

        self.assertEqual(len(new_messages), 1)
        msg = new_messages[0]
        self.assertFalse(any(isinstance(b, dict) and b.get("type") == "image" for b in msg.content))
        self.assertTrue(any("no vision" in b.get("text", "") for b in msg.content))

    def test_no_pending_is_noop(self):
        req, ctx = _req("anthropic/claude-opus-4-8", [])
        new_messages = []
        SimpleChatPipeline._append_pending_images(new_messages, req)
        self.assertEqual(new_messages, [])


@override_settings(MEDIA_ROOT=_MEDIA)
class ShowImageToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="siv@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-siv", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="chart.png", mime_type="image/png",
            doc_index=1, status=DataRoomDocument.Status.READY,
        )
        version = DataRoomDocumentVersion.objects.create(
            document=self.doc, parser_type="image", mime_type="image/png",
            native_blob=ContentFile(b"\x89PNG fake-image", name="chart.png"),
        )
        self.doc.current_version = version
        self.doc.save(update_fields=["current_version"])

    def _tool(self, data_room_ids):
        from chat.tools import ShowImageTool

        tool = ShowImageTool()
        tool.set_context(RunContext.create(user_id=self.user.pk, data_room_ids=data_room_ids))
        return tool

    def test_attaches_image_as_document(self):
        tool = self._tool([self.room.pk])
        result = tool._run([1])
        self.assertIn("attached", result.lower())
        pending = tool.context.pending_image_assets
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["media_type"], "image/png")
        self.assertTrue(pending[0]["b64"])

    def test_inaccessible_room_is_denied(self):
        other = User.objects.create_user(email="siv-other@test.com", password="pw")
        room2 = DataRoom.objects.create(name="R2", slug="r2-siv", created_by=other)
        tool = self._tool([room2.pk])  # this user does not own room2
        result = tool._run([1])
        self.assertEqual(tool.context.pending_image_assets, [])
        self.assertNotIn("attached", result.lower())


@override_settings(MEDIA_ROOT=_MEDIA)
class AttachmentMarkersTests(TestCase):
    """The summariser surfaces shared-file markers so compression doesn't
    silently lose that the user shared an image."""

    def test_marks_messages_with_attachments(self):
        from django.core.files.base import ContentFile

        from chat.models import ChatAttachment, ChatMessage, ChatThread
        from chat.services import _attachment_markers

        user = User.objects.create_user(email="amk@test.com", password="pw")
        thread = ChatThread.objects.create(created_by=user)
        msg = ChatMessage.objects.create(thread=thread, role="user", content="look at this")
        att = ChatAttachment.objects.create(
            thread=thread, message=msg, uploaded_by=user,
            file=ContentFile(b"x", name="chart.png"),
            original_filename="chart.png", content_type="image/png", size_bytes=1,
        )
        msg.metadata = {"attachment_ids": [str(att.id)]}
        msg.save(update_fields=["metadata"])

        markers = _attachment_markers([msg])
        self.assertIn(str(msg.id), markers)
        self.assertIn("chart.png", markers[str(msg.id)])

    def test_no_attachments_is_empty(self):
        from chat.models import ChatMessage, ChatThread
        from chat.services import _attachment_markers

        user = User.objects.create_user(email="amk2@test.com", password="pw")
        thread = ChatThread.objects.create(created_by=user)
        msg = ChatMessage.objects.create(thread=thread, role="user", content="hi")
        self.assertEqual(_attachment_markers([msg]), {})
