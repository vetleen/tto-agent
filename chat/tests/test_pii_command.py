"""Tests for the /pii slash command (ChatConsumer._cmd_pii).

Invokes the consumer handler directly with a mocked ``send`` so we exercise the
real thread-PII query (``_get_thread_pii_keys``) and Markdown formatting without
a full WebSocket round-trip. TransactionTestCase is required because the query
runs in a worker thread (``database_sync_to_async``) and must see committed rows.
"""
import json
from unittest.mock import AsyncMock

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from chat.consumers import ChatConsumer
from chat.models import ChatThread, ThreadChunkUsage
from documents.models import (
    DataRoom,
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentTag,
)

User = get_user_model()


@override_settings(CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}})
class PiiCommandTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="piicmd@example.com", password="pass")
        self.other = User.objects.create_user(email="piiother@example.com", password="pass")
        self.data_room = DataRoom.objects.create(name="PiiRoom", slug="pii-cmd-room", created_by=self.user)

    def _consumer(self, user=None):
        consumer = ChatConsumer()
        consumer.user = user or self.user
        consumer.send = AsyncMock()
        return consumer

    def _doc_with_tags(self, *keys):
        from documents.tests._helpers import make_version
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="d.txt", status=DataRoomDocument.Status.READY,
        )
        version = make_version(doc, chunks=[{"text": "x", "token_count": 1}])
        chunk = version.chunks.first()
        for key in keys:
            DataRoomDocumentTag.objects.create(version=version, key=key, value="true")
        return doc, chunk

    def _thread_using(self, doc, chunk):
        thread = ChatThread.objects.create(created_by=self.user, title="T")
        ThreadChunkUsage.objects.create(thread=thread, chunk=chunk, document=doc)
        return thread

    def _payload(self, consumer):
        return json.loads(consumer.send.call_args.kwargs["text_data"])

    async def test_reports_bucketed_categories(self):
        doc, chunk = await database_sync_to_async(self._doc_with_tags)(
            "pii_ordinary_identity", "pii_special_category",
        )
        thread = await database_sync_to_async(self._thread_using)(doc, chunk)

        consumer = self._consumer()
        await consumer._cmd_pii("", {"thread_id": thread.id})

        payload = self._payload(consumer)
        self.assertEqual(payload["command"], "/pii")
        self.assertEqual(payload["status"], "ok")
        self.assertIn("Ordinary Personal Data", payload["message"])
        self.assertIn("Special Category Data (Art. 9)", payload["message"])

    async def test_empty_when_thread_used_no_pii_docs(self):
        doc, chunk = await database_sync_to_async(self._doc_with_tags)()
        thread = await database_sync_to_async(self._thread_using)(doc, chunk)

        consumer = self._consumer()
        await consumer._cmd_pii("", {"thread_id": thread.id})

        payload = self._payload(consumer)
        self.assertEqual(payload["status"], "ok")
        self.assertIn("No personal data categories were detected", payload["message"])

    async def test_missing_thread_id_errors(self):
        consumer = self._consumer()
        await consumer._cmd_pii("", {})

        payload = self._payload(consumer)
        self.assertEqual(payload["status"], "error")
        self.assertIn("No active thread", payload["message"])

    async def test_other_users_thread_reports_not_found(self):
        doc, chunk = await database_sync_to_async(self._doc_with_tags)("pii_special_category")
        thread = await database_sync_to_async(self._thread_using)(doc, chunk)

        consumer = self._consumer(user=self.other)
        await consumer._cmd_pii("", {"thread_id": thread.id})

        payload = self._payload(consumer)
        self.assertEqual(payload["status"], "error")
        self.assertIn("Thread not found", payload["message"])
